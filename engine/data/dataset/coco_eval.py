"""
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
COCO evaluator that works in distributed mode.
Mostly copy-paste from https://github.com/pytorch/vision/blob/edfd5a7/references/detection/coco_eval.py
The difference is that there is less copy-pasting from pycocotools
in the end of the file, as python3 can suppress prints with contextlib

Extended with Recall@k and Grounding metrics
"""
import os
import contextlib
import copy
import numpy as np
import torch

from faster_coco_eval import COCO, COCOeval_faster
import faster_coco_eval.core.mask as mask_util
from ...core import register
from ...misc import dist_utils

__all__ = ['CocoEvaluator',]


@register()
class CocoEvaluator(object):
    def __init__(self, coco_gt, iou_types, output_detail=False):
        """
        Args:
            coco_gt: COCO ground truth object
            iou_types: list of iou types ('bbox', 'segm', 'keypoints')
            output_detail: bool, 是否输出详细信息，默认 False 只输出汇总表格
        """
        assert isinstance(iou_types, (list, tuple))
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt : COCO = coco_gt
        self.iou_types = iou_types
        self.output_detail = output_detail

        self.coco_eval = {}
        for iou_type in iou_types:
            self.coco_eval[iou_type] = COCOeval_faster(coco_gt, iouType=iou_type, print_function=print, separate_eval=True)

        self.img_ids = []
        self.eval_imgs = {k: [] for k in iou_types}
        
        # 用于存储扩展指标结果
        self.recall_at_k_results = {}
        self.grounding_metrics = {}
        
        # 用于存储所有预测结果（用于计算 Precision@IoU 和 meanIoU）
        self.all_predictions = {}
        
        # 汇总结果
        self.summary_results = {}
        
        # 每个类别的结果
        self.per_category_results = {}

    @staticmethod
    def _category_name(category, cat_id):
        if isinstance(category, dict):
            for key in ('name', 'category', 'supercategory'):
                value = category.get(key)
                if value:
                    return str(value)
        return str(cat_id)

    def cleanup(self):
        self.coco_eval = {}
        for iou_type in self.iou_types:
            self.coco_eval[iou_type] = COCOeval_faster(self.coco_gt, iouType=iou_type, print_function=print, separate_eval=True)
        self.img_ids = []
        self.eval_imgs = {k: [] for k in self.iou_types}
        self.recall_at_k_results = {}
        self.grounding_metrics = {}
        self.all_predictions = {}
        self.summary_results = {}
        self.per_category_results = {}

    def update(self, predictions):
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)
        
        # 保存预测结果用于后续计算 Grounding 指标
        for img_id, pred in predictions.items():
            if img_id not in self.all_predictions:
                self.all_predictions[img_id] = pred
            else:
                # 合并预测（如果有多次 update）
                for key in pred:
                    if key in self.all_predictions[img_id]:
                        self.all_predictions[img_id][key] = torch.cat(
                            [self.all_predictions[img_id][key], pred[key]], dim=0
                        )
                    else:
                        self.all_predictions[img_id][key] = pred[key]

        for iou_type in self.iou_types:
            results = self.prepare(predictions, iou_type)
            coco_eval = self.coco_eval[iou_type]

            # suppress pycocotools prints
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull):
                    coco_dt = self.coco_gt.loadRes(results) if results else COCO()
                    coco_eval.cocoDt = coco_dt
                    coco_eval.params.imgIds = list(img_ids)
                    coco_eval.evaluate()

            self.eval_imgs[iou_type].append(np.array(coco_eval._evalImgs_cpp).reshape(len(coco_eval.params.catIds), len(coco_eval.params.areaRng), len(coco_eval.params.imgIds)))

    def synchronize_between_processes(self):
        for iou_type in self.iou_types:
            img_ids, eval_imgs = merge(self.img_ids, self.eval_imgs[iou_type])

            coco_eval = self.coco_eval[iou_type]
            coco_eval.params.imgIds = img_ids
            coco_eval._paramsEval = copy.deepcopy(coco_eval.params)
            coco_eval._evalImgs_cpp = eval_imgs
        
        # 同步所有预测结果
        all_predictions_list = dist_utils.all_gather(self.all_predictions)
        merged_predictions = {}
        for preds in all_predictions_list:
            for img_id, pred in preds.items():
                if img_id not in merged_predictions:
                    merged_predictions[img_id] = pred
        self.all_predictions = merged_predictions

    def accumulate(self):
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()

    def summarize(self):
        """
        打印标准 COCO 指标 + Recall@k
        output_detail=False: 只输出汇总表格
        output_detail=True: 输出完整详细信息
        """
        ds_name = getattr(self, "dataset_name", "default")
        
        for iou_type, coco_eval in self.coco_eval.items():
            if self.output_detail:
                print("IoU metric: {}".format(iou_type))
                coco_eval.summarize()
            
            # 提取指标
            if iou_type == 'bbox':
                metrics, per_cat_metrics = self._extract_metrics(coco_eval, iou_type)
                self.summary_results[iou_type] = metrics
                self.per_category_results[iou_type] = per_cat_metrics
        
        # 打印汇总表格
        self._print_summary_table(ds_name)
        
        # 打印每个类别的结果
        self._print_per_category_table(ds_name)
    
    def _extract_metrics(self, coco_eval, iou_type='bbox'):
        """
        从 coco_eval 中提取所需指标
        返回: (汇总指标字典, 每类别指标字典)
        """
        metrics = {}
        per_category_metrics = {}
        
        # ========== AP 指标 ==========
        # coco_eval.eval['precision']: shape = [T, R, K, A, M]
        # T: IoU thresholds (10: 0.5:0.05:0.95)
        # R: recall thresholds (101: 0:0.01:1)
        # K: categories
        # A: areas (4: all, small, medium, large)
        # M: maxDets (3: 1, 10, 100)
        
        precision = coco_eval.eval['precision']
        recall = coco_eval.eval['recall']
        
        # 获取类别信息
        cat_ids = coco_eval.params.catIds
        cat_names = {}
        for cat_id in cat_ids:
            if cat_id in self.coco_gt.cats:
                cat_names[cat_id] = self._category_name(self.coco_gt.cats[cat_id], cat_id)
            else:
                cat_names[cat_id] = str(cat_id)
        
        # 参数索引
        # IoU: 0=0.5, 5=0.75, 0:10=0.5:0.95
        # Area: 0=all, 1=small, 2=medium, 3=large
        # MaxDet: 0=1, 1=10, 2=100
        
        # mAP (IoU=0.50:0.95, maxDet=100, area=all)
        ap_all = precision[:, :, :, 0, 2]  # [T, R, K]
        ap_all = ap_all[ap_all > -1]
        metrics['mAP'] = np.mean(ap_all) * 100 if len(ap_all) > 0 else 0.0
        
        # AP50 (IoU=0.50, maxDet=100, area=all)
        ap50 = precision[0, :, :, 0, 2]  # IoU=0.5
        ap50 = ap50[ap50 > -1]
        metrics['AP50'] = np.mean(ap50) * 100 if len(ap50) > 0 else 0.0
        
        # AP75 (IoU=0.75, maxDet=100, area=all)
        ap75 = precision[5, :, :, 0, 2]  # IoU=0.75
        ap75 = ap75[ap75 > -1]
        metrics['AP75'] = np.mean(ap75) * 100 if len(ap75) > 0 else 0.0
        
        # AP_small (IoU=0.50:0.95, maxDet=100, area=small)
        ap_s = precision[:, :, :, 1, 2]
        ap_s = ap_s[ap_s > -1]
        metrics['AP_S'] = np.mean(ap_s) * 100 if len(ap_s) > 0 else 0.0
        
        # AP_medium (IoU=0.50:0.95, maxDet=100, area=medium)
        ap_m = precision[:, :, :, 2, 2]
        ap_m = ap_m[ap_m > -1]
        metrics['AP_M'] = np.mean(ap_m) * 100 if len(ap_m) > 0 else 0.0
        
        # AP_large (IoU=0.50:0.95, maxDet=100, area=large)
        ap_l = precision[:, :, :, 3, 2]
        ap_l = ap_l[ap_l > -1]
        metrics['AP_L'] = np.mean(ap_l) * 100 if len(ap_l) > 0 else 0.0
        
        # ========== AR 指标 ==========
        # coco_eval.eval['recall']: shape = [T, K, A, M]
        
        # AR100 (IoU=0.50:0.95, maxDet=100, area=all)
        ar100 = recall[:, :, 0, 2]  # [T, K]
        ar100 = ar100[ar100 > -1]
        metrics['AR100'] = np.mean(ar100) * 100 if len(ar100) > 0 else 0.0
        
        # ========== Recall@k (IoU=0.5) ==========
        # Recall@1 (IoU=0.50, maxDet=1, area=all)
        r1 = recall[0, :, 0, 0]  # IoU=0.5, maxDet=1
        r1 = r1[r1 > -1]
        metrics['R@1'] = np.mean(r1) * 100 if len(r1) > 0 else 0.0
        
        # Recall@10 (IoU=0.50, maxDet=10, area=all)
        r10 = recall[0, :, 0, 1]  # IoU=0.5, maxDet=10
        r10 = r10[r10 > -1]
        metrics['R@10'] = np.mean(r10) * 100 if len(r10) > 0 else 0.0
        
        # Recall@100 (IoU=0.50, maxDet=100, area=all)
        r100 = recall[0, :, 0, 2]  # IoU=0.5, maxDet=100
        r100 = r100[r100 > -1]
        metrics['R@100'] = np.mean(r100) * 100 if len(r100) > 0 else 0.0
        
        # ========== 每个类别的指标 ==========
        num_cats = len(cat_ids)
        for k, cat_id in enumerate(cat_ids):
            cat_name = cat_names[cat_id]
            
            # AP50 for this category: precision[0, :, k, 0, 2]
            # IoU=0.5, all recall thresholds, category k, area=all, maxDet=100
            cat_ap50 = precision[0, :, k, 0, 2]
            cat_ap50 = cat_ap50[cat_ap50 > -1]
            cat_ap50_val = np.mean(cat_ap50) * 100 if len(cat_ap50) > 0 else -1.0
            
            # Recall@100 for this category: recall[0, k, 0, 2]
            # IoU=0.5, category k, area=all, maxDet=100
            cat_r100 = recall[0, k, 0, 2]
            cat_r100_val = cat_r100 * 100 if cat_r100 > -1 else -1.0
            
            per_category_metrics[cat_id] = {
                'name': cat_name,
                'AP50': cat_ap50_val,
                'R@100': cat_r100_val,
            }
        
        return metrics, per_category_metrics
    
    def _print_summary_table(self, ds_name):
        """打印简洁的汇总表格"""
        
        # 表头
        header = "| {:^5} | {:^5} | {:^5} | {:^5} | {:^5} | {:^5} | {:^5} | {:^5} | {:^5} | {:^5} |".format(
            "mAP", "AP50", "AP75", "AP_S", "AP_M", "AP_L", "AR100", "R@1", "R@10", "R@100"
        )
        sep_line = "+" + "-"*7 + "+" + "-"*7 + "+" + "-"*7 + "+" + "-"*7 + "+" + "-"*7 + "+" + "-"*7 + "+" + "-"*7 + "+" + "-"*7 + "+" + "-"*7 + "+" + "-"*7 + "+"
        
        print("\n")
        print("=" * len(sep_line))
        title = f"COCO DETECTION RESULTS [{ds_name}]"
        print(f"{title:^{len(sep_line)}}")
        print("=" * len(sep_line))
        print(sep_line)
        print(header)
        print(sep_line)
        
        for iou_type in self.iou_types:
            if iou_type in self.summary_results:
                m = self.summary_results[iou_type]
                row = "| {:>5.2f} | {:>5.2f} | {:>5.2f} | {:>5.2f} | {:>5.2f} | {:>5.2f} | {:>5.2f} | {:>5.2f} | {:>5.2f} | {:>5.2f} |".format(
                    m.get('mAP', 0.0),
                    m.get('AP50', 0.0),
                    m.get('AP75', 0.0),
                    m.get('AP_S', 0.0),
                    m.get('AP_M', 0.0),
                    m.get('AP_L', 0.0),
                    m.get('AR100', 0.0),
                    m.get('R@1', 0.0),
                    m.get('R@10', 0.0),
                    m.get('R@100', 0.0),
                )
                print(row)
        
        print(sep_line)
        print("")
    
    def _print_per_category_table(self, ds_name):
        """打印每个类别的 AP50 和 Recall@100 表格"""
        
        for iou_type in self.iou_types:
            if iou_type not in self.per_category_results:
                continue
            
            per_cat = self.per_category_results[iou_type]
            if not per_cat:
                continue
            
            # 确定类别名称的最大长度
            max_name_len = max(len(v['name']) for v in per_cat.values())
            max_name_len = max(max_name_len, 8)  # 至少 8 个字符
            
            # 表头
            header = "| {:^{width}} | {:^8} | {:^8} |".format(
                "Category", "AP50", "R@100", width=max_name_len
            )
            sep_line = "+" + "-"*(max_name_len+2) + "+" + "-"*10 + "+" + "-"*10 + "+"
            
            print("=" * len(sep_line))
            title = f"PER-CATEGORY RESULTS [{ds_name}] (IoU Type: {iou_type})"
            print(f"{title:^{len(sep_line)}}")
            print("=" * len(sep_line))
            print(sep_line)
            print(header)
            print(sep_line)
            
            # 按类别 ID 排序
            for cat_id in sorted(per_cat.keys()):
                cat_info = per_cat[cat_id]
                cat_name = cat_info['name']
                ap50 = cat_info['AP50']
                r100 = cat_info['R@100']
                
                # 处理无效值（-1 表示该类别没有样本）
                ap50_str = f"{ap50:>6.2f}" if ap50 >= 0 else "  N/A "
                r100_str = f"{r100:>6.2f}" if r100 >= 0 else "  N/A "
                
                row = "| {:<{width}} | {:>8} | {:>8} |".format(
                    cat_name, ap50_str, r100_str, width=max_name_len
                )
                print(row)
            
            print(sep_line)
            print("")
    
    def get_results(self, iou_type='bbox'):
        """获取所有指标结果"""
        return self.summary_results.get(iou_type, {})
    
    def get_per_category_results(self, iou_type='bbox'):
        """获取每个类别的指标结果"""
        return self.per_category_results.get(iou_type, {})
    
    def get_recall_at_k(self, iou_type='bbox'):
        """
        获取已计算的 Recall@k 结果
        """
        if iou_type in self.summary_results:
            m = self.summary_results[iou_type]
            return {
                'R@1': m.get('R@1', 0.0),
                'R@10': m.get('R@10', 0.0),
                'R@100': m.get('R@100', 0.0),
            }
        return {}
    
    def get_grounding_metrics(self, iou_type='bbox'):
        """
        获取已计算的 Grounding 指标
        """
        return self.grounding_metrics.get(iou_type, {})
    
    def get_all_metrics(self, iou_type='bbox'):
        """
        获取所有扩展指标
        """
        return self.summary_results.get(iou_type, {})

    def prepare(self, predictions, iou_type):
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        elif iou_type == "segm":
            return self.prepare_for_coco_segmentation(predictions)
        elif iou_type == "keypoints":
            return self.prepare_for_coco_keypoint(predictions)
        else:
            raise ValueError("Unknown iou type {}".format(iou_type))

    def prepare_for_coco_detection(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return coco_results

    def prepare_for_coco_segmentation(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            scores = prediction["scores"]
            labels = prediction["labels"]
            masks = prediction["masks"]

            masks = masks > 0.5

            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            rles = [
                mask_util.encode(np.array(mask[0, :, :, np.newaxis], dtype=np.uint8, order="F"))[0]
                for mask in masks
            ]
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "segmentation": rle,
                        "score": scores[k],
                    }
                    for k, rle in enumerate(rles)
                ]
            )
        return coco_results

    def prepare_for_coco_keypoint(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            keypoints = prediction["keypoints"]
            keypoints = keypoints.flatten(start_dim=1).tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        'keypoints': keypoint,
                        "score": scores[k],
                    }
                    for k, keypoint in enumerate(keypoints)
                ]
            )
        return coco_results


def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)

# 合并多卡结果
def merge(img_ids, eval_imgs):
    all_img_ids = dist_utils.all_gather(img_ids)
    all_eval_imgs = dist_utils.all_gather(eval_imgs)

    merged_img_ids = []
    for p in all_img_ids:
        merged_img_ids.extend(p)

    merged_eval_imgs = []
    for p in all_eval_imgs:
        merged_eval_imgs.extend(p)

    merged_img_ids = np.array(merged_img_ids)
    merged_eval_imgs = np.concatenate(merged_eval_imgs, axis=2).ravel()

    # keep only unique (and in sorted order) images
    merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)

    return merged_img_ids.tolist(), merged_eval_imgs.tolist()