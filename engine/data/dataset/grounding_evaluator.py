#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GroundingEvaluator - 仅计算 Grounding 定位指标（不依赖 COCOeval）
支持输出：
  - Recall@k (k ∈ max_dets)，按 IoU 阈值统计
  - Precision@IoU（Top-1 预测）
  - meanIoU、cmuIoU（RSVG 指标）
  - [NEW] Attr-Align 评测（两阶段判定，论文指标）
"""

import os
import numpy as np
import torch
from collections import defaultdict

from ...core import register
from ...misc import dist_utils

__all__ = ['GroundingEvaluator']


@register()
class GroundingEvaluator(object):
    def __init__(self, coco_gt, iou_types=None, max_dets=None, output_detail=False,
                 attr_thresholds=None, enable_attr_robust_eval=True,
                 vis_failed=False, vis_output_dir='./outputs/failed_cases',
                 img_root=None, vis_max_cases=100, vis_score_key='scores',
                 vis_iou_threshold=0.5):
        """
        Args:
            coco_gt: COCO 风格的 GT（只用到 annotations 和 images 字段）
            iou_types: 占位，保持与原接口兼容（默认 ['bbox']）
            max_dets: list, 计算 Recall@k 时使用的 k 值，默认 [1, 5, 10, 100]
            output_detail: bool, 是否输出详细信息，默认 False 只输出汇总表格
            attr_thresholds: list, Attr-Align 阈值，默认 [0.5, 0.6, 0.7]
            enable_attr_robust_eval: bool, 是否启用 Attr-Align 评测，默认 True
            vis_failed: bool, 是否可视化失败案例（Top-1 IoU < vis_iou_threshold），默认 False
            vis_output_dir: str, 失败案例可视化图像的保存目录，默认 './outputs/failed_cases'
            img_root: str, 图像根目录（用于加载原图），默认 None
            vis_max_cases: int, 最多保存的失败案例数，默认 100
            vis_score_key: str, 判断失败案例时使用的得分键，默认 'scores'
            vis_iou_threshold: float, 低于此 IoU 视为失败案例，默认 0.5
        """
        if iou_types is None:
            iou_types = ['bbox']
        self.iou_types = iou_types
        self.max_dets = max_dets or [1, 5, 10, 100]
        self.output_detail = output_detail
        
        # Attr-Align 评测配置
        self.attr_thresholds = attr_thresholds or [0.5, 0.6, 0.7]
        self.enable_attr_robust_eval = enable_attr_robust_eval

        # 失败案例可视化配置
        self.vis_failed = vis_failed
        self.vis_output_dir = vis_output_dir
        self.img_root = img_root
        self.vis_max_cases = vis_max_cases
        self.vis_score_key = vis_score_key
        self.vis_iou_threshold = vis_iou_threshold

        # 只保存 GT 标注信息
        self.coco_gt = coco_gt

        # 预测缓存
        self.all_predictions = defaultdict(lambda: defaultdict(list))
        self.img_ids = []

        # 评估结果缓存
        self.results = {}
        # 可用的得分字段，默认 caption logits
        self.score_keys = ['scores']
        self._seen_score_keys = set(self.score_keys)

    def cleanup(self):
        self.all_predictions = defaultdict(lambda: defaultdict(list))
        self.img_ids = []
        self.results = {}

    def update(self, predictions):
        """
        predictions: dict[img_id] -> {
            'boxes': Tensor [N,4], 
            'scores': Tensor [N], 
            'labels': Tensor [N],
            'attr_scores_list': Tensor [N, num_attrs] (可选，用于 Attr-Align 评测)
        }
        """
        for img_id, pred in predictions.items():
            self.img_ids.append(img_id)
            for k, v in pred.items():
                # 支持多次调用 update，逐步累积
                if isinstance(v, torch.Tensor):
                    v = v.detach().cpu()
                self.all_predictions[img_id][k].append(v)
                # 记录可用的得分键
                if k.startswith('scores'):
                    self._seen_score_keys.add(k)

    def synchronize_between_processes(self):
        # 分布式同步预测
        all_preds_list = dist_utils.all_gather(dict(self.all_predictions))
        merged = defaultdict(lambda: defaultdict(list))
        for preds in all_preds_list:
            for img_id, pred in preds.items():
                for k, v_list in pred.items():
                    merged[img_id][k].extend(v_list)
        self.all_predictions = merged
        self.img_ids = list(self.all_predictions.keys())

    def accumulate(self):
        # 不需要额外处理，计算在 summarize 里完成
        return

    def summarize(self):
        """
        打印 Grounding 定位指标。
        output_detail=False: 只输出汇总表格
        output_detail=True: 输出完整详细信息
        """
        ds_name = getattr(self, "dataset_name", "default")

        # 确定需要评测的得分键，按固定顺序排列
        score_key_order = ['scores', 'scores_attr', 'scores_avg']
        score_keys = [k for k in score_key_order if k in self._seen_score_keys]
        # 添加其他未在预定义顺序中的得分键
        score_keys += [k for k in sorted(self._seen_score_keys) if k not in score_key_order]

        # 收集所有得分键的指标
        all_metrics = {}
        for score_key in score_keys:
            metrics = self._compute_all_metrics(score_key, verbose=self.output_detail)
            all_metrics[score_key] = metrics

            # 合并结果并带上后缀标记
            for k, v in metrics.items():
                self.results[f"{k}[{score_key}]"] = v

        # 输出汇总表格
        self._print_summary_table(ds_name, score_keys, all_metrics)
        
        # ========== [NEW] Attr-Align 评测 ==========
        if self.enable_attr_robust_eval and self._check_attr_scores_available():
            attr_align_results = self._compute_attr_robust_metrics(verbose=self.output_detail)
            self.results.update(attr_align_results)
            self._print_attr_robust_table(ds_name, attr_align_results)

        # ========== 失败案例可视化 ==========
        if self.vis_failed:
            self._visualize_failed_cases(
                score_key=self.vis_score_key,
                iou_threshold=self.vis_iou_threshold,
            )

    def get_results(self, iou_type='bbox'):
        return self.results

    # ---------------- internal helpers ---------------- #
    def _gather_gt(self):
        gt_map = defaultdict(list)
        for ann in self.coco_gt.dataset['annotations']:
            gt_map[ann['image_id']].append(ann)
        return gt_map
    
    def _check_attr_scores_available(self):
        """检查是否存在属性得分列表 (attr_scores_list 或 attr_mean_for_topk)"""
        for img_id, preds in self.all_predictions.items():
            # 检查新格式字段
            if 'attr_scores_list' in preds and len(preds['attr_scores_list']) > 0:
                return True
            if 'attr_mean_for_topk' in preds and len(preds['attr_mean_for_topk']) > 0:
                return True
        return False

    def _compute_all_metrics(self, score_key='scores', verbose=False):
        """计算所有指标，返回字典"""
        gt_map = self._gather_gt()
        results = {}

        # ========== Recall@k ==========
        iou_thrs = (0.5, 0.75)
        if verbose:
            print(f"\n*** Using score field: {score_key} ***")
            print("\n📊 Recall@k (Top-k Predictions):")
            print("─" * 78)

        for iou_thr in iou_thrs:
            if verbose:
                print(f"\n  IoU Threshold = {iou_thr}:")
                print("  " + "─" * 74)
            for k in self.max_dets:
                recalls = []
                for img_id, gt_anns in gt_map.items():
                    if len(gt_anns) == 0:
                        continue
                    gt_boxes = [ann['bbox'] for ann in gt_anns]
                    preds = self._get_topk_preds(img_id, k, score_key=score_key)
                    recall = self._image_recall(gt_boxes, preds, iou_thr)
                    recalls.append(recall)
                avg_recall = np.mean(recalls) if len(recalls) else 0.0
                results[f'Recall@{k}[IoU={iou_thr}]'] = avg_recall * 100
                if verbose:
                    print(f"    Recall@{k:3d} [IoU={iou_thr}]  = {avg_recall:6.4f} ({avg_recall*100:5.2f}%)")

        # ========== Grounding Metrics (Top-1) ==========
        ious = []
        precision_counters = {0.5: 0, 0.6: 0, 0.7: 0, 0.8: 0, 0.9: 0}
        total_samples = 0
        cum_inter_area = 0.0
        cum_union_area = 0.0

        if verbose:
            print("\n📊 Grounding Metrics (Top-1 Prediction):")
            print("─" * 78)

        for img_id, gt_anns in gt_map.items():
            if len(gt_anns) == 0:
                continue
            gt_bbox = gt_anns[0]['bbox']

            preds = self._get_topk_preds(img_id, 1, score_key=score_key)
            if preds is None or preds['boxes'].shape[0] == 0:
                ious.append(0.0)
                total_samples += 1
                cum_union_area += gt_bbox[2] * gt_bbox[3]
                continue

            pred_bbox = preds['boxes'][0]
            iou, inter_area, union_area = self._compute_iou_with_area(gt_bbox, self._xyxy_to_xywh(pred_bbox))
            ious.append(iou)
            cum_inter_area += inter_area
            cum_union_area += union_area

            for thr in precision_counters:
                if iou >= thr:
                    precision_counters[thr] += 1

            total_samples += 1

        if verbose:
            print("\n  Precision @ Different IoU Thresholds:")
            print("  " + "─" * 74)

        for thr in sorted(precision_counters.keys()):
            precision = (precision_counters[thr] / total_samples * 100) if total_samples > 0 else 0.0
            results[f'Pr@{thr}'] = precision
            if verbose:
                print(f"    Pr @ IoU={thr:3.1f}  = {precision/100:6.4f} ({precision:5.2f}%)")

        mean_iou = np.mean(ious) * 100 if len(ious) else 0.0
        cmu_iou = (cum_inter_area / cum_union_area * 100) if cum_union_area > 0 else 0.0

        results['meanIoU'] = mean_iou
        results['cmuIoU'] = cmu_iou

        # Acc_Top1 = Recall@1[IoU=0.5], Acc_Top5 = Recall@5[IoU=0.5]
        results['Acc_Top1'] = results.get('Recall@1[IoU=0.5]', 0.0)
        results['Acc_Top5'] = results.get('Recall@5[IoU=0.5]', 0.0)

        if verbose:
            print("\n  IoU Statistics:")
            print("  " + "─" * 74)
            print(f"    meanIoU       = {mean_iou/100:6.4f} ({mean_iou:5.2f}%)")
            print(f"    cmuIoU        = {cmu_iou/100:6.4f} ({cmu_iou:5.2f}%)  ✅ OPT-RSVG method")
            print(f"      ↳ cumInterArea = {cum_inter_area:.2f}")
            print(f"      ↳ cumUnionArea = {cum_union_area:.2f}")

        return results

    def _compute_attr_robust_metrics(self, score_key='scores', verbose=False):
        """
        [NEW] 计算论文 Attr-Align 指标
        
        两阶段判定方案：
        Step1: 传统 RSVG 评测 - IoU > 0.5 定位正确
        Step2: 在 Step1 正确的基础上，要求与所有属性的平均相似度 >= τ
        
        同时满足 Step1 且 Step2，计入 Attr-Align@τ
        
        Returns:
            dict: {
                'AttrAlign@0.5': float,  # τ=0.5 时的准确率
                'AttrAlign@0.6': float,  # τ=0.6 时的准确率
                'AttrAlign@0.7': float,  # τ=0.7 时的准确率
                'MeanAttrScore': float,   # 定位正确样本的平均属性相似度
            }
        """
        gt_map = self._gather_gt()
        results = {}
        
        # 统计计数器
        total_samples = 0
        step1_correct = 0  # IoU > 0.5 的样本数
        attr_align_counters = {thr: 0 for thr in self.attr_thresholds}
        
        # 收集定位正确样本的属性得分
        correct_attr_scores = []
        all_attr_scores = []
        
        if verbose:
            print("\n" + "="*78)
            print("📊 Attr-Align Evaluation (Two-Stage, Paper Protocol)")
            print("="*78)
            print(f"  Attr-Align Thresholds: {self.attr_thresholds}")
            print("─" * 78)

        for img_id, gt_anns in gt_map.items():
            if len(gt_anns) == 0:
                continue
            
            gt_bbox = gt_anns[0]['bbox']
            total_samples += 1

            # 获取 Top-1 预测
            preds = self._get_topk_preds(img_id, 1, score_key=score_key)
            if preds is None or preds['boxes'].shape[0] == 0:
                continue

            pred_bbox = preds['boxes'][0]
            iou, _, _ = self._compute_iou_with_area(gt_bbox, self._xyxy_to_xywh(pred_bbox))
            
            # Step 1: IoU 检查
            if iou < 0.5:
                continue
            
            step1_correct += 1
            
            # Step 2: 属性相似度检查
            # 获取该预测框对应的属性得分列表
            attr_scores = self._get_attr_scores_for_top1(img_id, score_key=score_key)
            
            if attr_scores is None or len(attr_scores) == 0:
                # 如果没有属性得分，仅基于 Step1 判定
                for thr in self.attr_thresholds:
                    attr_align_counters[thr] += 1
                continue
            
            # 计算平均属性相似度
            mean_attr_score = float(np.mean(attr_scores))
            correct_attr_scores.append(mean_attr_score)
            all_attr_scores.append(mean_attr_score)
            
            # 检查是否满足各阈值
            for thr in self.attr_thresholds:
                if mean_attr_score >= thr:
                    attr_align_counters[thr] += 1

        # 计算结果
        for thr in self.attr_thresholds:
            acc = (attr_align_counters[thr] / total_samples * 100) if total_samples > 0 else 0.0
            results[f'AttrAlign@{thr}'] = acc
            results[f'AttrRobust@{thr}'] = acc  # backward-compatible alias
        
        # 额外统计指标
        results['Step1_Acc'] = (step1_correct / total_samples * 100) if total_samples > 0 else 0.0
        results['MeanAttrScore'] = np.mean(correct_attr_scores) if len(correct_attr_scores) > 0 else 0.0
        results['MeanAttrScore_All'] = np.mean(all_attr_scores) if len(all_attr_scores) > 0 else 0.0
        
        if verbose:
            print(f"\n  Step1 (IoU>0.5) Accuracy: {results['Step1_Acc']:.2f}%")
            print(f"  Mean Attr Score (Step1 correct): {results['MeanAttrScore']:.4f}")
            print("\n  Two-Stage Accuracy @ Different Attr-Align Thresholds:")
            print("  " + "─" * 74)
            for thr in self.attr_thresholds:
                print(f"    Attr-Align @ τ={thr}  = {results[f'AttrAlign@{thr}']:.2f}%")
        
        return results
    
    def _get_attr_scores_for_top1(self, img_id, score_key='scores'):
        """
        获取 Top-1 预测框对应的属性得分列表
        
        支持两种数据格式：
        1. 新格式（推荐）：postprocessor 返回 'attr_scores_list' [N, Attr] 和 'attr_mean_for_topk' [N]
           - attr_scores_list: 每个 top-k 框对应的各属性独立得分
           - attr_mean_for_topk: 每个 top-k 框的平均属性得分（已预计算）
        2. 旧格式：需要手动从原始数据中提取
        
        Args:
            img_id: 图像 ID
            score_key: 用于排序的得分键
            
        Returns:
            np.ndarray or None: 属性得分列表 [num_attrs]，或单个平均值 [1]
        """
        if img_id not in self.all_predictions:
            return None
        
        preds = self.all_predictions[img_id]
        
        # ========== 方式1：使用预计算的 attr_mean_for_topk（最简单） ==========
        attr_mean_raw = preds.get('attr_mean_for_topk', [])
        if attr_mean_raw:
            if isinstance(attr_mean_raw, list) and len(attr_mean_raw) > 0:
                attr_mean = torch.cat(attr_mean_raw, dim=0) if isinstance(attr_mean_raw[0], torch.Tensor) else attr_mean_raw[0]
            else:
                attr_mean = attr_mean_raw
            
            if isinstance(attr_mean, torch.Tensor) and attr_mean.numel() > 0:
                # 获取用于排序的得分，找到 Top-1 索引
                scores_raw = preds.get(score_key, preds.get('scores', []))
                if scores_raw:
                    if isinstance(scores_raw, list):
                        scores = torch.cat(scores_raw, dim=0)
                    else:
                        scores = scores_raw
                    
                    if scores.numel() > 0:
                        top1_idx = torch.argmax(scores).item()
                        # 返回 Top-1 对应的平均属性得分
                        return np.array([attr_mean[top1_idx].item()])
        
        # ========== 方式2：使用 attr_scores_list 获取各属性独立得分 ==========
        attr_scores_list_raw = preds.get('attr_scores_list', [])
        if attr_scores_list_raw:
            # 合并多次 update 的结果
            if isinstance(attr_scores_list_raw, list):
                if len(attr_scores_list_raw) == 0:
                    return None
                attr_scores_all = torch.cat(attr_scores_list_raw, dim=0) if isinstance(attr_scores_list_raw[0], torch.Tensor) else attr_scores_list_raw[0]
            else:
                attr_scores_all = attr_scores_list_raw
            
            if isinstance(attr_scores_all, torch.Tensor) and attr_scores_all.numel() > 0:
                # 获取用于排序的得分
                scores_raw = preds.get(score_key, preds.get('scores', []))
                if not scores_raw:
                    return None
                
                if isinstance(scores_raw, list):
                    scores = torch.cat(scores_raw, dim=0)
                else:
                    scores = scores_raw
                
                if scores.numel() == 0:
                    return None
                
                # 找到 Top-1 的索引
                top1_idx = torch.argmax(scores).item()
                
                # 返回对应的属性得分
                if attr_scores_all.dim() == 1:
                    # 如果是 1D，假设每个样本只有一个属性得分
                    return np.array([attr_scores_all[top1_idx].item()])
                else:
                    # 如果是 2D [N, num_attrs]，返回 Top-1 对应的所有属性得分
                    return attr_scores_all[top1_idx].cpu().numpy()
        
        return None

    def _print_summary_table(self, ds_name, score_keys, all_metrics):
        """打印简洁的汇总表格"""
        # 得分键显示名称映射
        display_names = {
            'scores': 'score',
            'scores_attr': 'attr_score',
            'scores_avg': 'combine',
        }

        # 表头
        header = "| {:^12} | {:^6} | {:^6} | {:^6} | {:^6} | {:^6} | {:^7} | {:^7} | {:^8} | {:^8} |".format(
            "Score Type", "Pr@0.5", "Pr@0.6", "Pr@0.7", "Pr@0.8", "Pr@0.9", "meanIoU", "cmuIoU", "Acc_Top1", "Acc_Top5"
        )
        sep_line = "+" + "-"*14 + "+" + "-"*8 + "+" + "-"*8 + "+" + "-"*8 + "+" + "-"*8 + "+" + "-"*8 + "+" + "-"*9 + "+" + "-"*9 + "+" + "-"*10 + "+" + "-"*10 + "+"

        print("\n")
        print("=" * len(sep_line))
        title = f"GROUNDING EVALUATION RESULTS [{ds_name}]"
        print(f"{title:^{len(sep_line)}}")
        print("=" * len(sep_line))
        print(sep_line)
        print(header)
        print(sep_line)

        for score_key in score_keys:
            m = all_metrics[score_key]
            name = display_names.get(score_key, score_key)
            row = "| {:^12} | {:>6.2f} | {:>6.2f} | {:>6.2f} | {:>6.2f} | {:>6.2f} | {:>7.2f} | {:>7.2f} | {:>8.2f} | {:>8.2f} |".format(
                name,
                m.get('Pr@0.5', 0.0),
                m.get('Pr@0.6', 0.0),
                m.get('Pr@0.7', 0.0),
                m.get('Pr@0.8', 0.0),
                m.get('Pr@0.9', 0.0),
                m.get('meanIoU', 0.0),
                m.get('cmuIoU', 0.0),
                m.get('Acc_Top1', 0.0),
                m.get('Acc_Top5', 0.0),
            )
            print(row)

        print(sep_line)
        print("")
    
    def _print_attr_robust_table(self, ds_name, attr_results):
        """[NEW] 打印论文 Attr-Align 评测结果表格"""
        print("\n")
        print("=" * 78)
        title = f"ATTR-ALIGN RESULTS [{ds_name}]"
        print(f"{title:^78}")
        print("=" * 78)
        
        # 构建表头
        thr_headers = " | ".join([f"Attr-Align@{thr}" for thr in self.attr_thresholds])
        header = f"| {'Step1_Acc':^10} | {thr_headers} | {'MeanAttrScore':^14} |"
        
        sep_parts = ["-"*12] + ["-"*14 for _ in self.attr_thresholds] + ["-"*16]
        sep_line = "+" + "+".join(sep_parts) + "+"
        
        print(sep_line)
        print(header)
        print(sep_line)
        
        # 数据行
        thr_values = " | ".join([f"{attr_results.get(f'AttrAlign@{thr}', 0.0):>12.2f}" for thr in self.attr_thresholds])
        row = f"| {attr_results.get('Step1_Acc', 0.0):>10.2f} | {thr_values} | {attr_results.get('MeanAttrScore', 0.0):>14.4f} |"
        print(row)
        print(sep_line)
        
        # 额外说明
        print("\n  📝 Two-Stage Evaluation Protocol:")
        print("     Step 1: Traditional RSVG - IoU(pred, GT) > 0.5")
        print("     Step 2: Mean Attribute Similarity >= τ")
        print("     Final: Attr-Align@τ counts a sample only if BOTH Step1 AND Step2 pass")
        print("")

    def _visualize_failed_cases(self, score_key='scores', iou_threshold=0.5):
        """
        可视化 Top-1 预测失败的案例（IoU < iou_threshold）。
        GT 框用绿色绘制，Pred 框用红色绘制，保存到 vis_output_dir 目录。
        使用 Nano Banana 风格：半透明填充框、胶囊标签、Glass HUD caption 卡片。

        需要在初始化时提供 img_root，且 coco_gt.dataset['images'] 包含文件名信息。
        """
        try:
            from PIL import Image, ImageDraw, ImageFont, ImageFilter
        except ImportError:
            print("[VisFailedCases] Pillow not available, skipping visualization.")
            return

        if self.img_root is None:
            print("[VisFailedCases] img_root is None, skipping visualization. "
                  "Please set img_root when constructing GroundingEvaluator.")
            return

        os.makedirs(self.vis_output_dir, exist_ok=True)

        gt_map = self._gather_gt()

        # 构建 image_id -> file_name 的映射
        img_info_map = {}
        if hasattr(self.coco_gt, 'dataset') and 'images' in self.coco_gt.dataset:
            for img_info in self.coco_gt.dataset['images']:
                img_id = img_info['id']
                fname = img_info.get('file_name', img_info.get('filename', ''))
                img_info_map[img_id] = fname

        # ---- visualization style config ----
        _REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
        _FONT_FALLBACKS = [
            # bundled Roboto fonts (preferred)
            os.path.join(_REPO_ROOT, "font/static/Roboto-Bold.ttf"),
            os.path.join(_REPO_ROOT, "font/static/Roboto-Regular.ttf"),
            os.path.join(_REPO_ROOT, "font/Roboto-VariableFont_wdth,wght.ttf"),
            # system fonts (fallback)
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        ]
        _COLOR_GT   = (46, 204, 113)   # Emerald Green
        _COLOR_PRED = (255, 107, 107)  # Pastel Red
        _HUD_BG     = (255, 255, 255, 217)
        _HUD_STROKE = (255, 255, 255, 153)
        _HUD_HDR    = (85, 85, 85)
        _HUD_TEXT   = (68, 68, 68)

        def _load_font(size):
            for p in _FONT_FALLBACKS:
                if os.path.exists(p):
                    try:
                        return ImageFont.truetype(p, size)
                    except Exception:
                        pass
            return ImageFont.load_default()

        def _text_wh(draw, text, font):
            if hasattr(draw, 'textbbox'):
                bb = draw.textbbox((0, 0), text, font=font)
                return bb[2] - bb[0], bb[3] - bb[1]
            return font.getlength(text), 14

        def _draw_box_nano(overlay_draw, x1, y1, x2, y2, color_rgb):
            """带半透明填充的矩形框"""
            fill  = color_rgb + (46,)
            stroke = color_rgb + (255,)
            overlay_draw.rectangle([x1, y1, x2, y2], fill=fill)
            overlay_draw.rectangle([x1, y1, x2, y2], outline=stroke, width=3)

        def _draw_pill(overlay_draw, x1, y1, text, color_rgb, font):
            """在框左上角绘制胶囊形标签"""
            dummy = ImageDraw.Draw(Image.new("RGB", (1, 1)))
            tw, th = _text_wh(dummy, text, font)
            pad_x, pad_y = 8, 4
            pw = tw + pad_x * 2
            ph = th + pad_y * 2
            px = x1
            py = max(0, y1 - ph - 4)
            stroke = color_rgb + (255,)
            overlay_draw.rounded_rectangle([px, py, px + pw, py + ph], radius=8, fill=stroke)
            overlay_draw.text((px + pad_x, py + pad_y - 1), text, fill="white", font=font)

        def _draw_glass_hud(im_rgba, caption_text, font):
            """在图像左上角绘制 Glass HUD caption 卡片"""
            img_w = im_rgba.width
            max_w = int(img_w * 0.72)
            dummy = ImageDraw.Draw(Image.new("RGB", (1, 1)))

            # 文字换行
            words = caption_text.split()
            lines, cur = [], []
            for w in words:
                test = ' '.join(cur + [w])
                tw, _ = _text_wh(dummy, test, font)
                if tw <= max_w - 40:
                    cur.append(w)
                else:
                    if cur:
                        lines.append(' '.join(cur))
                    cur = [w]
            if cur:
                lines.append(' '.join(cur))

            if hasattr(dummy, 'textbbox'):
                bb = dummy.textbbox((0, 0), "Ag", font=font)
                lh = bb[3] - bb[1] + 10
            else:
                lh = 22
            hdr_h = lh + 8
            padding = 16

            line_widths = [_text_wh(dummy, ln, font)[0] for ln in lines]
            hdr_w, _ = _text_wh(dummy, "CAPTION:", font)
            content_w = max([hdr_w] + line_widths) if lines else hdr_w
            w = content_w + padding * 2
            h = padding + hdr_h + len(lines) * lh + padding

            shadow_blur = 8
            card = Image.new("RGBA", (int(w + shadow_blur * 2), int(h + shadow_blur * 2)), (0, 0, 0, 0))
            sd = ImageDraw.Draw(card)
            sd.rounded_rectangle(
                [shadow_blur + 2, shadow_blur + 4, shadow_blur + w + 2, shadow_blur + h + 4],
                radius=12, fill=(0, 0, 0, 40))
            card = card.filter(ImageFilter.GaussianBlur(4))
            cd = ImageDraw.Draw(card)
            cd.rounded_rectangle(
                [shadow_blur, shadow_blur, shadow_blur + w, shadow_blur + h],
                radius=12, fill=_HUD_BG, outline=_HUD_STROKE, width=1)

            cx = shadow_blur + padding
            cy = shadow_blur + padding
            cd.text((cx, cy), "CAPTION:", fill=_HUD_HDR, font=font)
            sep_y = cy + lh
            cd.line([(cx, sep_y), (cx + content_w, sep_y)], fill=(200, 200, 200, 180), width=1)
            cy += hdr_h + 4
            for ln in lines:
                cd.text((cx, cy), ln, fill=_HUD_TEXT, font=font)
                cy += lh

            im_rgba.paste(card, (16, 16), card)
        # ---- 风格配置结束 ----

        font_tag = _load_font(30)  # 标签字体：匹配 visualize_annotations.py
        font_hud = _load_font(40)  # Caption 字体：匹配 visualize_annotations.py (28->40)

        count = 0
        for img_id, gt_anns in gt_map.items():
            if count >= self.vis_max_cases:
                break
            if len(gt_anns) == 0:
                continue

            gt_bbox_xywh = gt_anns[0]['bbox']  # [x, y, w, h]
            caption = gt_anns[0].get('caption', '')

            # 获取 Top-1 预测
            preds = self._get_topk_preds(img_id, 1, score_key=score_key)

            iou = 0.0
            pred_box_xyxy = None
            pred_score = 0.0
            if preds is not None and preds['boxes'].shape[0] > 0:
                pred_box = preds['boxes'][0]
                pred_box_xywh = self._xyxy_to_xywh(pred_box)
                iou, _, _ = self._compute_iou_with_area(gt_bbox_xywh, pred_box_xywh)
                pred_box_xyxy = pred_box.tolist() if isinstance(pred_box, torch.Tensor) else list(pred_box)
                pred_score = preds['scores'][0].item() if preds['scores'].numel() > 0 else 0.0

            if iou >= iou_threshold:
                continue  # 不是失败案例，跳过

            # 找到图像路径
            fname = img_info_map.get(img_id, '')
            if fname:
                img_path = os.path.join(self.img_root, fname)
            else:
                # fallback：尝试常见扩展名
                for ext in ('.jpg', '.jpeg', '.png', '.tif', '.tiff'):
                    candidate = os.path.join(self.img_root, str(img_id) + ext)
                    if os.path.exists(candidate):
                        img_path = candidate
                        break
                else:
                    continue

            if not os.path.exists(img_path):
                continue

            try:
                image = Image.open(img_path).convert('RGB')
            except Exception:
                continue

            # ---- Nano Banana 风格绘制 ----
            im = image.convert("RGBA")
            overlay = Image.new("RGBA", im.size, (0, 0, 0, 0))
            ov_draw = ImageDraw.Draw(overlay)

            # 绘制 GT（绿色）
            gx, gy, gw, gh = gt_bbox_xywh
            _draw_box_nano(ov_draw, gx, gy, gx + gw, gy + gh, _COLOR_GT)
            _draw_pill(ov_draw, gx, gy, 'GT', _COLOR_GT, font_tag)

            # 绘制 Pred（红色）
            if pred_box_xyxy is not None:
                px1, py1, px2, py2 = pred_box_xyxy
                _draw_box_nano(ov_draw, px1, py1, px2, py2, _COLOR_PRED)
                pred_label = f'Pred {pred_score:.2f}'
                _draw_pill(ov_draw, px1, py1, pred_label, _COLOR_PRED, font_tag)

            im = Image.alpha_composite(im, overlay)

            # 绘制 Glass HUD caption
            if caption:
                _draw_glass_hud(im, caption, font_hud)
            # ---- 绘制结束 ----

            # 保存
            safe_fname = os.path.splitext(os.path.basename(fname))[0] if fname else str(img_id)
            save_name = f'{safe_fname}_iou{iou:.3f}.jpg'
            save_path = os.path.join(self.vis_output_dir, save_name)
            im.convert("RGB").save(save_path, quality=92)
            count += 1

        print(f"[VisFailedCases] Saved {count} failed cases to '{self.vis_output_dir}' "
              f"(score_key={score_key}, iou_thr={iou_threshold})")

    # ---------------- small utilities ---------------- #
    def _image_recall(self, gt_boxes, preds, iou_thr):
        if preds is None or preds['boxes'].shape[0] == 0:
            return 0.0

        pred_xywh = [self._xyxy_to_xywh(box) for box in preds['boxes']]
        matched = 0
        used = set()

        for gt in gt_boxes:
            best_iou = 0.0
            best_idx = -1
            for idx, pb in enumerate(pred_xywh):
                if idx in used:
                    continue
                iou, _, _ = self._compute_iou_with_area(gt, pb)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_iou >= iou_thr and best_idx >= 0:
                matched += 1
                used.add(best_idx)

        return matched / len(gt_boxes) if len(gt_boxes) else 0.0

    def _get_topk_preds(self, img_id, k, score_key='scores'):
        if img_id not in self.all_predictions:
            return None
        preds = self.all_predictions[img_id]
        
        # 根据 score_key 确定对应的 boxes_key 和 labels_key
        # scores -> boxes, labels
        # scores_attr -> boxes_attr, labels_attr
        # scores_avg -> boxes_avg, labels_avg
        if score_key == 'scores':
            boxes_key = 'boxes'
            labels_key = 'labels'
        else:
            # 从 score_key 提取后缀，如 'scores_attr' -> 'attr'
            suffix = score_key.split('scores_')[-1] if 'scores_' in score_key else ''
            boxes_key = f'boxes_{suffix}' if suffix else 'boxes'
            labels_key = f'labels_{suffix}' if suffix else 'labels'

        # 合并列表
        def _concat(key, empty_shape, fallback_key=None):
            lst = preds.get(key, [])
            if not lst and fallback_key:
                lst = preds.get(fallback_key, [])
            if not lst:
                return torch.empty(empty_shape)
            return torch.cat(lst, dim=0) if isinstance(lst[0], torch.Tensor) else lst[0]

        # 获取对应的 boxes 和 scores，如果不存在则回退到默认的 boxes/scores
        boxes = _concat(boxes_key, (0, 4), fallback_key='boxes')
        scores = _concat(score_key, (0,), fallback_key='scores')
        labels = _concat(labels_key, (0,), fallback_key='labels')

        if boxes.shape[0] == 0 or scores.numel() == 0:
            return None

        # 取 top-k
        actual_k = min(k, scores.shape[0])
        topk_indices = torch.topk(scores, actual_k).indices
        return {
            'boxes': boxes[topk_indices],
            'scores': scores[topk_indices],
            'labels': labels[topk_indices] if labels.numel() > 0 else None,
        }

    @staticmethod
    def _xyxy_to_xywh(box):
        """将 [x1,y1,x2,y2] 转换为 [x,y,w,h]"""
        if isinstance(box, torch.Tensor):
            box = box.tolist()
        return [box[0], box[1], box[2] - box[0], box[3] - box[1]]

    @staticmethod
    def _compute_iou_with_area(box1_xywh, box2_xywh):
        """
        计算 IoU 并返回交集/并集面积
        box format: [x, y, w, h]
        """
        x1, y1, w1, h1 = box1_xywh
        x2, y2, w2, h2 = box2_xywh

        # 转换为 [x1, y1, x2, y2]
        box1_x2, box1_y2 = x1 + w1, y1 + h1
        box2_x2, box2_y2 = x2 + w2, y2 + h2

        # 计算交集
        inter_x1 = max(x1, x2)
        inter_y1 = max(y1, y2)
        inter_x2 = min(box1_x2, box2_x2)
        inter_y2 = min(box1_y2, box2_y2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        # 计算并集
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area

        # 计算 IoU
        iou = inter_area / union_area if union_area > 0 else 0
        return iou, inter_area, union_area