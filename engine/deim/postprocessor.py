"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.

Modified: 新增 attr_scores_for_caption_topk 用于 Attr-Align 评测
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

from ..core import register


__all__ = ['PostProcessor']


def mod(a, b):
    out = a - a // b * b
    return out


@register()
class PostProcessor(nn.Module):
    __share__ = [
        'num_classes',
        'use_focal_loss',
        'num_top_queries',
        'remap_mscoco_category'
    ]

    def __init__(
        self,
        num_classes=80,
        use_focal_loss=True,
        num_top_queries=300,
        remap_mscoco_category=False
    ) -> None:
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.num_classes = int(num_classes)
        self.remap_mscoco_category = remap_mscoco_category
        self.deploy_mode = False

    def extra_repr(self) -> str:
        return f'use_focal_loss={self.use_focal_loss}, num_classes={self.num_classes}, num_top_queries={self.num_top_queries}'

    def forward(self, outputs, orig_target_sizes: torch.Tensor, cap_to_attr_map_batch=None):
        logits, boxes = outputs['pred_logits'], outputs['pred_boxes']
        attr_logits = outputs.get('pred_attr_logits', None)

        bbox_pred = torchvision.ops.box_convert(boxes, in_fmt='cxcywh', out_fmt='xyxy')
        bbox_pred *= orig_target_sizes.repeat(1, 2).unsqueeze(1)

        if self.use_focal_loss:
            caption_scores = F.sigmoid(logits)
            attr_score_mean = None
            
            # ========== [NEW] 保存原始属性得分用于后续对齐 ==========
            attr_scores_raw = None  # [B, Q, Attr] 原始属性得分
            
            if attr_logits is not None and cap_to_attr_map_batch is not None:
                # 将属性分类分数映射并对齐到 caption 级别，取有效属性的平均
                attr_scores = F.sigmoid(attr_logits)  # [B, Q, Attr]
                attr_scores_raw = attr_scores  # [NEW] 保存原始属性得分
                
                map_mask = (cap_to_attr_map_batch != -1)
                safe_map = cap_to_attr_map_batch.clone()
                safe_map[~map_mask] = 0

                attr_scores_expanded = attr_scores.unsqueeze(2)  # [B, Q, 1, Attr]
                num_queries = attr_scores.shape[1]
                safe_map_expanded = safe_map.unsqueeze(1).expand(-1, num_queries, -1, -1)  # [B, Q, C, Attr_per_cap]

                gathered_scores = torch.gather(
                    attr_scores_expanded.expand(-1, -1, safe_map.shape[1], -1),
                    dim=3,
                    index=safe_map_expanded
                )

                mask_expanded = map_mask.unsqueeze(1).expand_as(gathered_scores)
                gathered_scores = gathered_scores * mask_expanded.float()

                valid_counts = mask_expanded.sum(dim=-1).float().clamp(min=1.0)  # [B, Q, C]
                attr_score_mean = gathered_scores.sum(dim=-1) / valid_counts    # [B, Q, C]

            # 推理的时候，允许传入attr-set Interaction
            score_source = getattr(self, "_force_score_source", "caption")
            if score_source == "attr" and attr_score_mean is not None:
                caption_scores = attr_score_mean
            # elif score_source == "avg" and attr_score_mean is not None:
            #     has_attrs = (cap_to_attr_map_batch != -1).any(dim=-1).float().unsqueeze(1)
            #     caption_scores = caption_scores * (1 - has_attrs) + attr_score_mean * has_attrs

            # 推理时允许一部分用属性得分，一部分用global 得分
            caption_pref = getattr(self, "_per_caption_score_pref", None)
            if caption_pref is not None and attr_score_mean is not None:
                pref_mask = caption_pref
                if pref_mask.dim() == 1:
                    pref_mask = pref_mask.unsqueeze(0)
                pref_mask = pref_mask.to(caption_scores.device, dtype=torch.bool)  # [B, C]
                pref_mask = pref_mask.unsqueeze(1)  # [B, 1, C]
                caption_scores = caption_scores * (~pref_mask) + attr_score_mean * pref_mask

            # 为每个query只取最高分的那个类别
            flat_scores = caption_scores.flatten(1)
            scores, topk_indices = torch.topk(flat_scores, self.num_top_queries, dim=-1)
            labels = mod(topk_indices, logits.shape[-1])
            query_indices = topk_indices // logits.shape[-1]
            boxes = bbox_pred.gather(dim=1, index=query_indices.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1]))

            # ========== [NEW] 获取 Caption Top-K 对应的属性得分 ==========
            # 用于 Attr-Align 评测
            attr_scores_for_caption_topk = None  # [B, num_top_queries, Attr]
            attr_mean_for_caption_topk = None    # [B, num_top_queries] 每个 top-k 框的平均属性得分
            valid_attr_counts_for_topk = None    # [B, num_top_queries] 每个 top-k 框的有效属性个数
            
            if attr_scores_raw is not None and cap_to_attr_map_batch is not None:
                # 根据 caption top-k 的 query_indices 获取对应的属性得分
                # attr_scores_raw: [B, Q, Attr]
                # query_indices: [B, num_top_queries]
                # labels: [B, num_top_queries] - 每个 top-k 框对应的 caption 索引
                num_attrs = attr_scores_raw.shape[-1]
                B, K = labels.shape
                
                attr_scores_for_caption_topk = attr_scores_raw.gather(
                    dim=1, 
                    index=query_indices.unsqueeze(-1).expand(-1, -1, num_attrs)
                )  # [B, num_top_queries, Attr]
                
                # ===== 向量化计算：排除 padding 的平均属性得分 =====
                # cap_to_attr_map_batch: [B, C, Attr_per_cap]，-1 表示 padding
                # 根据 labels 索引获取每个 top-k 框对应 caption 的属性映射
                # labels: [B, K] -> 用于索引 cap_to_attr_map_batch 的 dim=1
                
                Attr_per_cap = cap_to_attr_map_batch.shape[-1]
                # 扩展 labels 用于 gather: [B, K] -> [B, K, Attr_per_cap]
                labels_expanded = labels.unsqueeze(-1).expand(-1, -1, Attr_per_cap)
                # 获取每个 top-k 框对应的属性索引映射: [B, K, Attr_per_cap]
                attr_map_for_topk = cap_to_attr_map_batch.gather(dim=1, index=labels_expanded)
                
                # 构建有效性 mask: [B, K, Attr_per_cap]
                valid_mask = (attr_map_for_topk != -1)
                # 统计每个 top-k 框的有效属性个数
                valid_attr_counts_for_topk = valid_mask.sum(dim=-1).float().clamp(min=1.0)  # [B, K]
                
                # 将无效索引替换为 0（避免 gather 越界）
                safe_attr_map = attr_map_for_topk.clone()
                safe_attr_map[~valid_mask] = 0
                
                # 从 attr_scores_for_caption_topk 中 gather 有效属性的得分
                # attr_scores_for_caption_topk: [B, K, Attr]
                # safe_attr_map: [B, K, Attr_per_cap]
                gathered_attr_scores = attr_scores_for_caption_topk.gather(dim=2, index=safe_attr_map)  # [B, K, Attr_per_cap]
                
                # 应用 mask 并求和
                gathered_attr_scores = gathered_attr_scores * valid_mask.float()
                attr_sum = gathered_attr_scores.sum(dim=-1)  # [B, K]
                
                # 计算平均值
                attr_mean_for_caption_topk = attr_sum / valid_attr_counts_for_topk  # [B, K]

            # 对齐属性分数和组合分数（各自独立取 top-k），使用与 caption 相同的 [B, Q, C] 维度
            attr_scores_top = None
            attr_boxes_top, attr_labels_top = None, None
            avg_scores_top = None
            avg_boxes_top, avg_labels_top = None, None

            if attr_score_mean is not None:
                # 属性分支：直接基于 [B, Q, C] 的均值分数取 top-k
                attr_flat = attr_score_mean.flatten(1)
                attr_scores_top, attr_top_idx = torch.topk(attr_flat, self.num_top_queries, dim=-1)
                attr_labels_top = mod(attr_top_idx, logits.shape[-1])
                attr_query_idx = attr_top_idx // logits.shape[-1]
                attr_boxes_top = bbox_pred.gather(dim=1, index=attr_query_idx.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1]))

                # 平均分数：caption_scores 与 attr_score_mean 按元素平均后取 top-k
                has_attrs = (cap_to_attr_map_batch != -1).any(dim=-1).float().unsqueeze(1)  # [B, 1, C]
                weight = 0.5 * has_attrs
                avg_scores = caption_scores * (1 - weight) + attr_score_mean * weight  # [B, Q, C]
                avg_flat = avg_scores.flatten(1)
                avg_scores_top, avg_top_idx = torch.topk(avg_flat, self.num_top_queries, dim=-1)
                avg_labels_top = mod(avg_top_idx, logits.shape[-1])
                avg_query_idx = avg_top_idx // logits.shape[-1]
                avg_boxes_top = bbox_pred.gather(dim=1, index=avg_query_idx.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1]))

        else:
            scores = F.softmax(logits)[:, :, :-1]
            scores, labels = scores.max(dim=-1)
            if scores.shape[1] > self.num_top_queries:
                scores, index = torch.topk(scores, self.num_top_queries, dim=-1)
                labels = torch.gather(labels, dim=1, index=index)
                boxes = torch.gather(boxes, dim=1, index=index.unsqueeze(-1).tile(1, 1, boxes.shape[-1]))

        if self.deploy_mode:
            return labels, boxes, scores

        if self.remap_mscoco_category:
            from ..data.dataset import mscoco_label2category
            labels = torch.tensor([mscoco_label2category[int(x.item())] for x in labels.flatten()])\
                .to(boxes.device).reshape(labels.shape)

        results = []
        for i, (lab, box, sco) in enumerate(zip(labels, boxes, scores)):
            result = dict(labels=lab, boxes=box, scores=sco)
            results.append(result)

        # 附加属性/平均分支的 top-k 结果
        if attr_logits is not None and attr_scores_top is not None:
            for i in range(attr_scores_top.shape[0]):
                results[i]['scores_attr'] = attr_scores_top[i]
                results[i]['boxes_attr'] = attr_boxes_top[i]
                results[i]['labels_attr'] = attr_labels_top[i]
                results[i]['scores_avg'] = avg_scores_top[i]
                results[i]['boxes_avg'] = avg_boxes_top[i]
                results[i]['labels_avg'] = avg_labels_top[i]
                
                # ========== [NEW] 附加 Caption Top-K 对应的属性得分 ==========
                # 用于 Attr-Align 评测
                if attr_scores_for_caption_topk is not None:
                    # attr_scores_list: [num_top_queries, Attr] - 每个 top-k 框对应的各属性得分
                    results[i]['attr_scores_list'] = attr_scores_for_caption_topk[i]
                    # attr_mean_for_topk: [num_top_queries] - 每个 top-k 框的平均属性得分（已排除 padding）
                    results[i]['attr_mean_for_topk'] = attr_mean_for_caption_topk[i]
                    # valid_attr_counts: [num_top_queries] - 每个 top-k 框的有效属性个数
                    if valid_attr_counts_for_topk is not None:
                        results[i]['valid_attr_counts'] = valid_attr_counts_for_topk[i]

        return results

    def deploy(self, ):
        self.eval()
        self.deploy_mode = True
        return self