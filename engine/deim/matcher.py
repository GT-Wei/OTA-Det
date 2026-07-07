"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
Modules to compute the matching cost and solve the corresponding LSAP.

Copyright (c) 2024 The D-FINE Authors All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.optimize import linear_sum_assignment
from typing import Dict

from .box_ops import box_cxcywh_to_xyxy, generalized_box_iou, box_iou

from ..core import register
import numpy as np


@register()
class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    __share__ = ['use_focal_loss', ]

    def __init__(self, weight_dict, use_focal_loss=False, alpha=0.25, gamma=2.0,
                change_matcher=False, iou_order_alpha=1.0, matcher_change_epoch=10000):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = weight_dict['cost_class']
        self.cost_bbox = weight_dict['cost_bbox']
        self.cost_giou = weight_dict['cost_giou']

        self.change_matcher = change_matcher
        self.iou_order_alpha = iou_order_alpha
        self.matcher_change_epoch = matcher_change_epoch
        if self.change_matcher:
            print(f"Using the new matching cost with iou_order_alpha = {iou_order_alpha} at epoch {matcher_change_epoch}")

        self.use_focal_loss = use_focal_loss
        self.alpha = alpha
        self.gamma = gamma

        assert self.cost_class != 0 or self.cost_bbox != 0 or self.cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs: Dict[str, torch.Tensor], targets, return_topk=False, epoch=0):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        # bs, num_queries = outputs["pred_logits"].shape[:2]

        # ===== 检测batch模式 =====
        batch_mode = self._detect_batch_mode(targets)
        
        if batch_mode == "pure_grounding":  # 当前主要仅考虑pure_grounding和pure_detection任务
            # 纯grounding batch，使用批量处理
            return self._forward_batch_grounding(outputs, targets, return_topk, epoch)
        elif batch_mode == "pure_detection":
            # 纯detection batch，使用原有逻辑
            return self._forward_batch_detection(outputs, targets, return_topk, epoch)
        else:
            raise ValueError('only support pure detection / grounding. OTA-Det unify them into pure_grounding mode.')

    def _detect_batch_mode(self, targets):
        """
        检测batch的模式
        
        Returns:
            "pure_grounding": 全部grounding
            "pure_detection": 全部detection
            "mixed": 混合
        """
        if targets[0].get('OTA-Det', False):
            return "pure_grounding"
        else:
            return "pure_detection"
        
        # modes = []
        # for tgt in targets:
        #     OTA-Det = "OTA-Det" in tgt and tgt["OTA-Det"]
        #     has_labels = "labels" in tgt and tgt["labels"] is not None and len(tgt["labels"]) > 0
            
        #     if has_onehot:  # 有onehot就按照grounding的形式训练
        #         modes.append("grounding")
        #     elif has_labels:
        #         modes.append("detection")
        #     else:
        #         # 既有onehot又有labels，或都没有，视为异常，归为mixed
        #         modes.append("mixed")
        
        # if all(m == "grounding" for m in modes):
        #     return "pure_grounding"
        # elif all(m == "detection" for m in modes):
        #     return "pure_detection"
        # else:
        #     raise ValueError("Grounding / Detection mixed, check.")

    def _forward_batch_grounding(self, outputs, targets, return_topk, epoch):
        """
        纯grounding batch的批量处理
        由于每个batch,caption的位置和内容都是不同的,因此,分类cost不能够直接展平,因为展平同样是0,对应的caption语义其实是不同的
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]
        eps = 1e-8

        # 展平预测
        # out_prob = F.sigmoid(outputs["pred_logits"].flatten(0, 1))  # [B*Q, C]
        out_prob = F.sigmoid(outputs["pred_logits"])  # [B, Q, C] 不 flatten
        out_bbox = outputs["pred_boxes"].flatten(0, 1)     # [B*Q, 4]
        
        # has_attr = "pred_attr_logits" in outputs and outputs["pred_attr_logits"] is not None
        # if has_attr:
        #     out_attr_prob = F.sigmoid(outputs["pred_attr_logits"])  # [B, Q, C_attr] 也不 flatten
            
        tgt_bbox = torch.cat([v["boxes"] for v in targets], dim=0)  # [sum_T, 4]
        sizes = [len(v["boxes"]) for v in targets]
        
        sum_T = sum(sizes)

        # ===== 根据 change_matcher 计算代价 =====
        if self.change_matcher and epoch >= self.matcher_change_epoch:
            # 新匹配策略：IoU * class_score
            bbox_iou, _ = box_iou(
                box_cxcywh_to_xyxy(out_bbox),
                box_cxcywh_to_xyxy(tgt_bbox)
            )
            
            # class_score = out_prob @ tgt_onehot.t()  # [B*Q, sum_T]
            class_score = torch.zeros(bs * num_queries, sum_T, 
                                  device=out_prob.device, dtype=out_prob.dtype)
            col_start = 0
            for b in range(bs):
                T_b = sizes[b]
                if T_b == 0:
                    continue
                
                row_start = b * num_queries
                row_end = (b + 1) * num_queries
                col_end = col_start + T_b
                
                prob_b = out_prob[b]  # [Q, C]
                tgt_onehot_b = targets[b]["caption_indices_onehot"].to(prob_b.dtype)  # [T_b, C]
                
                class_score[row_start:row_end, col_start:col_end] = prob_b @ tgt_onehot_b.t()
                col_start = col_end
            
            C = (-1) * (class_score * torch.pow(bbox_iou, self.iou_order_alpha))
            
            # if has_attr_target:
            #     attr_score = torch.zeros(bs * num_queries, sum_T, 
            #                              device=out_attr_prob.device, dtype=out_attr_prob.dtype)
            #     col_start = 0
            #     for b in range(bs):
            #         T_b = sizes[b]
            #         if T_b == 0:
            #             continue
            #         row_start, row_end = b * num_queries, (b + 1) * num_queries
            #         col_end = col_start + T_b
            #         attr_prob_b = out_attr_prob[b]
            #         tgt_attr_onehot_b = targets[b]["caption_attributes_onehot"].to(attr_prob_b.dtype)
            #         attr_score[row_start:row_end, col_start:col_end] = attr_prob_b @ tgt_attr_onehot_b.t()
            #         col_start = col_end
            #     C_attr = (-1) * (attr_score * torch.pow(bbox_iou, self.iou_order_alpha))
            #     C = 0.5 * C + 0.5 * C_attr
        else:
            # 原始匹配策略：Focal Loss + L1 + GIoU
            # 计算几何代价
            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
            cost_giou = -generalized_box_iou(
                box_cxcywh_to_xyxy(out_bbox),
                box_cxcywh_to_xyxy(tgt_bbox)
            )
            
            # out_prob = out_prob @ tgt_onehot.t()  # select
            # pos_cost_class = self.alpha * ((1 - out_prob) ** self.gamma) * (-(out_prob + eps).log())
            # neg_cost_class = (1 - self.alpha) * (out_prob ** self.gamma) * (-(1 - out_prob).log())
            
            # 最终分类代价：正样本代价 - 负样本代价（使其与detection保持一致）
            # cost_class = pos_cost_class - neg_cost_class
            
            # if has_attr_target:
            #     out_prob_attr = out_attr_prob @ tgt_attr_onehot.t()
            #     pos_cost_attr = self.alpha * ((1 - out_prob_attr) ** self.gamma) * (-(out_prob_attr + eps).log())
            #     neg_cost_attr = (1 - self.alpha) * (out_prob_attr ** self.gamma) * (-(1 - out_prob_attr).log())
            #     cost_class_attr = pos_cost_attr - neg_cost_attr
            #     cost_class = 0.5 * cost_class + 0.5 * cost_class_attr  # 平权融合
            # 分类代价：逐样本计算
            cost_class = torch.full((bs * num_queries, sum_T), 1e6, 
                                    device=out_prob.device, dtype=out_prob.dtype)
            
            col_start = 0
            for b in range(bs):
                T_b = sizes[b]
                if T_b == 0:
                    continue
                
                row_start = b * num_queries
                row_end = (b + 1) * num_queries
                col_end = col_start + T_b
                
                prob_b = out_prob[b]  # [Q, C]
                tgt_onehot_b = targets[b]["caption_indices_onehot"].to(prob_b.dtype)  # [T_b, C]
                
                score_b = prob_b @ tgt_onehot_b.t()  # [Q, T_b]
                pos_cost = self.alpha * ((1 - score_b) ** self.gamma) * (-(score_b + eps).log())
                neg_cost = (1 - self.alpha) * (score_b ** self.gamma) * (-(1 - score_b + eps).log())
                
                cost_class[row_start:row_end, col_start:col_end] = pos_cost - neg_cost
                col_start = col_end
                    
            # 总代价
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou

        # ===== 求解匈牙利匹配 =====
        C = C.view(bs, num_queries, -1).cpu()
        C = torch.nan_to_num(C, nan=1.0)

        indices_pre = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) 
                for i, j in indices_pre]

        if return_topk:
            return {'indices_o2m': self.get_top_k_matches(C, sizes=sizes, k=return_topk, initial_indices=indices_pre)}
        
        return {'indices': indices}

    def _forward_batch_detection(self, outputs, targets, return_topk, epoch):
        """
        纯detection batch的批量处理（原有逻辑）
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # 展平预测
        if self.use_focal_loss:
            out_prob = F.sigmoid(outputs["pred_logits"].flatten(0, 1))
        else:
            out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]

        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        if self.change_matcher and epoch >= self.matcher_change_epoch:
            # Compute the class_score
            class_score = out_prob[:, tgt_ids]  # shape = [batch_size * num_queries, gt num within a batch]

            # # Compute iou
            bbox_iou, _ = box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

            # Final cost matrix
            C = (-1) * (class_score * torch.pow(bbox_iou, self.iou_order_alpha))
        else:
            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            if self.use_focal_loss:
                out_prob = out_prob[:, tgt_ids]
                neg_cost_class = (1 - self.alpha) * (out_prob ** self.gamma) * (-(1 - out_prob + 1e-8).log())
                pos_cost_class = self.alpha * ((1 - out_prob) ** self.gamma) * (-(out_prob + 1e-8).log())
                cost_class = pos_cost_class - neg_cost_class
            else:
                cost_class = -out_prob[:, tgt_ids]

            # Compute the L1 cost between boxes
            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

            # Compute the giou cost betwen boxes
            cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

            # Final cost matrix 3 * self.cost_bbox + 2 * self.cost_class + self.cost_giou
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou

        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        C = torch.nan_to_num(C, nan=1.0)
        indices_pre = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices_pre]

        # Compute topk indices
        if return_topk:
            return {'indices_o2m': self.get_top_k_matches(C, sizes=sizes, k=return_topk, initial_indices=indices_pre)}

        return {'indices': indices} # , 'indices_o2m': C.min(-1)[1]}

    def _compute_grounding_cost_block(self, logits_b, tgt_b, eps):
        """计算grounding的分类cost"""
        prob_b = torch.sigmoid(logits_b)
        
        g_b = tgt_b["caption_indices_onehot"].to(prob_b.device).to(prob_b.dtype)
        
        # if "caption_padding_mask" in tgt_b and tgt_b["caption_padding_mask"] is not None:
        #     mask = tgt_b["caption_padding_mask"].to(prob_b.device).to(prob_b.dtype)
        #     g_b = g_b * mask.unsqueeze(0)
        
        prob_b = prob_b @ g_b.t()
        # prob_b = torch.clamp(prob_b, min=eps, max=1-eps)
        pos_cost_class = self.alpha * ((1 - prob_b) ** self.gamma) * (-(prob_b + eps).log())
        neg_cost_class = (1 - self.alpha) * (prob_b ** self.gamma) * (-(1 - prob_b).log())
        
        # 最终分类代价：正样本代价 - 负样本代价（使其与detection保持一致）
        cost_class = pos_cost_class - neg_cost_class
        
        # pos_cost_class = -(prob_b + eps).log()
        # if self.gamma != 0:
        #     pos_cost_class = ((1 - prob_b) ** self.gamma) * pos_cost_class
        
        # if self.alpha is not None:
        #     pos_cost_class = self.alpha * pos_cost_class
        
        return cost_class

    def _compute_grounding_score_block(self, logits_b, tgt_b, eps):
        """计算grounding的分类score（用于change_matcher）"""
        prob_b = torch.sigmoid(logits_b)
        g_b = tgt_b["caption_indices_onehot"].to(prob_b.device).to(prob_b.dtype)
        
        # if "caption_padding_mask" in tgt_b and tgt_b["caption_padding_mask"] is not None:
        #     mask = tgt_b["caption_padding_mask"].to(prob_b.device).to(prob_b.dtype)
        #     g_b = g_b * mask.unsqueeze(0)
        return prob_b @ g_b.t()

    def _compute_traditional_cost_block(self, logits_b, tgt_ids_b, eps):
        """计算传统labels的分类cost"""
        if self.use_focal_loss:
            out_prob_b = torch.sigmoid(logits_b)
            out_prob_b = out_prob_b[:, tgt_ids_b]

            neg_cost_class = (1 - self.alpha) * (out_prob_b ** self.gamma) * \
                            (-(1 - out_prob_b + eps).log())
            pos_cost_class = self.alpha * ((1 - out_prob_b) ** self.gamma) * \
                            (-(out_prob_b + eps).log())
            return pos_cost_class - neg_cost_class
        else:
            out_prob_b = F.softmax(logits_b, dim=-1)
            return -out_prob_b[:, tgt_ids_b]

    def get_top_k_matches(self, C, sizes, k=1, initial_indices=None):
        indices_list = []
        # C_original = C.clone()
        for i in range(k):
            indices_k = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))] if i > 0 else initial_indices
            indices_list.append([
                (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
                for i, j in indices_k
            ])
            for c, idx_k in zip(C.split(sizes, -1), indices_k):
                idx_k = np.stack(idx_k)
                c[:, idx_k] = 1e6
        indices_list = [(torch.cat([indices_list[i][j][0] for i in range(k)], dim=0),
                        torch.cat([indices_list[i][j][1] for i in range(k)], dim=0)) for j in range(len(sizes))]
        # C.copy_(C_original)
        return indices_list

    
    # def _forward_mixed(self, outputs, targets, return_topk, epoch):
    #     """
    #     混合batch的逐样本处理
    #     """
    #     bs, num_queries = outputs["pred_logits"].shape[:2]
    #     eps = 1e-8

    #     # 展平预测
    #     out_logits = outputs["pred_logits"].flatten(0, 1)
    #     out_bbox = outputs["pred_boxes"].flatten(0, 1)

    #     # 拼接目标
    #     tgt_bbox = torch.cat([v["boxes"] for v in targets], dim=0)

    #     # ===== 根据 change_matcher 决定是否预先计算几何代价 =====
    #     if self.change_matcher and epoch >= self.matcher_change_epoch:
    #         C = torch.zeros(bs * num_queries, len(tgt_bbox), device=out_bbox.device)
    #     else:
    #         cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
    #         cost_giou = -generalized_box_iou(
    #             box_cxcywh_to_xyxy(out_bbox),
    #             box_cxcywh_to_xyxy(tgt_bbox)
    #         )
    #         C = self.cost_bbox * cost_bbox + self.cost_giou * cost_giou

    #     # 逐样本计算分类代价
    #     sizes = [len(v["boxes"]) for v in targets]
    #     col_start = 0

    #     for b in range(bs):
    #         T_b = sizes[b]
    #         if T_b == 0:
    #             continue
            
    #         col_end = col_start + T_b
    #         row_start = b * num_queries
    #         row_end = (b + 1) * num_queries

    #         tgt_b = targets[b]
    #         use_grounding = self._is_grounding_sample(tgt_b)

    #         if use_grounding:
    #             # ===== Grounding样本（总是使用sigmoid） =====
    #             if self.change_matcher and epoch >= self.matcher_change_epoch:
    #                 class_score_block = self._compute_grounding_score_block(
    #                     out_logits[row_start:row_end],
    #                     tgt_b,
    #                     eps
    #                 )
    #                 bbox_iou, _ = box_iou(
    #                     box_cxcywh_to_xyxy(out_bbox[row_start:row_end]),
    #                     box_cxcywh_to_xyxy(tgt_bbox[col_start:col_end])
    #                 )
    #                 C[row_start:row_end, col_start:col_end] = \
    #                     (-1) * (class_score_block * torch.pow(bbox_iou, self.iou_order_alpha))
    #             else:
    #                 cost_class_block = self._compute_grounding_cost_block(
    #                     out_logits[row_start:row_end], 
    #                     tgt_b, 
    #                     eps
    #                 )
    #                 C[row_start:row_end, col_start:col_end] += self.cost_class * cost_class_block

    #         else:
    #             # ===== Detection样本（需要判断use_focal_loss） =====
    #             tgt_ids_b = tgt_b["labels"].to(out_logits.device)
                
    #             if self.change_matcher and epoch >= self.matcher_change_epoch:
    #                 # 新匹配策略：根据 use_focal_loss 选择激活函数
    #                 if self.use_focal_loss:
    #                     out_prob_b = torch.sigmoid(out_logits[row_start:row_end])
    #                 else:
    #                     out_prob_b = F.softmax(out_logits[row_start:row_end], dim=-1)
                    
    #                 class_score_block = out_prob_b[:, tgt_ids_b]
    #                 bbox_iou = box_iou(
    #                     box_cxcywh_to_xyxy(out_bbox[row_start:row_end]),
    #                     box_cxcywh_to_xyxy(tgt_bbox[col_start:col_end])
    #                 )[0]
    #                 C[row_start:row_end, col_start:col_end] = \
    #                     (-1) * (class_score_block * torch.pow(bbox_iou, self.iou_order_alpha))
    #             else:
    #                 # 原始匹配策略
    #                 cost_class_block = self._compute_traditional_cost_block(
    #                     out_logits[row_start:row_end],
    #                     tgt_ids_b,
    #                     eps
    #                 )
    #                 C[row_start:row_end, col_start:col_end] += self.cost_class * cost_class_block

    #         col_start = col_end

    #     # 求解匈牙利匹配
    #     C = C.view(bs, num_queries, -1).cpu()
    #     C = torch.nan_to_num(C, nan=1.0)

    #     sizes = [len(v["boxes"]) for v in targets]
    #     indices_pre = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
    #     indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) 
    #             for i, j in indices_pre]

    #     if return_topk:
    #         return {'indices_o2m': self.get_top_k_matches(C, sizes=sizes, k=return_topk, initial_indices=indices_pre)}
        
    #     return {'indices': indices}
