"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE/)
Copyright (c) 2024 D-FINE Authors. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.distributed
import torch.nn.functional as F
import torchvision

import copy

from .dfine_utils import bbox2distance
from .box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou
from ..misc.dist_utils import get_world_size, is_dist_available_and_initialized
from ..core import register
from .img_text_ot_loss import OTAlign_Batch

@register()
class DEIMCriterion(nn.Module):
    """ This class computes the loss for DEIM.
    """
    __share__ = ['num_classes', ]
    __inject__ = ['matcher', ]

    def __init__(self, \
        matcher,
        weight_dict,
        losses,
        alpha=0.2,
        gamma=2.0,
        num_classes=80,
        reg_max=32,
        boxes_weight_format=None,
        share_matched_indices=False,
        mal_alpha=None,
        use_uni_set=True,
        attr_ot_cfg=None,
        ):
        """Create the criterion.
        Parameters:
            matcher: module able to compute a matching between targets and proposals.
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            num_classes: number of object categories, omitting the special no-object category.
            reg_max (int): Max number of the discrete bins in D-FINE.
            boxes_weight_format: format for boxes weight (iou, ).
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.boxes_weight_format = boxes_weight_format
        self.share_matched_indices = share_matched_indices
        self.alpha = alpha
        self.gamma = gamma
        self.fgl_targets, self.fgl_targets_dn = None, None
        self.own_targets, self.own_targets_dn = None, None
        self.reg_max = reg_max
        self.num_pos, self.num_neg = None, None
        self.mal_alpha = mal_alpha
        self.use_uni_set = use_uni_set
        
        self.ot_align = OTAlign_Batch(
            blur=0.05,  scaling=0.90, ent_tau=0.3,
        )

    def _is_grounding_mode(self, targets):  # 当前仅考虑Pure_Grounding Data情况，因此只判断第一个是不是Grounding任务即可
        """Check if we are in Grounding mode."""
        if len(targets) == 0:
            return False
        # Check if first target has caption_indices_onehot
        
        return targets[0].get('OTA-Det', False)

    def _collect_grounding_targets(self, targets, indices, num_captions, device):
        """Collect and expand caption targets and padding masks.
        
        Args:
            targets: List of target dicts
            indices: Matching indices
            num_captions: Number of captions
            device: torch device
            
        Returns:
            target_caption_onehot: [total_matched_boxes, num_captions]
        """
        caption_indices_list = []
        
        for t, (_, i) in zip(targets, indices):
            # num_matched_boxes = len(i)
            # Get caption one-hot for matched boxes
            # caption_indices_list.append(t['caption_indices_onehot'][i])
            if len(i) > 0:  # 只添加非空的匹配
                caption_indices_list.append(t['caption_indices_onehot'][i])
    
        # 处理所有样本都没有匹配的情况
        if len(caption_indices_list) == 0:
            # 返回空张量，shape = (0, num_captions)
            return torch.zeros((0, num_captions), dtype=torch.float32, device=device)

        target_caption_onehot = torch.cat(caption_indices_list, dim=0)  # [total_matched, num_captions]
        return target_caption_onehot

    def _collect_grounding_targets_attr(self, targets, indices, num_attr_captions, device):
        caption_indices_list = []
        
        for t, (_, i) in zip(targets, indices):
            # num_matched_boxes = len(i)
            # Get caption one-hot for matched boxes
            # caption_indices_list.append(t['caption_indices_onehot'][i])
            if len(i) > 0:  # 只添加非空的匹配
                caption_indices_list.append(t['caption_attributes_onehot'][i])
    
        # 处理所有样本都没有匹配的情况
        if len(caption_indices_list) == 0:
            # 返回空张量，shape = (0, num_captions)
            return torch.zeros((0, num_attr_captions), dtype=torch.float32, device=device)

        target_caption_onehot = torch.cat(caption_indices_list, dim=0)  # [total_matched, num_captions]
        return target_caption_onehot

    def _apply_text_mask_and_compute_loss(self, src_logits, target_score, weight, valid_mask, num_boxes):
        if valid_mask is not None:
            loss = F.binary_cross_entropy_with_logits(
                src_logits, target_score, 
                weight=weight, reduction='none'
            )
            loss = torch.nan_to_num(loss, nan=0.0)
            # 把padding位置置0
            loss = loss * valid_mask.float()
            # Detection式归一化
            loss = loss.mean(-1).sum() * src_logits.shape[1] / num_boxes
        else:
            # No padding mask: standard computation
            loss = F.binary_cross_entropy_with_logits(
                src_logits, target_score, 
                weight=weight, reduction='none'
            )
            loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
            
        return loss
    
    def loss_labels_focal(self, outputs, targets, indices, num_boxes):
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes+1)[..., :-1]
        loss = torchvision.ops.sigmoid_focal_loss(src_logits, target, self.alpha, self.gamma, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes

        return {'loss_focal': loss}

    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, values=None):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        if values is None:
            src_boxes = outputs['pred_boxes'][idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
            ious = torch.diag(ious).detach()
        else:
            ious = values

        src_logits = outputs['pred_logits']
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = F.sigmoid(src_logits).detach()
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score

        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_vfl': loss}

    # 分类损失，我只改了mal其他vfl / focal同理
    def loss_labels_mal(self, outputs, targets, indices, num_boxes, values=None):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        
        # Compute IoU
        if values is None:
            src_boxes = outputs['pred_boxes'][idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
            ious = torch.diag(ious).detach()
        else:
            ious = values

        src_logits = outputs['pred_logits']
        is_grounding = self._is_grounding_mode(targets)
        
        if is_grounding:
            num_captions = src_logits.shape[-1]
            
            # Collect caption targets (注意: mask现在只用于匹配的box,待会重新构建)
            target_caption_onehot = self._collect_grounding_targets(
                targets, indices, num_captions, src_logits.device
            )
            
            # Build full target tensor
            target = torch.zeros(src_logits.shape[0], src_logits.shape[1], num_captions,
                            dtype=src_logits.dtype, device=src_logits.device)
            target[idx[0], idx[1]] = target_caption_onehot.to(src_logits.dtype)
            
            # Build target score (IoU * one-hot)
            target_score_o = torch.zeros(src_logits.shape[0], src_logits.shape[1],
                                        dtype=src_logits.dtype, device=src_logits.device)
            target_score_o[idx] = ious.to(target_score_o.dtype)
            target_score = target_score_o.unsqueeze(-1) * target
            target_score = target_score.pow(self.gamma)
            
            # ⭐ 修正: 构建valid_mask_full - 所有query都应用样本级的mask
            valid_mask_full = torch.zeros(src_logits.shape[0], src_logits.shape[1], num_captions,
                                        dtype=torch.bool, device=src_logits.device)
            for batch_idx, t in enumerate(targets):
                if 'caption_padding_mask' in t:
                    sample_valid_mask = t['caption_padding_mask']  # [num_captions], True=valid, False=padding
                    # 广播到所有query
                    valid_mask_full[batch_idx, :, :] = sample_valid_mask.unsqueeze(0)
                else:
                    # 没有mask则全部有效
                    valid_mask_full[batch_idx, :, :] = True
            
            # Build weight (MAL style)
            pred_score = F.sigmoid(src_logits).detach()
            if self.mal_alpha is not None:
                weight = self.mal_alpha * pred_score.pow(self.gamma) * (1 - target) + target
            else:
                weight = pred_score.pow(self.gamma) * (1 - target) + target
            
            # Compute loss
            loss = self._apply_text_mask_and_compute_loss(
                src_logits, target_score, weight, valid_mask_full, num_boxes
            )
        else:
            target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
            target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                        dtype=torch.int64, device=src_logits.device)
            target_classes[idx] = target_classes_o
            target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]
            
            target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
            target_score_o[idx] = ious.to(target_score_o.dtype)
            target_score = target_score_o.unsqueeze(-1) * target
            target_score = target_score.pow(self.gamma)
            
            pred_score = F.sigmoid(src_logits).detach()
            if self.mal_alpha != None:
                weight = self.mal_alpha * pred_score.pow(self.gamma) * (1 - target) + target
            else:
                weight = pred_score.pow(self.gamma) * (1 - target) + target

            # print(" ### DEIM-gamma{}-alpha{} ### ".format(self.gamma, self.mal_alpha))
            loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
            loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
            
        # loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_mal': loss}

    def loss_labels_attr_mal(self, outputs, targets, indices, num_boxes, values=None):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        
        # Compute IoU
        if values is None:
            src_boxes = outputs['pred_boxes'][idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
            ious = torch.diag(ious).detach()
        else:
            ious = values

        src_logits = outputs['pred_attr_logits']
        is_grounding = self._is_grounding_mode(targets)
        
        if is_grounding:
            num_attr_captions = src_logits.shape[-1]
            
            # Collect caption targets (注意: mask现在只用于匹配的box,待会重新构建)
            target_caption_onehot = self._collect_grounding_targets_attr(
                targets, indices, num_attr_captions, src_logits.device
            )
            
            # Build full target tensor
            target = torch.zeros(src_logits.shape[0], src_logits.shape[1], num_attr_captions,
                            dtype=src_logits.dtype, device=src_logits.device)
            target[idx[0], idx[1]] = target_caption_onehot.to(src_logits.dtype)
            
            # Build target score (IoU * one-hot)
            target_score_o = torch.zeros(src_logits.shape[0], src_logits.shape[1],
                                        dtype=src_logits.dtype, device=src_logits.device)
            target_score_o[idx] = ious.to(target_score_o.dtype)
            target_score = target_score_o.unsqueeze(-1) * target
            target_score = target_score.pow(self.gamma)
            
            # ⭐ 修正: 构建valid_mask_full - 所有query都应用样本级的mask
            valid_mask_full = torch.zeros(src_logits.shape[0], src_logits.shape[1], num_attr_captions,
                                        dtype=torch.bool, device=src_logits.device)
            for batch_idx, t in enumerate(targets):
                if 'caption_attributes_padding_mask' in t:
                    sample_valid_mask = t['caption_attributes_padding_mask']  # [num_captions], True=valid, False=padding
                    # 广播到所有query
                    valid_mask_full[batch_idx, :, :] = sample_valid_mask.unsqueeze(0)
                else:
                    # 没有mask则全部有效
                    valid_mask_full[batch_idx, :, :] = True
            
            # Build weight (MAL style)
            pred_score = F.sigmoid(src_logits).detach()
            if self.mal_alpha is not None:
                weight = self.mal_alpha * pred_score.pow(self.gamma) * (1 - target) + target
            else:
                weight = pred_score.pow(self.gamma) * (1 - target) + target
            
            # Compute loss
            loss = F.binary_cross_entropy_with_logits(
                src_logits, target_score, 
                weight=weight, reduction='none'
            )
            loss = torch.nan_to_num(loss, nan=0.0)

            filtering_mask = torch.ones_like(loss)
            # matched_targets = target[idx]
            # foreground_mask = matched_targets + (1 - matched_targets) * 0.05  # 这里按你的要求，纯正样本
            # filtering_mask[idx] = foreground_mask
            
            # 不去计算正样本 和 未标注属性的损失
            # filtering_mask[idx] = target[idx]
            final_mask = valid_mask_full.float() * filtering_mask
            
            loss = loss * final_mask
            loss = loss.mean(-1).sum() * src_logits.shape[1] / num_boxes
        else:
            target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
            target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                        dtype=torch.int64, device=src_logits.device)
            target_classes[idx] = target_classes_o
            target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]
            
            target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
            target_score_o[idx] = ious.to(target_score_o.dtype)
            target_score = target_score_o.unsqueeze(-1) * target
            target_score = target_score.pow(self.gamma)
            
            pred_score = F.sigmoid(src_logits).detach()
            if self.mal_alpha != None:
                weight = self.mal_alpha * pred_score.pow(self.gamma) * (1 - target) + target
            else:
                weight = pred_score.pow(self.gamma) * (1 - target) + target

            # print(" ### DEIM-gamma{}-alpha{} ### ".format(self.gamma, self.mal_alpha))
            loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
            loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
            
        return {'loss_attr_mal': loss}

    def loss_attr_ot(self, outputs, targets, indices, num_boxes, **kwargs):
        """OT alignment for positives: temporarily disabled to save compute."""
        base = outputs['pred_boxes'] if 'pred_boxes' in outputs else torch.tensor(0., device=targets[0]['boxes'].device)
        return {'loss_attr_ot': base.sum() * 0.}

        batch_idx, src_idx = self._get_src_permutation_idx(indices)
        _, tgt_idx = self._get_tgt_permutation_idx(indices)
        if len(src_idx) == 0:
            return {'loss_attr_ot': proj_queries.sum() * 0.}

        img_feats = proj_queries[batch_idx, src_idx]  # (P, D)
        v_pos = F.normalize(img_feats, dim=-1).unsqueeze(1)  # (P, 1, D)

        text_slots, weights, valid_masks = [], [], []
        device = proj_queries.device
        max_attr_slots = text_groups.shape[2] - 1

        for pos, (b, t, q) in enumerate(zip(batch_idx.tolist(), tgt_idx.tolist(), src_idx.tolist())):
            cap_onehot = targets[b].get('caption_indices_onehot', None)
            if cap_onehot is None or cap_onehot.numel() == 0:
                continue
            cap_ids = (cap_onehot[t] > 0).nonzero(as_tuple=False).flatten()
            if cap_ids.numel() == 0:
                continue
            rand_idx = torch.randint(low=0, high=cap_ids.numel(), size=(1,), device=device).item()
            cap_id = int(cap_ids[rand_idx].item())
            if cap_id >= text_groups.shape[1]:
                raise ValueError('index out')

            group_feats = text_groups[b, cap_id]          # (1 + K, D)
            group_mask = text_group_mask[b, cap_id]       # (1 + K)
            if not group_mask.any():
                continue

            # normalize weights so column sum = 1 over valid slots
            weight_scores = group_mask.to(dtype=proj_queries.dtype)

            norm = weight_scores.sum()
            if norm > 0:
                weight_scores = weight_scores / norm

            text_slots.append(group_feats)
            weights.append(weight_scores)
            valid_masks.append(group_mask)

        if len(text_slots) == 0:
            return {'loss_attr_ot': proj_queries.sum() * 0.}

        text_slots = torch.stack(text_slots, dim=0)          # (P, 1+K, D)
        weights = torch.stack(weights, dim=0)                # (P, 1+K)
        valid_masks = torch.stack(valid_masks, dim=0)        # (P, 1+K)

        loss_ot = self.ot_align(
            v_pos,
            text_slots,
            w_col=weights,
            valid_mask=valid_masks
        )
        return {'loss_attr_ot': loss_ot}



    def loss_boxes(self, outputs, targets, indices, num_boxes, boxes_weight=None):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        losses = {}
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(generalized_box_iou(\
            box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)))
        loss_giou = loss_giou if boxes_weight is None else loss_giou * boxes_weight
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        return losses

    def loss_local(self, outputs, targets, indices, num_boxes, T=5):
        """Compute Fine-Grained Localization (FGL) Loss
            and Decoupled Distillation Focal (DDF) Loss. """

        losses = {}
        if 'pred_corners' in outputs:
            idx = self._get_src_permutation_idx(indices)
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

            pred_corners = outputs['pred_corners'][idx].reshape(-1, (self.reg_max+1))
            ref_points = outputs['ref_points'][idx].detach()
            with torch.no_grad():
                if self.fgl_targets_dn is None and 'is_dn' in outputs:
                        self.fgl_targets_dn= bbox2distance(ref_points, box_cxcywh_to_xyxy(target_boxes),
                                                        self.reg_max, outputs['reg_scale'], outputs['up'])
                if self.fgl_targets is None and 'is_dn' not in outputs:
                        self.fgl_targets = bbox2distance(ref_points, box_cxcywh_to_xyxy(target_boxes),
                                                        self.reg_max, outputs['reg_scale'], outputs['up'])

            target_corners, weight_right, weight_left = self.fgl_targets_dn if 'is_dn' in outputs else self.fgl_targets

            ious = torch.diag(box_iou(\
                        box_cxcywh_to_xyxy(outputs['pred_boxes'][idx]), box_cxcywh_to_xyxy(target_boxes))[0])
            weight_targets = ious.unsqueeze(-1).repeat(1, 1, 4).reshape(-1).detach()

            losses['loss_fgl'] = self.unimodal_distribution_focal_loss(
                pred_corners, target_corners, weight_right, weight_left, weight_targets, avg_factor=num_boxes)

            if 'teacher_corners' in outputs:
                pred_corners = outputs['pred_corners'].reshape(-1, (self.reg_max+1))
                target_corners = outputs['teacher_corners'].reshape(-1, (self.reg_max+1))
                if not torch.equal(pred_corners, target_corners):
                    weight_targets_local = outputs['teacher_logits'].sigmoid().max(dim=-1)[0]

                    mask = torch.zeros_like(weight_targets_local, dtype=torch.bool)
                    mask[idx] = True
                    mask = mask.unsqueeze(-1).repeat(1, 1, 4).reshape(-1)

                    weight_targets_local[idx] = ious.reshape_as(weight_targets_local[idx]).to(weight_targets_local.dtype)
                    weight_targets_local = weight_targets_local.unsqueeze(-1).repeat(1, 1, 4).reshape(-1).detach()

                    loss_match_local = weight_targets_local * (T ** 2) * (nn.KLDivLoss(reduction='none')
                    (F.log_softmax(pred_corners / T, dim=1), F.softmax(target_corners.detach() / T, dim=1))).sum(-1)
                    if 'is_dn' not in outputs:
                        batch_scale = 8 / outputs['pred_boxes'].shape[0]  # Avoid the influence of batch size per GPU
                        self.num_pos, self.num_neg = (mask.sum() * batch_scale) ** 0.5, ((~mask).sum() * batch_scale) ** 0.5
                    loss_match_local1 = loss_match_local[mask].mean() if mask.any() else 0
                    loss_match_local2 = loss_match_local[~mask].mean() if (~mask).any() else 0
                    losses['loss_ddf'] = (loss_match_local1 * self.num_pos + loss_match_local2 * self.num_neg) / (self.num_pos + self.num_neg)

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def _get_go_indices(self, indices, indices_aux_list):
        """Get a matching union set across all decoder layers. 其实也包括了encoder layer"""
        results = []
        for indices_aux in indices_aux_list:
            indices = [(torch.cat([idx1[0], idx2[0]]), torch.cat([idx1[1], idx2[1]]))
                        for idx1, idx2 in zip(indices.copy(), indices_aux.copy())]

        for ind in [torch.cat([idx[0][:, None], idx[1][:, None]], 1) for idx in indices]:
            unique, counts = torch.unique(ind, return_counts=True, dim=0)
            count_sort_indices = torch.argsort(counts, descending=True)
            unique_sorted = unique[count_sort_indices]
            column_to_row = {}
            for idx in unique_sorted:
                row_idx, col_idx = idx[0].item(), idx[1].item()
                if row_idx not in column_to_row:
                    column_to_row[row_idx] = col_idx
            final_rows = torch.tensor(list(column_to_row.keys()), device=ind.device)
            final_cols = torch.tensor(list(column_to_row.values()), device=ind.device)
            results.append((final_rows.long(), final_cols.long()))
        return results

    def _clear_cache(self):
        self.fgl_targets, self.fgl_targets_dn = None, None
        self.own_targets, self.own_targets_dn = None, None
        self.num_pos, self.num_neg = None, None

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'boxes': self.loss_boxes,
            'focal': self.loss_labels_focal,
            'vfl': self.loss_labels_vfl,
            'mal': self.loss_labels_mal,
            'attr_mal': self.loss_labels_attr_mal,
            'attr_ot': self.loss_attr_ot,
            'local': self.loss_local,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, epoch=0, **kwargs):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if 'aux' not in k}  # 将辅助分支抽出，只保留最后一层的输出

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets, epoch=epoch)['indices']
        self._clear_cache()

        # Get the matching union set across all decoder layers.
        if 'aux_outputs' in outputs:
            indices_aux_list, cached_indices, cached_indices_enc = [], [], []  # 所有aux头匹配结果的indices(包括decoder/encoder), 后面俩个是存储decoder和encoder的匹配索引
            aux_outputs_list = outputs['aux_outputs']  # aux_outputs 是decoder除了最后一层+第一层外的输出
            if 'pre_outputs' in outputs:  # pre_outputs是第一层的输出
                aux_outputs_list = outputs['aux_outputs'] + [outputs['pre_outputs']]
            for i, aux_outputs in enumerate(aux_outputs_list):
                indices_aux = self.matcher(aux_outputs, targets, epoch=epoch)['indices']
                cached_indices.append(indices_aux)
                indices_aux_list.append(indices_aux)
            for i, aux_outputs in enumerate(outputs['enc_aux_outputs']):
                indices_enc = self.matcher(aux_outputs, targets, epoch=epoch)['indices']
                cached_indices_enc.append(indices_enc)
                indices_aux_list.append(indices_enc)
            indices_go = self._get_go_indices(indices, indices_aux_list)  # 获取所有层匹配的并集

            num_boxes_go = sum(len(x[0]) for x in indices_go)  # 统计并集中的正样本总数
            num_boxes_go = torch.as_tensor([num_boxes_go], dtype=torch.float, device=next(iter(outputs.values())).device)
            if is_dist_available_and_initialized():
                torch.distributed.all_reduce(num_boxes_go)
            num_boxes_go = torch.clamp(num_boxes_go / get_world_size(), min=1).item()  # 所有卡正样本数量mean()
        else:
            assert 'aux_outputs' in outputs, ''

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["boxes"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses, main loss
        losses = {}
        for loss in self.losses:
            use_uni_set = self.use_uni_set and (loss in ['boxes', 'local'])
            indices_in = indices_go if use_uni_set else indices  # 计算box loss时，需要使用到所有层匹配的并集
            num_boxes_in = num_boxes_go if use_uni_set else num_boxes
            meta = self.get_loss_meta_info(loss, outputs, targets, indices_in)
            l_dict = self.get_loss(loss, outputs, targets, indices_in, num_boxes_in, **meta)
            l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
            losses.update(l_dict)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                if 'local' in self.losses:      # only work for local loss
                    aux_outputs['up'], aux_outputs['reg_scale'] = outputs['up'], outputs['reg_scale']
                for loss in self.losses:
                    use_uni_set = self.use_uni_set and (loss in ['boxes', 'local'])
                    indices_in = indices_go if use_uni_set else cached_indices[i]
                    num_boxes_in = num_boxes_go if use_uni_set else num_boxes
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_in)
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices_in, num_boxes_in, **meta)

                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_aux_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # In case of auxiliary traditional head output at first decoder layer. just for dfine
        if 'pre_outputs' in outputs:
            aux_outputs = outputs['pre_outputs']
            for loss in self.losses:
                use_uni_set = self.use_uni_set and (loss in ['boxes', 'local'])
                indices_in = indices_go if use_uni_set else cached_indices[-1]
                num_boxes_in = num_boxes_go if use_uni_set else num_boxes
                meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_in)
                l_dict = self.get_loss(loss, aux_outputs, targets, indices_in, num_boxes_in, **meta)

                l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                l_dict = {k + '_pre': v for k, v in l_dict.items()}
                losses.update(l_dict)

        # In case of encoder auxiliary losses.
        if 'enc_aux_outputs' in outputs:
            assert 'enc_meta' in outputs, ''
            class_agnostic = outputs['enc_meta']['class_agnostic']
            if class_agnostic:
                orig_num_classes = self.num_classes
                self.num_classes = 1
                enc_targets = copy.deepcopy(targets)
                for t in enc_targets:
                    t['labels'] = torch.zeros_like(t["labels"])
            else:
                enc_targets = targets

            for i, aux_outputs in enumerate(outputs['enc_aux_outputs']):
                for loss in self.losses:
                    use_uni_set = self.use_uni_set and (loss == 'boxes')
                    indices_in = indices_go if use_uni_set else cached_indices_enc[i]
                    num_boxes_in = num_boxes_go if use_uni_set else num_boxes
                    meta = self.get_loss_meta_info(loss, aux_outputs, enc_targets, indices_in)
                    l_dict = self.get_loss(loss, aux_outputs, enc_targets, indices_in, num_boxes_in, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_enc_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

            if class_agnostic:
                self.num_classes = orig_num_classes

        # In case of cdn auxiliary losses.
        if 'dn_outputs' in outputs:
            assert 'dn_meta' in outputs, ''
            indices_dn = self.get_cdn_matched_indices(outputs['dn_meta'], targets)
            dn_num_boxes = num_boxes * outputs['dn_meta']['dn_num_group']

            for i, aux_outputs in enumerate(outputs['dn_outputs']):
                if 'local' in self.losses:      # only work for local loss
                    aux_outputs['is_dn'] = True
                    aux_outputs['up'], aux_outputs['reg_scale'] = outputs['up'], outputs['reg_scale']
                for loss in self.losses:
                    if loss == 'attr_ot':
                        continue  # skip OT on DN branches
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_dn)
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices_dn, dn_num_boxes, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_dn_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

            # In case of auxiliary traditional head output at first decoder layer, just for dfine
            if 'dn_pre_outputs' in outputs:
                aux_outputs = outputs['dn_pre_outputs']
                for loss in self.losses:
                    if loss == 'attr_ot':
                        continue  # skip OT on DN branches
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_dn)
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices_dn, dn_num_boxes, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + '_dn_pre': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # For debugging Objects365 pre-train.
        losses = {k:torch.nan_to_num(v, nan=0.0) for k, v in losses.items()}
        return losses

    def get_loss_meta_info(self, loss, outputs, targets, indices):
        if self.boxes_weight_format is None:
            return {}

        src_boxes = outputs['pred_boxes'][self._get_src_permutation_idx(indices)]
        target_boxes = torch.cat([t['boxes'][j] for t, (_, j) in zip(targets, indices)], dim=0)

        if self.boxes_weight_format == 'iou':
            iou, _ = box_iou(box_cxcywh_to_xyxy(src_boxes.detach()), box_cxcywh_to_xyxy(target_boxes))
            iou = torch.diag(iou)
        elif self.boxes_weight_format == 'giou':
            iou = torch.diag(generalized_box_iou(\
                box_cxcywh_to_xyxy(src_boxes.detach()), box_cxcywh_to_xyxy(target_boxes)))
        else:
            raise AttributeError()

        if loss in ('boxes', ):
            meta = {'boxes_weight': iou}
        elif loss in ('vfl', 'mal'):
            meta = {'values': iou}
        else:
            meta = {}

        return meta

    @staticmethod
    def get_cdn_matched_indices(dn_meta, targets):
        """get_cdn_matched_indices
        """
        dn_positive_idx, dn_num_group = dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
        num_gts = [len(t['boxes']) for t in targets]
        device = targets[0]['boxes'].device

        dn_match_indices = []
        for i, num_gt in enumerate(num_gts):
            if num_gt > 0:
                gt_idx = torch.arange(num_gt, dtype=torch.int64, device=device)
                gt_idx = gt_idx.tile(dn_num_group)
                assert len(dn_positive_idx[i]) == len(gt_idx)
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else:
                dn_match_indices.append((torch.zeros(0, dtype=torch.int64, device=device), \
                    torch.zeros(0, dtype=torch.int64,  device=device)))

        return dn_match_indices


    def feature_loss_function(self, fea, target_fea):
        loss = (fea - target_fea) ** 2 * ((fea > 0) | (target_fea > 0)).float()
        return torch.abs(loss)


    def unimodal_distribution_focal_loss(self, pred, label, weight_right, weight_left, weight=None, reduction='sum', avg_factor=None):
        dis_left = label.long()
        dis_right = dis_left + 1

        loss = F.cross_entropy(pred, dis_left, reduction='none') * weight_left.reshape(-1) \
             + F.cross_entropy(pred, dis_right, reduction='none') * weight_right.reshape(-1)

        if weight is not None:
            weight = weight.float()
            loss = loss * weight

        if avg_factor is not None:
            loss = loss.sum() / avg_factor
        elif reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()

        return loss

    def get_gradual_steps(self, outputs):
        num_layers = len(outputs['aux_outputs']) + 1 if 'aux_outputs' in outputs else 1
        step = .5 / (num_layers - 1)
        opt_list = [.5  + step * i for i in range(num_layers)] if num_layers > 1 else [1]
        return opt_list
