"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
Modifications Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
"""

import torch

from .utils import inverse_sigmoid
from .box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh



def get_contrastive_denoising_training_group(targets,
                                             num_classes,
                                             num_queries,
                                             class_embed,
                                             num_denoising=100,
                                             label_noise_ratio=0.5,
                                             box_noise_scale=1.0,
                                             text_features=None):
    """cnd"""
    if num_denoising <= 0:
        return None, None, None, None

    device = targets[0]['boxes'].device
    num_gts = [len(t['boxes']) if t['boxes'] is not None else 0 for t in targets]

    max_gt_num = max(num_gts)
    # 因为num_classes == -1，代表embeddings_dn_sim
    if max_gt_num == 0:
        return None, None, None, None

    num_group = num_denoising // max_gt_num
    num_group = 1 if num_group == 0 else num_group
    # pad gt to max_num of a batch
    bs = len(num_gts)

    input_query_class = torch.full([bs, max_gt_num], num_classes, dtype=torch.int32, device=device)
    input_query_bbox = torch.zeros([bs, max_gt_num, 4], device=device)
    pad_gt_mask = torch.zeros([bs, max_gt_num], dtype=torch.bool, device=device)

    for i in range(bs):
        num_gt = num_gts[i]
        if num_gt > 0:
            if num_classes != -1:
                input_query_class[i, :num_gt] = targets[i]['labels']
            input_query_bbox[i, :num_gt] = targets[i]['boxes']
            pad_gt_mask[i, :num_gt] = 1
    # each group has positive and negative queries.
    input_query_class = input_query_class.tile([1, 2 * num_group])
    input_query_bbox = input_query_bbox.tile([1, 2 * num_group, 1])
    pad_gt_mask = pad_gt_mask.tile([1, 2 * num_group])
    # positive and negative mask
    negative_gt_mask = torch.zeros([bs, max_gt_num * 2, 1], device=device)
    negative_gt_mask[:, max_gt_num:] = 1
    negative_gt_mask = negative_gt_mask.tile([1, num_group, 1])
    positive_gt_mask = 1 - negative_gt_mask
    # contrastive denoising training positive index
    positive_gt_mask = positive_gt_mask.squeeze(-1) * pad_gt_mask
    dn_positive_idx = torch.nonzero(positive_gt_mask)[:, 1]
    dn_positive_idx = torch.split(dn_positive_idx, [n * num_group for n in num_gts])
    # total denoising queries
    num_denoising = int(max_gt_num * 2 * num_group)

    if label_noise_ratio > 0:
        if num_classes != -1:
            mask = torch.rand_like(input_query_class, dtype=torch.float) < (label_noise_ratio * 0.5)
            # randomly put a new one here
            new_label = torch.randint_like(mask, 0, num_classes, dtype=input_query_class.dtype)
            input_query_class = torch.where(mask & pad_gt_mask, new_label, input_query_class)
            input_query_logits = class_embed(input_query_class)
        else:
            # 使用text_features来模拟visual embeddings
            # text_features shape: [bs, num_caption, dim]
            text_features_proj = class_embed(text_features)  # [bs, num_caption, embed_dim]
            
            # 初始化input_query_logits，shape: [bs, max_gt_num * 2 * num_group, embed_dim]
            embed_dim = text_features_proj.shape[-1]
            input_query_logits = torch.zeros([bs, max_gt_num * 2 * num_group, embed_dim], 
                                            dtype=text_features_proj.dtype, device=device)
            
            # 为每个batch填充GT box对应的text embeddings
            for i in range(bs):
                num_gt = num_gts[i]
                if num_gt > 0:
                    # 获取当前样本的caption索引列表
                    caption_indices_list = targets[i]['caption_indices_list']  # list of list, length = num_gt
                    
                    # 批量计算所有GT embeddings
                    gt_embeds = torch.zeros(num_gt, embed_dim, dtype=text_features_proj.dtype, device=device)
                    for j, caption_indices in enumerate(caption_indices_list[:num_gt]):
                        if len(caption_indices) > 0:
                            # 获取所有对应的text embeddings并取平均
                            gt_embeds[j] = text_features_proj[i, caption_indices].mean(dim=0)
                    
                    # 使用repeat和索引一次性填充所有group的正样本位置
                    if num_group > 1:
                        # 重复gt_embeds以覆盖所有group
                        gt_embeds_repeated = gt_embeds.repeat(num_group, 1)  # [num_gt * num_group, embed_dim]
                        # 计算所有正样本的索引位置
                        indices = torch.cat([torch.arange(num_gt, device=device) + g * max_gt_num * 2 
                                            for g in range(num_group)])
                        input_query_logits[i, indices] = gt_embeds_repeated
                    else:
                        # 如果只有一个group，直接填充
                        input_query_logits[i, :num_gt] = gt_embeds

            # 添加标签噪声：随机替换一些embeddings
            # 生成噪声mask，只对pad_gt_mask为True的位置添加噪声
            noise_mask = torch.rand([bs, max_gt_num * 2 * num_group], device=device) < (label_noise_ratio * 0.5)
            noise_mask = noise_mask & pad_gt_mask  # 只对有效位置添加噪声
            
            # 为需要添加噪声的位置随机选择text embeddings
            for i in range(bs):
                caption_padding_mask = targets[i].get('caption_padding_mask', None)
                
                noise_indices = torch.nonzero(noise_mask[i]).squeeze(-1)
                # num_caption = text_features_proj.shape[1]  # 当前batch样本的caption数量
                # noise_indices = torch.nonzero(noise_mask[i]).squeeze(-1)
                
                if len(noise_indices) > 0:
                    if caption_padding_mask is not None:
                        # 只从有效的caption中选择（padding_mask为True的位置）
                        valid_caption_indices = torch.nonzero(caption_padding_mask).squeeze(-1)
                        num_valid_captions = len(valid_caption_indices)
                        
                        if num_valid_captions > 0:
                            # 从有效caption索引中随机选择
                            random_valid_idx = torch.randint(0, num_valid_captions, (len(noise_indices),), device=device)
                            random_caption_indices = valid_caption_indices[random_valid_idx]
                            input_query_logits[i, noise_indices] = text_features_proj[i, random_caption_indices]
                    else:
                        # 如果没有padding_mask，则假设所有caption都是有效的
                        num_caption = text_features_proj.shape[1]
                        random_caption_indices = torch.randint(0, num_caption, (len(noise_indices),), device=device)
                        input_query_logits[i, noise_indices] = text_features_proj[i, random_caption_indices]
                
    if box_noise_scale > 0:
        known_bbox = box_cxcywh_to_xyxy(input_query_bbox)
        diff = torch.tile(input_query_bbox[..., 2:] * 0.5, [1, 1, 2]) * box_noise_scale
        rand_sign = torch.randint_like(input_query_bbox, 0, 2) * 2.0 - 1.0
        rand_part = torch.rand_like(input_query_bbox)
        rand_part = (rand_part + 1.0) * negative_gt_mask + rand_part * (1 - negative_gt_mask)
        known_bbox += (rand_sign * rand_part * diff)
        known_bbox = torch.clip(known_bbox, min=0.0, max=1.0)
        input_query_bbox = box_xyxy_to_cxcywh(known_bbox)
        input_query_bbox[input_query_bbox < 0] *= -1
        input_query_bbox_unact = inverse_sigmoid(input_query_bbox)

    

    tgt_size = num_denoising + num_queries
    attn_mask = torch.full([tgt_size, tgt_size], False, dtype=torch.bool, device=device)
    # match query cannot see the reconstruction
    attn_mask[num_denoising:, :num_denoising] = True

    # reconstruct cannot see each other
    for i in range(num_group):
        if i == 0:
            attn_mask[max_gt_num * 2 * i: max_gt_num * 2 * (i + 1), max_gt_num * 2 * (i + 1): num_denoising] = True
        if i == num_group - 1:
            attn_mask[max_gt_num * 2 * i: max_gt_num * 2 * (i + 1), :max_gt_num * i * 2] = True
        else:
            attn_mask[max_gt_num * 2 * i: max_gt_num * 2 * (i + 1), max_gt_num * 2 * (i + 1): num_denoising] = True
            attn_mask[max_gt_num * 2 * i: max_gt_num * 2 * (i + 1), :max_gt_num * 2 * i] = True

    dn_meta = {
        "dn_positive_idx": dn_positive_idx,
        "dn_num_group": num_group,
        "dn_num_split": [num_denoising, num_queries]
    }

    # print(input_query_class.shape) # torch.Size([4, 196, 256])
    # print(input_query_bbox.shape) # torch.Size([4, 196, 4])
    # print(attn_mask.shape) # torch.Size([496, 496])

    return input_query_logits, input_query_bbox_unact, attn_mask, dn_meta
