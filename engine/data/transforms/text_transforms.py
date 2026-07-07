# Copyright (c) Tencent Inc. All rights reserved.
# copy from yolo-world YOLO-World/yolo_world/datasets/transformers/mm_transforms.py

import json
import random
from typing import Tuple, Any, Dict
import torch
import numpy as np
from ...core import register


@register()
class RandomLoadText:

    def __init__(self,
                 text_path: str = None,
                 prompt_format: str = '{}',
                 num_neg_samples: Tuple[int, int] = (80, 80),
                 max_num_samples: int = 80,
                 padding_to_max: bool = False,
                 padding_value: str = '') -> None:
        self.prompt_format = prompt_format
        self.num_neg_samples = num_neg_samples
        self.max_num_samples = max_num_samples
        self.padding_to_max = padding_to_max
        self.padding_value = padding_value
        if text_path is not None:
            with open(text_path, 'r') as f:
                self.captions = json.load(f)

    # def __call__(self, results: dict) -> dict:
        # assert 'texts' in results or hasattr(self, 'captions'), (
        #     'No texts found in results.')
        
    def __call__(self, *inputs: Any) -> Any:
        if len(inputs) == 1:
            inputs = inputs[0]
        image, results, dataset = inputs  # results=target

        captions = results.get(
            'captions',
            None)

        num_classes = len(captions)
        if 'labels' in results:
            gt_label_tag = 'labels'
        else:
            raise ValueError('No valid labels found in results.')
        
        positive_labels = set(results[gt_label_tag].tolist())

        if len(positive_labels) > self.max_num_samples:
            positive_labels = set(random.sample(list(positive_labels),
                                  k=self.max_num_samples))

        num_neg_samples = min(
            min(num_classes, self.max_num_samples) - len(positive_labels),
            random.randint(*self.num_neg_samples))
        candidate_neg_labels = []
        for idx in range(num_classes):
            if idx not in positive_labels:
                candidate_neg_labels.append(idx)
        negative_labels = random.sample(
            candidate_neg_labels, k=num_neg_samples)

        sampled_labels = list(positive_labels) + list(negative_labels)
        random.shuffle(sampled_labels)

        label2ids = {label: i for i, label in enumerate(sampled_labels)}

        gt_valid_mask = np.zeros(len(results['boxes']), dtype=bool)
        for idx, label in enumerate(results[gt_label_tag].tolist()):
            if label in label2ids:
                gt_valid_mask[idx] = True
                results[gt_label_tag][idx] = label2ids[label]
        results['boxes'] = results['boxes'][gt_valid_mask]
        results[gt_label_tag] = results[gt_label_tag][gt_valid_mask]

        texts = []
        for label in sampled_labels:
            cls_caps = captions[label]
            assert len(cls_caps) > 0
            cap_id = random.randrange(len(cls_caps))
            sel_cls_cap = self.prompt_format.format(cls_caps[cap_id])
            texts.append(sel_cls_cap)

        if self.padding_to_max:
            num_valid_labels = len(positive_labels) + len(negative_labels)
            num_padding = self.max_num_samples - num_valid_labels
            if num_padding > 0:
                texts += [self.padding_value] * num_padding

        results['captions'] = texts
        
        return image, results, dataset


@register()
class LoadText:

    def __init__(self,
                 text_path: str = None,
                 prompt_format: str = '{}',
                 multi_prompt_flag: str = '/') -> None:
        self.prompt_format = prompt_format
        self.multi_prompt_flag = multi_prompt_flag
        if text_path is not None:
            with open(text_path, 'r') as f:
                self.captions = json.load(f)

    def __call__(self, *inputs: Any) -> Any:
        if len(inputs) == 1:
            inputs = inputs[0]
        image, results, dataset = inputs  # results=target

        captions = results.get(
            'captions',
            None)

        texts = []
        for idx, cls_caps in enumerate(captions):
            assert len(cls_caps) > 0
            sel_cls_cap = cls_caps[0]
            # sel_cls_cap = cls_caps
            sel_cls_cap = self.prompt_format.format(sel_cls_cap)
            texts.append(sel_cls_cap)

        results['captions'] = texts

        return image, results, dataset

@register()
class RandomLoadCaptions:
    """
    随机采样和打乱captions的数据增强（Training Phase）
    
    策略：
    1. 采样正负样本 Captions 并打乱。
    2. 构建 Captions 的 One-Hot 矩阵。
    3. 提取属性，构建属性 One-Hot 和 cap_to_attr_map。
    """
    def __init__(self,
                 num_neg_samples: Tuple[int, int] = (30, 30),
                 max_num_samples: int = 80,       # Captions 最大数量
                 max_attrs_per_caption: int = 10, # 单句最大属性数，用于构建 Map
                 padding_to_max: bool = False,
                 padding_value: str = '',
                 random_sample_positive: bool = False,
                 attributes_align_use=False) -> None:
        self.num_neg_samples = num_neg_samples
        self.max_num_samples = max_num_samples
        self.max_attrs_per_caption = max_attrs_per_caption
        self.padding_to_max = padding_to_max
        self.padding_value = padding_value
        self.random_sample_positive = random_sample_positive
        
        self.attributes_align_use = attributes_align_use
        # 全局属性最大长度 = caption数 * 单句属性数
        self.max_num_attributes_samples = max_num_samples * max_attrs_per_caption

    def build_attributes_tensors(self, target):
        """
        构建属性相关的 Tensors, Mask, One-Hot 和 Map
        """
        captions = target['captions']  # 采样、打乱并 Pad 后的 Captions
        caption_attributes_dict = target.get('caption_attributes_dict', {})
        caption_indices_list = target['caption_indices_list']
        
        num_boxes = len(target['boxes'])
        # 当前 caption 列表的长度 (可能是 max_num_samples)
        current_num_captions = len(captions) 
        
        # ===== 1. 全局去重属性列表与映射 =====
        attributes_global = []
        attr_to_idx = {}
        cap_to_attr_map = torch.full((current_num_captions, self.max_attrs_per_caption), -1, dtype=torch.long)
        max_attr_global = self.max_num_attributes_samples

        for cap_idx, caption in enumerate(captions):
            # 跳过 Padding 的 Caption
            if caption == self.padding_value or caption == '':
                continue

            if caption not in caption_attributes_dict:
                raise ValueError(f'Caption {caption} has no attributes')

            raw_attrs = caption_attributes_dict[caption]
            attr_indices = []
            for a in raw_attrs:
                if a in attr_to_idx:
                    attr_indices.append(attr_to_idx[a])
                elif len(attr_to_idx) < max_attr_global:
                    attr_to_idx[a] = len(attr_to_idx)
                    attributes_global.append(a)
                    attr_indices.append(attr_to_idx[a])
                # 超上限则忽略

            valid_len = min(len(attr_indices), self.max_attrs_per_caption)
            if valid_len > 0:
                cap_to_attr_map[cap_idx, :valid_len] = torch.as_tensor(attr_indices[:valid_len], dtype=torch.long)

        # ===== 2. Attribute Padding (全局列表) =====
        caption_attributes_list = attributes_global
        valid_caption_attr_len = len(caption_attributes_list)
        if valid_caption_attr_len < self.max_num_attributes_samples:
            caption_attributes_list.extend([self.padding_value] * (self.max_num_attributes_samples - valid_caption_attr_len))
        else:
            # 截断
            caption_attributes_list = caption_attributes_list[:self.max_num_attributes_samples]
            valid_caption_attr_len = self.max_num_attributes_samples
        
        # ===== 4. Attribute Padding Mask =====
        caption_attributes_padding_mask = torch.cat([
            torch.ones(valid_caption_attr_len, dtype=torch.bool),
            torch.zeros(self.max_num_attributes_samples - valid_caption_attr_len, dtype=torch.bool)
        ])
        
        # ===== 5. Attribute One-Hot (Box -> Attributes) =====
        caption_attributes_onehot = torch.zeros((num_boxes, self.max_num_attributes_samples), dtype=torch.float32)
        
        for box_idx, caption_indices in enumerate(caption_indices_list):
            for cap_idx in caption_indices:
                # 确保 cap_idx 有效
                if cap_idx < current_num_captions:
                    # 使用 map tensor 快速查找
                    attr_indices = cap_to_attr_map[cap_idx]
                    
                    # 过滤无效值 (-1) 和越界值
                    valid_indices = attr_indices[
                        (attr_indices != -1) & (attr_indices < self.max_num_attributes_samples)
                    ]
                    
                    if len(valid_indices) > 0:
                        caption_attributes_onehot[box_idx, valid_indices] = 1.0
        
        # ===== 更新 Target =====
        target['caption_attributes'] = caption_attributes_list
        target['caption_attributes_padding_mask'] = caption_attributes_padding_mask
        target['caption_attributes_onehot'] = caption_attributes_onehot
        
        target['cap_to_attr_map'] = cap_to_attr_map
        
        return target

    def __call__(self, *inputs: Any) -> Any:
        # caption和attr的one_hot矩阵都不进行归一化，因为最终采用use_sigmoid，因此都是单独预测的，那么visual应该和每个caption以及每个属性的相似度都接近1
        if len(inputs) == 1:
            inputs = inputs[0]
        image, target, dataset = inputs
        
        captions = target.get('captions', [])
        num_captions = len(captions)
        boxes = target.get('boxes', torch.empty((0, 4)))
        num_boxes = len(boxes)
        caption_indices_list = target.get('caption_indices_list', [])
        
        assert len(caption_indices_list) == num_boxes
        
        if num_captions == 0:
            raise ValueError("Caption num = 0, please check data.")
        
        # 1. 收集正样本
        positive_caption_indices = set()
        for cap_indices in caption_indices_list:
            positive_caption_indices.update(cap_indices)
        
        # 2. 随机采样正样本 (如果开启)
        if self.random_sample_positive and len(positive_caption_indices) > 1:
            num_samples_positive = random.randint(1, len(positive_caption_indices))
            positive_caption_indices = set(random.sample(list(positive_caption_indices), k=num_samples_positive))
            
            # 过滤 boxes
            valid_boxes_mask = []
            filtered_caption_indices_list = []
            for cap_indices in caption_indices_list:
                valid_caps = [idx for idx in cap_indices if idx in positive_caption_indices]
                if len(valid_caps) > 0:
                    valid_boxes_mask.append(True)
                    filtered_caption_indices_list.append(valid_caps)
                else:
                    valid_boxes_mask.append(False)
            boxes = boxes[torch.tensor(valid_boxes_mask, dtype=torch.bool)]
            caption_indices_list = filtered_caption_indices_list
            num_boxes = len(boxes)
        
        # 3. 正样本截断
        if len(positive_caption_indices) > self.max_num_samples:
            positive_caption_indices = set(random.sample(list(positive_caption_indices), k=self.max_num_samples))
            # 再次过滤 boxes (逻辑同上)
            valid_boxes_mask = []
            filtered_caption_indices_list = []
            for cap_indices in caption_indices_list:
                valid_caps = [idx for idx in cap_indices if idx in positive_caption_indices]
                if len(valid_caps) > 0:
                    valid_boxes_mask.append(True)
                    filtered_caption_indices_list.append(valid_caps)
                else:
                    valid_boxes_mask.append(False)
            boxes = boxes[torch.tensor(valid_boxes_mask, dtype=torch.bool)]
            caption_indices_list = filtered_caption_indices_list
            num_boxes = len(boxes)
        
        # 4. 负样本采样
        num_positive = len(positive_caption_indices)
        max_neg_samples = min(
            min(num_captions, self.max_num_samples) - num_positive,
            random.randint(*self.num_neg_samples)
        )
        candidate_neg_indices = [idx for idx in range(num_captions) if idx not in positive_caption_indices]
        num_neg_samples = min(max_neg_samples, len(candidate_neg_indices))
        negative_caption_indices = set(random.sample(candidate_neg_indices, k=num_neg_samples) if num_neg_samples > 0 else [])
        
        # 5. 合并 & Shuffle
        sampled_caption_indices = list(positive_caption_indices) + list(negative_caption_indices)
        random.shuffle(sampled_caption_indices)
        
        old_to_new_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(sampled_caption_indices)}
        
        # 6. 更新 Captions 和 List
        new_captions = [captions[idx] for idx in sampled_caption_indices]
        new_caption_indices_list = []
        for cap_indices in caption_indices_list:
            new_indices = [old_to_new_mapping[old_idx] for old_idx in cap_indices]
            new_indices.sort()
            new_caption_indices_list.append(new_indices)
        
        num_sampled_captions = len(new_captions)
        
        # 7. Padding (Captions)
        caption_padding_mask = torch.ones(num_sampled_captions, dtype=torch.bool)
        if self.padding_to_max:
            num_padding = self.max_num_samples - num_sampled_captions
            if num_padding > 0:
                new_captions += [self.padding_value] * num_padding
                caption_padding_mask = torch.cat([caption_padding_mask, torch.zeros(num_padding, dtype=torch.bool)])
        
        final_num_captions = len(new_captions)
        
        # 8. 重建 Caption One-Hot
        if num_boxes > 0 and final_num_captions > 0:
            caption_indices_onehot = torch.zeros((num_boxes, final_num_captions), dtype=torch.float32)
            for box_idx, cap_indices in enumerate(new_caption_indices_list):
                for cap_idx in cap_indices:
                    if 0 <= cap_idx < final_num_captions:
                        caption_indices_onehot[box_idx, cap_idx] = 1.0
        elif num_boxes == 0 and final_num_captions > 0:
            caption_indices_onehot = torch.zeros((0, final_num_captions), dtype=torch.float32)
        else:
            raise ValueError("final_num_captions == 0")

        assert len(new_caption_indices_list) == len(boxes)
        
        # 更新 Target
        target.update({
            'captions': new_captions,
            'num_captions': final_num_captions,
            'boxes': boxes,
            'num_regions': num_boxes,
            'caption_indices_list': new_caption_indices_list,
            'caption_indices_onehot': caption_indices_onehot,
            'caption_padding_mask': caption_padding_mask,
        })
        
        # 9. 属性处理 (含 cap_to_attr_map)
        if self.attributes_align_use and 'caption_attributes_dict' in target:
            target = self.build_attributes_tensors(target)
        
        return image, target, dataset
    

@register()
class RandomLoadCaptions_Det:
    """
    检测任务的Caption采样和处理（Training Phase）

    策略：
    1. 采样逻辑与 RandomLoadText 一致：基于 labels 采样正负样本并打乱
    2. 通过 labels 构建 caption_indices_list、caption_indices_onehot 和 caption_padding_mask
    3. (可选) 直接将 captions 作为属性字段，便于与属性分支/Grounding 对齐
    """
    def __init__(self,
                 prompt_format: str = '{}',
                 num_neg_samples: Tuple[int, int] = (80, 80),
                 max_num_samples: int = 80,
                 padding_to_max: bool = False,
                 padding_value: str = '',
                 max_attrs_per_caption: int = 10,
                 attributes_align_use: bool = False) -> None:
        self.prompt_format = prompt_format
        self.num_neg_samples = num_neg_samples
        self.max_num_samples = max_num_samples
        self.padding_to_max = padding_to_max
        self.padding_value = padding_value
        self.max_attrs_per_caption = max_attrs_per_caption
        self.attributes_align_use = attributes_align_use
        self.max_num_attributes_samples = max_num_samples * max_attrs_per_caption

    def build_attributes_tensors(self, target):
        """
        与 RandomLoadCaptions 保持一致的属性构建逻辑
        """
        captions = target['captions']
        caption_attributes_dict = target.get('caption_attributes_dict', {})
        caption_indices_list = target['caption_indices_list']

        num_boxes = len(target['boxes'])
        current_num_captions = len(captions)

        attributes_global = []
        attr_to_idx = {}
        cap_to_attr_map = torch.full((current_num_captions, self.max_attrs_per_caption), -1, dtype=torch.long)
        max_attr_global = self.max_num_attributes_samples

        for cap_idx, caption in enumerate(captions):
            if caption == self.padding_value or caption == '':
                continue

            if caption not in caption_attributes_dict:
                raise ValueError(f'Caption {caption} has no attributes')

            raw_attrs = caption_attributes_dict[caption]
            attr_indices = []
            for a in raw_attrs:
                if a in attr_to_idx:
                    attr_indices.append(attr_to_idx[a])
                elif len(attr_to_idx) < max_attr_global:
                    attr_to_idx[a] = len(attr_to_idx)
                    attributes_global.append(a)
                    attr_indices.append(attr_to_idx[a])

            valid_len = min(len(attr_indices), self.max_attrs_per_caption)
            if valid_len > 0:
                cap_to_attr_map[cap_idx, :valid_len] = torch.as_tensor(attr_indices[:valid_len], dtype=torch.long)

        caption_attributes_list = attributes_global
        valid_caption_attr_len = len(caption_attributes_list)
        if valid_caption_attr_len < self.max_num_attributes_samples:
            caption_attributes_list.extend([self.padding_value] * (self.max_num_attributes_samples - valid_caption_attr_len))
        else:
            caption_attributes_list = caption_attributes_list[:self.max_num_attributes_samples]
            valid_caption_attr_len = self.max_num_attributes_samples

        caption_attributes_padding_mask = torch.cat([
            torch.ones(valid_caption_attr_len, dtype=torch.bool),
            torch.zeros(self.max_num_attributes_samples - valid_caption_attr_len, dtype=torch.bool)
        ])

        caption_attributes_onehot = torch.zeros((num_boxes, self.max_num_attributes_samples), dtype=torch.float32)
        for box_idx, caption_indices in enumerate(caption_indices_list):
            for cap_idx in caption_indices:
                if cap_idx < current_num_captions:
                    attr_indices = cap_to_attr_map[cap_idx]
                    valid_indices = attr_indices[
                        (attr_indices != -1) & (attr_indices < self.max_num_attributes_samples)
                    ]
                    if len(valid_indices) > 0:
                        caption_attributes_onehot[box_idx, valid_indices] = 1.0

        target['caption_attributes'] = caption_attributes_list
        target['caption_attributes_padding_mask'] = caption_attributes_padding_mask
        target['caption_attributes_onehot'] = caption_attributes_onehot
        target['cap_to_attr_map'] = cap_to_attr_map
        return target

    def __call__(self, *inputs: Any) -> Any:
        if len(inputs) == 1:
            inputs = inputs[0]
        image, target, dataset = inputs

        captions = target.get('captions', [])
        num_captions = len(captions)

        if 'labels' in target:
            gt_label_tag = 'labels'
        else:
            raise ValueError('No valid labels found in target.')

        if num_captions == 0:
            raise ValueError("Caption num = 0, please check data.")

        # ==== 1) 收集正样本 caption 索引 (与 RandomLoadText 一致) ====
        positive_labels = set(target[gt_label_tag].tolist())

        # 如果正样本数超过 max_num_samples，随机采样
        if len(positive_labels) > self.max_num_samples:
            positive_labels = set(random.sample(list(positive_labels),
                                  k=self.max_num_samples))

        # ==== 2) 负采样 (与 RandomLoadText 一致) ====
        num_neg_samples = min(
            min(num_captions, self.max_num_samples) - len(positive_labels),
            random.randint(*self.num_neg_samples))
        candidate_neg_labels = []
        for idx in range(num_captions):
            if idx not in positive_labels:
                candidate_neg_labels.append(idx)
        negative_labels = random.sample(
            candidate_neg_labels, k=num_neg_samples)

        # ==== 3) 合并并打乱 (与 RandomLoadText 一致) ====
        sampled_labels = list(positive_labels) + list(negative_labels)
        random.shuffle(sampled_labels)

        label2ids = {label: i for i, label in enumerate(sampled_labels)}

        # ==== 4) 过滤 boxes 和更新 labels (与 RandomLoadText 一致) ====
        gt_valid_mask = np.zeros(len(target['boxes']), dtype=bool)
        for idx, label in enumerate(target[gt_label_tag].tolist()):
            if label in label2ids:
                gt_valid_mask[idx] = True
                target[gt_label_tag][idx] = label2ids[label]
        target['boxes'] = target['boxes'][gt_valid_mask]
        target[gt_label_tag] = target[gt_label_tag][gt_valid_mask]
        # 重新计算 area，避免长度不一致
        boxes_filtered = target['boxes']
        if isinstance(boxes_filtered, torch.Tensor):
            target['area'] = (boxes_filtered[:, 2] * boxes_filtered[:, 3]).clone()
        else:
            target['area'] = boxes_filtered[:, 2] * boxes_filtered[:, 3]

        num_boxes = len(target['boxes'])

        # ==== 5) 重建 captions (支持多个caption随机选择) ====
        def _pick_cap(cap):
            if isinstance(cap, list):
                if len(cap) == 0:
                    return self.padding_value
                cap_id = random.randrange(len(cap))
                return cap[cap_id]
            return cap

        texts = []
        for label in sampled_labels:
            cls_caps = captions[label]
            sel_cls_cap = self.prompt_format.format(_pick_cap(cls_caps))
            texts.append(sel_cls_cap)

        # ==== 6) Padding (与 RandomLoadText 一致) ====
        num_sampled = len(texts)
        caption_padding_mask = torch.ones(num_sampled, dtype=torch.bool)

        if self.padding_to_max:
            num_valid_labels = len(positive_labels) + len(negative_labels)
            num_padding = self.max_num_samples - num_valid_labels
            if num_padding > 0:
                texts += [self.padding_value] * num_padding
                caption_padding_mask = torch.cat([caption_padding_mask,
                                                 torch.zeros(num_padding, dtype=torch.bool)])

        final_num_captions = len(texts)

        # ==== 7) 通过 labels 构建 caption_indices_list 和 caption_indices_onehot ====
        # 每个 box 对应一个 label，因此 caption_indices_list 中每个元素只包含一个索引
        caption_indices_list = [[int(label.item())] for label in target[gt_label_tag]]

        # 构建 one-hot 矩阵
        if num_boxes > 0 and final_num_captions > 0:
            caption_indices_onehot = torch.zeros((num_boxes, final_num_captions), dtype=torch.float32)
            for box_idx, cap_indices in enumerate(caption_indices_list):
                for cap_idx in cap_indices:
                    if 0 <= cap_idx < final_num_captions:
                        caption_indices_onehot[box_idx, cap_idx] = 1.0
        elif num_boxes == 0 and final_num_captions > 0:
            caption_indices_onehot = torch.zeros((0, final_num_captions), dtype=torch.float32)
            caption_indices_list = []
        else:
            raise ValueError("final_num_captions == 0")

        assert len(caption_indices_list) == num_boxes, \
            f"caption_indices_list length {len(caption_indices_list)} != num_boxes {num_boxes}"

        # ==== 8) 更新 target ====
        target.update({
            'captions': texts,
            'num_captions': final_num_captions,
            'num_regions': num_boxes,
            'caption_indices_list': caption_indices_list,
            'caption_indices_onehot': caption_indices_onehot,
            'caption_padding_mask': caption_padding_mask,
        })

        # 9) 构造属性字段：与 RandomLoadCaptions 保持一致的逻辑
        if self.attributes_align_use:
            target = self.build_attributes_tensors(target)

        return image, target, dataset
    

@register()
class LoadCaptionAttr_Grounding:
    """
    推理/验证时的属性处理类 (支持 Batch 处理，属性与Caption分离存储版本)
    
    核心功能：
    1. Caption处理: 将captions Pad到 max_num_captions。
    2. Attribute处理: 提取有效caption的属性，展平，并Pad到 max_num_attributes。
    3. 映射构建: 构建 [max_num_captions, max_attrs_per_caption] 的索引映射 Tensor，
       方便模型通过 gather 操作直接获取每个 Caption 对应的属性得分。
    """
    def __init__(self,
                 max_num_captions: int = 80,      # Caption列表的最大长度
                 max_attrs_per_caption: int = 10, # 【关键】每个Caption最多关联多少个属性 (用于构建映射Tensor)
                 padding_to_max: bool = True,     # 推理时通常必须为True以支持Batch
                 padding_value: str = '',
                 attributes_align_use: bool = False) -> None:
        
        self.max_num_captions = max_num_captions
        self.max_attrs_per_caption = max_attrs_per_caption
        
        # 全局属性列表的最大长度，通常为 caption数量 * 单个caption最大属性数
        self.max_num_attributes = max_num_captions * max_attrs_per_caption
        
        self.padding_to_max = padding_to_max
        self.padding_value = padding_value
        self.attributes_align_use = attributes_align_use

    def build_attributes_tensors(self, target: Dict) -> Dict:
        """
        构建属性相关的Tensor, Mask, One-Hot 和 Map
        """
        captions = target['captions']  # 此时包含 padding 后的 captions
        num_valid_captions = target['num_captions'] # 有效caption数量
        max_attr_global = self.max_num_attributes  # 全局属性上限

        caption_attributes_dict = target.get('caption_attributes_dict', False)
        caption_indices_list = target.get('caption_indices_list', [])
        
        if not caption_attributes_dict and len(captions)==1:
            caption_attributes_dict = {captions[0]: target['caption_attributes']}
        
        # 检查是否有 Boxes 信息 (验证模式 vs 纯推理模式)
        has_boxes = 'boxes' in target and len(target['boxes']) > 0

        # ===== Step 1: 提取属性、展平、构建映射 Tensor =====
        caption_attributes_list = []
        
        # [Map Tensor]: 初始化为 -1 (Padding)
        # Shape: [max_num_captions, max_attrs_per_caption]
        cap_to_attr_map = torch.full((self.max_num_captions, self.max_attrs_per_caption), 
                                     -1, dtype=torch.long)
        
        valid_captions_list = captions[:num_valid_captions]
        
        attr_to_idx = {}
        caption_to_attr_range = {} 
        attributes_global = []

        for cap_idx, caption in enumerate(valid_captions_list):
            if caption not in caption_attributes_dict:
                continue
            attrs = caption_attributes_dict[caption]

            attr_indices = []
            for a in attrs:
                if a in attr_to_idx:
                    attr_indices.append(attr_to_idx[a])
                elif len(attr_to_idx) < max_attr_global:
                    attr_to_idx[a] = len(attr_to_idx)
                    attributes_global.append(a)
                    attr_indices.append(attr_to_idx[a])
                # 超出上限则跳过

            caption_to_attr_range[caption] = attr_indices

            valid_indices = [idx for idx in attr_indices if idx < max_attr_global]
            valid_len = min(len(valid_indices), self.max_attrs_per_caption)
            if valid_len > 0:
                cap_to_attr_map[cap_idx, :valid_len] = torch.as_tensor(valid_indices[:valid_len], dtype=torch.long)

        # ===== Step 2: Attribute Padding (展平列表) =====
        caption_attributes_list = attributes_global
        valid_attr_len = len(caption_attributes_list)
        
        if self.padding_to_max:
            if valid_attr_len < self.max_num_attributes:
                caption_attributes_list.extend([self.padding_value] * (self.max_num_attributes - valid_attr_len))
            else:
                # 全局截断
                caption_attributes_list = caption_attributes_list[:self.max_num_attributes]
                valid_attr_len = self.max_num_attributes

        # ===== Step 3: 构建 Mask =====
        # [max_num_attributes]
        attr_padding_mask = torch.cat([
            torch.ones(valid_attr_len, dtype=torch.bool),
            torch.zeros(self.max_num_attributes - valid_attr_len, dtype=torch.bool)
        ])

        # ===== Step 4: 构建 Attribute One-Hot (仅当有 Box 时) =====
        # [num_boxes, max_num_attributes]
        if has_boxes and len(caption_indices_list) > 0:
            num_boxes = len(target['boxes'])
            caption_attributes_onehot = torch.zeros((num_boxes, self.max_num_attributes), dtype=torch.float32)
            
            for box_idx, cap_indices in enumerate(caption_indices_list):
                for cap_idx in cap_indices:
                    if cap_idx < num_valid_captions:
                        # 利用 map tensor 快速构建 (或者使用 range 字典)
                        # 这里演示使用 map tensor 的方式，保持一致性
                        attr_indices = cap_to_attr_map[cap_idx]
                        
                        # 过滤无效索引 (-1) 和 被全局截断的索引 (>= max_num_attributes)
                        valid_indices = attr_indices[
                            (attr_indices != -1) & (attr_indices < self.max_num_attributes)
                        ]
                        
                        if len(valid_indices) > 0:
                            caption_attributes_onehot[box_idx, valid_indices] = 1.0
        else:
            caption_attributes_onehot = torch.zeros((0, self.max_num_attributes), dtype=torch.float32)

        # ===== 更新 Target =====
        target['caption_attributes'] = caption_attributes_list
        target['caption_attributes_padding_mask'] = attr_padding_mask
        target['caption_attributes_onehot'] = caption_attributes_onehot
        
        # [max_num_captions, max_attrs_per_caption]
        target['cap_to_attr_map'] = cap_to_attr_map
        
        return target

    def __call__(self, *inputs: Any) -> Any:
        if len(inputs) == 1:
            inputs = inputs[0]
        image, target, dataset = inputs

        # 获取原始 Captions
        captions = target.get('captions', [])
        if isinstance(captions, str): 
            captions = [captions]
        
        original_num_captions = len(captions)
        
        # ===== Step 1: Caption Padding =====
        # 必须先处理 Caption 的 Padding，保证 batch 维度一致
        if self.padding_to_max:
            if original_num_captions > self.max_num_captions:
                # 验证/推理时，如果超过最大长度，通常报错或截断。这里选择报错以警示配置问题。
                raise ValueError(f"Sample captions num ({original_num_captions}) > max_num_caption ({self.max_num_captions}), please check config!")
            
            num_pad = self.max_num_captions - original_num_captions
            if num_pad > 0:
                captions = captions + [self.padding_value] * num_pad
        
        # 构建 Caption Mask
        caption_padding_mask = torch.zeros(len(captions), dtype=torch.bool)
        caption_padding_mask[:original_num_captions] = True
        
        # ===== Step 2: 处理 Caption One-Hot (如果有) =====
        if 'caption_indices_onehot' in target and target['caption_indices_onehot'] is not None:
            original_onehot = target['caption_indices_onehot'] # [num_boxes, orig_num_caps]
            num_boxes = original_onehot.shape[0]
            
            new_onehot = torch.zeros((num_boxes, self.max_num_captions), dtype=original_onehot.dtype)
            copy_len = min(original_onehot.shape[1], self.max_num_captions)
            new_onehot[:, :copy_len] = original_onehot[:, :copy_len]
            
            target['caption_indices_onehot'] = new_onehot
        
        target['captions'] = captions
        target['num_captions'] = original_num_captions
        target['caption_padding_mask'] = caption_padding_mask
        target['caption_section_len'] = len(captions)
        
        # ===== Step 3: Attribute 处理 =====
        if self.attributes_align_use:
            target = self.build_attributes_tensors(target)
            
        return image, target, dataset
