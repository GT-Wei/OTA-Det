"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE)
Copyright (c) 2024 D-FINE authors. All Rights Reserved.
"""

import torch
import torch.utils.data as data
from torch.utils.data import default_collate

import torchvision
import torchvision.transforms.v2 as VT
from torchvision.transforms.v2 import functional as VF, InterpolationMode

import random
from functools import partial

from ..core import register
torchvision.disable_beta_transforms_warning()
from copy import deepcopy
from PIL import Image, ImageDraw
import os
from collections import defaultdict, deque


__all__ = [
    'DataLoader',
    'BaseCollateFunction',
    'BatchImageCollateFunction',
    'batch_image_collate_fn'
]


@register()
class DataLoader(data.DataLoader):
    __inject__ = ['dataset', 'collate_fn']

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for n in ['dataset', 'batch_size', 'num_workers', 'drop_last', 'collate_fn']:
            format_string += "\n"
            format_string += "    {0}: {1}".format(n, getattr(self, n))
        format_string += "\n)"
        return format_string

    def set_epoch(self, epoch):
        self._epoch = epoch
        self.dataset.set_epoch(epoch)
        self.collate_fn.set_epoch(epoch)

    @property
    def epoch(self):
        return self._epoch if hasattr(self, '_epoch') else -1

    @property
    def shuffle(self):
        return self._shuffle

    @shuffle.setter
    def shuffle(self, shuffle):
        assert isinstance(shuffle, bool), 'shuffle must be a boolean'
        self._shuffle = shuffle


@register()
def batch_image_collate_fn(items):
    """only batch image
    """
    return torch.cat([x[0][None] for x in items], dim=0), [x[1] for x in items]


class BaseCollateFunction(object):
    def set_epoch(self, epoch):
        self._epoch = epoch

    @property
    def epoch(self):
        return self._epoch if hasattr(self, '_epoch') else -1

    def __call__(self, items):
        raise NotImplementedError('')


class CaptionMergeHandler:
    """
    Normalize caption sets after mixup/copyblend so batch samples respect a fixed budget.
    Keeps captions referenced by boxes first, truncates to max_num_captions, pads if needed,
    and remaps labels/caption_indices to the new caption space (optionally dropping boxes
    whose captions were trimmed out).

    Similar to RandomLoadCaptions in text_transforms.py, but works on merged captions after augmentation.
    """

    def __init__(self, max_num_captions=None, num_neg_samples=None, padding_to_max=True, padding_value=''):
        self.max_num_captions = max_num_captions
        self.num_neg_samples = num_neg_samples  # (min, max) tuple for negative sampling, e.g., (10, 20)
        self.padding_to_max = padding_to_max
        self.padding_value = padding_value

    # ----- shared helpers -----
    def caption_key(self, caption):
        return tuple(caption) if isinstance(caption, list) else caption

    def valid_caption_ids(self, captions, padding_mask):
        if padding_mask is None:
            return list(range(len(captions)))
        return [idx for idx, m in enumerate(padding_mask) if bool(m)]

    def merge_caption_sets(self, targets_list):
        """Merge captions across targets and return unified list/mask plus per-target remap dicts."""
        if any('captions' not in t for t in targets_list):
            return None

        unified_captions, key_to_new, mappings = [], {}, []
        for tgt in targets_list:
            captions = tgt.get('captions', [])
            padding_mask = tgt.get('caption_padding_mask', None)
            mapping = {}
            for idx in self.valid_caption_ids(captions, padding_mask):
                key = self.caption_key(captions[idx])
                if key not in key_to_new:
                    key_to_new[key] = len(unified_captions)
                    unified_captions.append(captions[idx])
                mapping[idx] = key_to_new[key]
            mappings.append(mapping)

        caption_padding_mask = torch.ones(len(unified_captions), dtype=torch.bool)
        return unified_captions, caption_padding_mask, mappings

    def remap_tensor_labels(self, labels, mapping_dict):
        mapped = labels.clone()
        for old_idx, new_idx in mapping_dict.items():
            mapped[labels == old_idx] = new_idx
        return mapped

    def remap_caption_indices_list(self, caption_indices_list, mapping_dict):
        return [[mapping_dict[idx] for idx in cap_indices if idx in mapping_dict] for cap_indices in caption_indices_list]

    def build_onehot_from_cil(self, caption_indices_list, num_captions, device, dtype=torch.float32):
        onehot = torch.zeros((len(caption_indices_list), num_captions), dtype=dtype, device=device)
        for box_idx, cap_indices in enumerate(caption_indices_list):
            for cap_idx in cap_indices:
                if 0 <= cap_idx < num_captions:
                    onehot[box_idx, cap_idx] = 1.0
        return onehot

    def __call__(self, target):
        """
        Caption merge and sampling after mixup/copyblend.
        Only triggers resampling if caption count exceeds max_num_captions.
        Logic fully based on RandomLoadCaptions_Det from text_transforms.py.
        """
        if self.max_num_captions is None or 'captions' not in target:
            return target

        captions = target.get('captions', [])
        num_captions = len(captions)

        if 'labels' not in target:
            return target

        if num_captions == 0:
            return target

        # Check if resampling is needed
        need_resample = num_captions > self.max_num_captions

        if not need_resample:
            # No resampling needed, just padding if required
            if self.padding_to_max and num_captions < self.max_num_captions:
                num_pad = self.max_num_captions - num_captions
                captions = captions + [self.padding_value] * num_pad

                # Update caption_padding_mask
                if 'caption_padding_mask' in target:
                    old_mask = target['caption_padding_mask']
                    target['caption_padding_mask'] = torch.cat([
                        old_mask[:num_captions] if len(old_mask) > num_captions else old_mask,
                        torch.zeros(num_pad, dtype=torch.bool)
                    ])
                else:
                    target['caption_padding_mask'] = torch.cat([
                        torch.ones(num_captions, dtype=torch.bool),
                        torch.zeros(num_pad, dtype=torch.bool)
                    ])

                # Update caption_indices_onehot if exists
                if 'caption_indices_onehot' in target:
                    old_onehot = target['caption_indices_onehot']
                    num_boxes = old_onehot.shape[0]
                    new_onehot = torch.zeros((num_boxes, self.max_num_captions), dtype=old_onehot.dtype, device=old_onehot.device)
                    new_onehot[:, :num_captions] = old_onehot[:, :num_captions]
                    target['caption_indices_onehot'] = new_onehot

                target['captions'] = captions
                target['num_captions'] = len(captions)
            return target

        # ==== Resampling logic - exactly same as RandomLoadCaptions_Det ====

        # 1) 收集正样本 caption 索引 (基于 labels)
        gt_label_tag = 'labels'
        positive_labels = set(target[gt_label_tag].tolist())

        # 如果正样本数超过 max_num_captions，随机采样
        if len(positive_labels) > self.max_num_captions:
            positive_labels = set(random.sample(list(positive_labels), k=self.max_num_captions))

        # 2) 负采样
        if self.num_neg_samples is not None:
            num_neg_samples = min(
                min(num_captions, self.max_num_captions) - len(positive_labels),
                random.randint(*self.num_neg_samples))
        else:
            # Default: fill up to max_num_captions
            num_neg_samples = min(num_captions, self.max_num_captions) - len(positive_labels)

        candidate_neg_labels = [idx for idx in range(num_captions) if idx not in positive_labels]
        num_neg_samples = min(num_neg_samples, len(candidate_neg_labels))
        negative_labels = random.sample(candidate_neg_labels, k=num_neg_samples) if num_neg_samples > 0 else []

        # 3) 合并并打乱
        sampled_labels = list(positive_labels) + list(negative_labels)
        random.shuffle(sampled_labels)

        label2ids = {label: i for i, label in enumerate(sampled_labels)}

        # 4) 过滤 boxes 和更新 labels
        gt_valid_mask = torch.zeros(len(target['boxes']), dtype=torch.bool, device=target['boxes'].device)
        for idx, label in enumerate(target[gt_label_tag].tolist()):
            if label in label2ids:
                gt_valid_mask[idx] = True
                target[gt_label_tag][idx] = label2ids[label]

        target['boxes'] = target['boxes'][gt_valid_mask]
        target[gt_label_tag] = target[gt_label_tag][gt_valid_mask]

        # Also filter area and mixup if exist
        if 'area' in target:
            target['area'] = target['area'][gt_valid_mask]
        if 'mixup' in target:
            target['mixup'] = target['mixup'][gt_valid_mask]

        num_boxes = len(target['boxes'])

        # 5) 重建 captions
        texts = [captions[label] for label in sampled_labels]

        # 6) Padding
        num_sampled = len(texts)
        caption_padding_mask = torch.ones(num_sampled, dtype=torch.bool)

        if self.padding_to_max:
            num_valid_labels = len(positive_labels) + len(negative_labels)
            num_padding = self.max_num_captions - num_valid_labels
            if num_padding > 0:
                texts += [self.padding_value] * num_padding
                caption_padding_mask = torch.cat([
                    caption_padding_mask,
                    torch.zeros(num_padding, dtype=torch.bool)
                ])

        final_num_captions = len(texts)

        # 7) 通过 labels 构建 caption_indices_list 和 caption_indices_onehot
        # 每个 box 对应一个 label，因此 caption_indices_list 中每个元素只包含一个索引
        caption_indices_list = [[int(label.item())] for label in target[gt_label_tag]]

        # 构建 one-hot 矩阵
        if num_boxes > 0 and final_num_captions > 0:
            caption_indices_onehot = torch.zeros((num_boxes, final_num_captions), dtype=torch.float32, device=target['boxes'].device)
            for box_idx, cap_indices in enumerate(caption_indices_list):
                for cap_idx in cap_indices:
                    if 0 <= cap_idx < final_num_captions:
                        caption_indices_onehot[box_idx, cap_idx] = 1.0
        elif num_boxes == 0 and final_num_captions > 0:
            caption_indices_onehot = torch.zeros((0, final_num_captions), dtype=torch.float32, device=target['boxes'].device)
            caption_indices_list = []
        else:
            raise ValueError("final_num_captions == 0")

        # 8) 更新 target
        target.update({
            'captions': texts,
            'num_captions': final_num_captions,
            'num_regions': num_boxes,
            'caption_indices_list': caption_indices_list,
            'caption_indices_onehot': caption_indices_onehot,
            'caption_padding_mask': caption_padding_mask,
        })

        return target


def generate_scales(base_size, base_size_repeat):
    scale_repeat = (base_size - int(base_size * 0.75 / 32) * 32) // 32
    scales = [int(base_size * 0.75 / 32) * 32 + i * 32 for i in range(scale_repeat)]
    scales += [base_size] * base_size_repeat
    scales += [int(base_size * 1.25 / 32) * 32 - i * 32 for i in range(scale_repeat)]
    return scales


@register() 
class BatchImageCollateFunction(BaseCollateFunction):
    def __init__(
        self, 
        stop_epoch=None, 
        ema_restart_decay=0.9999,
        base_size=640,
        base_size_repeat=None,
        mixup_prob=0.0,
        mixup_epochs=[0, 0],
        copyblend_prob=0.0,
        copyblend_epochs=[0, 0],
        copyblend_type='blend',
        conflict_with_mixup=False,
        area_threshold=100,
        num_objects=3,
        with_expand=False,
        expand_ratios=[0.1, 0.25],
        random_num_objects=False,
        data_vis=False,
        vis_save='./vis_dataset/',
        caption_merge_cfg=None,
    ) -> None:
        super().__init__()
        self.base_size = base_size
        self.scales = generate_scales(base_size, base_size_repeat) if base_size_repeat is not None else None
        self.stop_epoch = stop_epoch if stop_epoch is not None else 100000000
        self.ema_restart_decay = ema_restart_decay
        self.mixup_prob, self.mixup_epochs = mixup_prob, mixup_epochs

        self.copyblend_prob, self.copyblend_epochs, self.copyblend_type = copyblend_prob, copyblend_epochs, copyblend_type
        self.area_threshold, self.num_objects = area_threshold, num_objects
        self.data_vis, self.vis_save = data_vis, vis_save
        self.with_expand, self.expand_ratios, self.random_num_objects = with_expand, expand_ratios, random_num_objects
        self.conflict_with_mixup = conflict_with_mixup  # 是否冲突
        merge_cfg = caption_merge_cfg or {}
        self.debug_caption_vis = merge_cfg.get('debug_caption_vis', False)
        self.debug_caption_vis_max = merge_cfg.get('debug_caption_vis_max', 2)
        self.debug_caption_vis_save = merge_cfg.get('debug_caption_vis_save', './vis_caption_debug/')
        self.caption_merge_handler = CaptionMergeHandler(
            max_num_captions=merge_cfg.get('max_num_captions', merge_cfg.get('max_num_samples', None)),
            num_neg_samples=merge_cfg.get('num_neg_samples', None),  # e.g., [10, 20] for random(10, 20) negative samples
            padding_to_max=merge_cfg.get('padding_to_max', True),
            padding_value=merge_cfg.get('padding_value', '')
        )

        if self.mixup_prob > 0 or self.copyblend_prob > 0:
            if os.path.isdir(self.vis_save):
                for file in os.listdir(self.vis_save):
                    os.remove('{}/{}'.format(self.vis_save, file))
            os.makedirs(self.vis_save, exist_ok=True) if self.data_vis else None

            if self.mixup_prob > 0:
                print("     ### Using MixUp with Prob@{} in {} epochs ### ".format(mixup_prob, mixup_epochs))
            if self.copyblend_prob > 0:
                print("     ### Using CopyBlend-{} with Prob@{} in {} epochs ### ".format(copyblend_type, copyblend_prob, copyblend_epochs))
                print(f'     ### CopyBlend -- area threshold@{area_threshold} and num of object@{num_objects} ###     ')
                if self.with_expand:
                    print(f'     ### CopyBlend -- expand@{expand_ratios} ###     ')
                if self.random_num_objects:
                    print(f'     ### CopyBlend -- random num of objects@{[1, self.num_objects]} ###     ')

        if stop_epoch is not None:
            print("     ### Multi-scale Training until {} epochs ### ".format(self.stop_epoch))
            print("     ### Multi-scales@ {} ###        ".format(self.scales))
        self.print_info_flag = True
        self.print_copyblend_flag = True
        # self.interpolation = interpolation

    def _tensor_to_pil(self, image_tensor):
        """Convert tensor [C,H,W] to uint8 PIL Image with simple de-normalization handling."""
        img = image_tensor
        if img.min() < 0:  # assume normalized with ImageNet stats
            img = img * torch.tensor([0.229, 0.224, 0.225], device=img.device).view(3, 1, 1) \
                + torch.tensor([0.485, 0.456, 0.406], device=img.device).view(3, 1, 1)
        img = img.clamp(0, 1)
        img_uint8 = (img * 255).to(torch.uint8).cpu().permute(1, 2, 0).numpy()
        return Image.fromarray(img_uint8)

    def _draw_debug_image(self, pil_img, target, save_path, title=''):
        draw = ImageDraw.Draw(pil_img)
        w, h = pil_img.size
        boxes = target.get('boxes', [])
        labels = target.get('labels', [])
        caps = target.get('captions', [])
        cil = target.get('caption_indices_list', [])

        for idx, box in enumerate(boxes):
            cx, cy, bw, bh = box
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            draw.rectangle([x1, y1, x2, y2], outline=(255, 255, 0))
            label_txt = f"id:{int(labels[idx].item())}" if len(labels) > idx else ""
            cap_txt = ""
            if cil and idx < len(cil) and cil[idx]:
                ids_str = ",".join(str(c) for c in cil[idx])
                cap_idx = cil[idx][0]
                cap_str = caps[cap_idx] if isinstance(caps, list) and 0 <= cap_idx < len(caps) else ""
                cap_txt = f"{str(cap_str)[:32]}"
            text = f"{label_txt}{cap_txt}"
            if text:
                # draw a solid background to keep text readable
                bbox = draw.textbbox((x1, max(y1 - 12, 0)), text)
                draw.rectangle(bbox, fill=(0, 0, 0))
                draw.text((x1, max(y1 - 12, 0)), text, fill=(255, 255, 255))

        if title:
            title_bbox = draw.textbbox((5, 5), title)
            draw.rectangle(title_bbox, fill=(0, 0, 0))
            draw.text((5, 5), title, fill=(0, 255, 0))
        dirpath = os.path.dirname(save_path) or '.'
        os.makedirs(dirpath, exist_ok=True)
        pil_img.save(save_path)

    def _debug_caption_visualize(self, images, targets_before, targets_after, tag=''):
        if not self.debug_caption_vis:
            return
        max_vis = min(len(images), self.debug_caption_vis_max)
        for idx in range(max_vis):
            img_pil = self._tensor_to_pil(images[idx])
            base_name = f"{tag}_sample{idx}"
            before_path = os.path.join(self.debug_caption_vis_save, base_name + "_before.jpg")
            after_path = os.path.join(self.debug_caption_vis_save, base_name + "_after.jpg")
            self._draw_debug_image(img_pil.copy(), targets_before[idx], before_path, title='before')
            self._draw_debug_image(img_pil.copy(), targets_after[idx], after_path, title='after')

    def apply_mixup(self, images, targets):
        """
        Applies Mixup augmentation to the batch if conditions are met.

        Args:
            images (torch.Tensor): Batch of images.
            targets (list[dict]): List of target dictionaries corresponding to images.

        Returns:
            tuple: Updated images and targets
        """
        # Log when Mixup is permanently disabled
        if self.epoch == self.mixup_epochs[-1] and self.print_info_flag:
            print(f"     ### Attention --- Mixup is closed after epoch@ {self.epoch} ###")
            self.print_info_flag = False

        MixUp_flag, CopyBlend_flag = False, False
        beta = round(random.uniform(0.45, 0.55), 6)
        cmh = self.caption_merge_handler  # cmh: caption merge handler
        targets_before_merge = None
        stage_tag = 'none'
        # Apply Mixup if within specified epoch range and probability threshold
        if random.random() < self.mixup_prob and self.mixup_epochs[0] <= self.epoch < self.mixup_epochs[-1]:
            # Generate mixup ratio
            beta = round(random.uniform(0.45, 0.55), 6)
            MixUp_flag = True

            # Mix images
            images = images.roll(shifts=1, dims=0).mul_(1.0 - beta).add_(images.mul(beta))

            # Prepare targets for Mixup
            shifted_targets = targets[-1:] + targets[:-1]
            updated_targets = deepcopy(targets)

            for i in range(len(targets)):
                caption_merge = cmh.merge_caption_sets([targets[i], shifted_targets[i]]) if cmh else None
                if caption_merge is not None:
                    unified_captions, unified_mask, mappings = caption_merge
                    mapping_a, mapping_b = mappings
                    num_caps = len(unified_captions)

                    # Remap labels and caption indices for both samples
                    remapped_labels_a = cmh.remap_tensor_labels(targets[i]['labels'], mapping_a)
                    remapped_labels_b = cmh.remap_tensor_labels(shifted_targets[i]['labels'], mapping_b)

                    caption_indices_list_a = targets[i].get('caption_indices_list', [])
                    caption_indices_list_b = shifted_targets[i].get('caption_indices_list', [])
                    remapped_cil_a = cmh.remap_caption_indices_list(caption_indices_list_a, mapping_a) if caption_indices_list_a else []
                    remapped_cil_b = cmh.remap_caption_indices_list(caption_indices_list_b, mapping_b) if caption_indices_list_b else []

                    dtype_onehot = targets[i].get('caption_indices_onehot', None)
                    dtype_onehot = dtype_onehot.dtype if dtype_onehot is not None else torch.float32
                    onehot_a = cmh.build_onehot_from_cil(remapped_cil_a, num_caps, device=targets[i]['labels'].device, dtype=dtype_onehot)
                    onehot_b = cmh.build_onehot_from_cil(remapped_cil_b, num_caps, device=targets[i]['labels'].device, dtype=dtype_onehot)

                # Combine boxes, labels, and areas from original and shifted targets
                updated_targets[i]['boxes'] = torch.cat([targets[i]['boxes'], shifted_targets[i]['boxes']], dim=0)
                updated_targets[i]['labels'] = torch.cat([remapped_labels_a, remapped_labels_b], dim=0)
                updated_targets[i]['area'] = torch.cat([targets[i]['area'], shifted_targets[i]['area']], dim=0)

                # Combine caption-related fields if they exist
                if caption_merge is not None:
                    updated_targets[i]['captions'] = unified_captions
                    updated_targets[i]['caption_padding_mask'] = unified_mask
                    updated_targets[i]['num_captions'] = len(unified_captions)
                    updated_targets[i]['caption_indices_list'] = remapped_cil_a + remapped_cil_b
                    updated_targets[i]['caption_indices_onehot'] = torch.cat([onehot_a, onehot_b], dim=0)
                elif 'caption_indices_list' in targets[i]:
                    raise ValueError('Check, forget use caption_merge?')
                
                if caption_merge is None and 'caption_indices_onehot' in targets[i] and 'caption_indices_onehot' in shifted_targets[i]:
                    raise ValueError('Check, forget use caption_merge?')
                
                # Add mixup ratio to targets
                updated_targets[i]['mixup'] = torch.tensor(
                    [beta] * len(targets[i]['labels']) + [1.0 - beta] * len(shifted_targets[i]['labels']),
                    dtype=torch.float32
                    )
            targets = updated_targets
            targets_before_merge = deepcopy(targets)
            stage_tag = 'mixup'

        elif (self.copyblend_epochs[0] <= self.epoch < self.copyblend_epochs[-1] and random.random() < self.copyblend_prob):
            if self.epoch == self.copyblend_epochs[-1] and self.print_copyblend_flag:
                print(f"     ### Attention --- CopyBlend closed after epoch@ {self.epoch} ###")
                self.print_copyblend_flag = False

            CopyBlend_flag = True
            objects_pool = defaultdict(list)
            img_height, img_width = images[0].shape[-2:]

            # get all valid objects in batch
            for i in range(len(images)):
                source_boxes = targets[i]['boxes']
                source_labels = targets[i]['labels']
                source_areas = targets[i]['area']
                
                # filter valid objects
                valid_objects = [idx for idx in range(len(source_boxes)) if source_areas[idx] >= self.area_threshold]
                for idx in valid_objects:
                    objects_pool['boxes'].append(source_boxes[idx])
                    objects_pool['labels'].append(source_labels[idx])
                    objects_pool['areas'].append(source_areas[idx])
                    objects_pool['image_idx'].append(i)
                    objects_pool['image_height'].append(img_height)
                    objects_pool['image_width'].append(img_width)
                    objects_pool['obj_idx'].append(idx)
            
            # check if objects_pool is empty
            if len(objects_pool['boxes']) == 0:
                if cmh is not None:
                    targets_before_merge = deepcopy(targets)
                    targets_after = [cmh(t) for t in targets_before_merge]
                    self._debug_caption_visualize(images, targets_before_merge, targets_after, tag='copyblend_empty')
                    return images, targets_after
                return images, targets
            
            # convert list to tensor for convenient operation
            for key in ['boxes', 'labels', 'areas']:
                objects_pool[key] = torch.stack(objects_pool[key]) if objects_pool[key] else torch.tensor([])
                
            # apply CopyBlend
            batch_size = len(images)
            updated_images = images.clone()
            updated_targets = deepcopy(targets)

            for i in range(batch_size):
                base_target = updated_targets[i]
                # randomly decide the number of objects to blend
                if self.random_num_objects:
                    num_objects = random.randint(1, min(self.num_objects, len(objects_pool['boxes'])))
                else:
                    num_objects = min(self.num_objects, len(objects_pool['boxes']))
                
                # randomly select objects to blend
                selected_indices = random.sample(range(len(objects_pool['boxes'])), num_objects)
                
                blend_boxes = []
                blend_labels = []
                blend_areas = []
                blend_mixup_ratios = []
                blend_source_indices = []
                blend_obj_indices = []
                used_indices = []

                source_indices_set = set()

                for idx in selected_indices:
                    # get source object information
                    box = objects_pool['boxes'][idx]
                    label = objects_pool['labels'][idx]
                    area = objects_pool['areas'][idx]
                    source_idx = objects_pool['image_idx'][idx]
                    source_height = objects_pool['image_height'][idx]
                    source_width = objects_pool['image_width'][idx]
                    obj_idx = objects_pool['obj_idx'][idx]
                    
                    # calculate source object size and position
                    cx, cy, w, h = box
                    x1_src, y1_src = int((cx - w / 2) * source_width), int((cy - h / 2) * source_height)
                    x2_src, y2_src = int((cx + w / 2) * source_width), int((cy + h / 2) * source_height)

                    # check if source object is out of bound
                    x1_src, y1_src = max(x1_src, 0), max(y1_src, 0)
                    x2_src, y2_src = min(x2_src, img_width), min(y2_src, img_height)
                    new_w_px, new_h_px = x2_src - x1_src, y2_src - y1_src
                    # check if source object is valid
                    if new_w_px <= 0 or new_h_px <= 0:
                        continue

                    source_indices_set.add(source_idx)
                    blend_source_indices.append(source_idx)
                    blend_obj_indices.append(obj_idx)
                    used_indices.append(idx)

                    # randomly determine blend position
                    x1 = random.randint(0, img_width - new_w_px) if new_w_px < img_width else 0
                    y1 = random.randint(0, img_height - new_h_px) if new_h_px < img_height else 0
                    # after the above limit, [x2, y2] will not be out of bound, so no need to check
                    x2, y2 = x1 + new_w_px, y1 + new_h_px
                    
                    # calculate new normalized coordinates
                    new_cx, new_cy = (x1 + new_w_px / 2) / img_width, (y1 + new_h_px / 2) / img_height
                    new_w, new_h = new_w_px / img_width, new_h_px / img_height

                    # add to blend list - use original unexpanded box
                    blend_boxes.append(torch.tensor([new_cx, new_cy, new_w, new_h]))
                    blend_labels.append(label)
                    blend_areas.append(area)
                    # mixup ratio
                    blend_mixup_ratios.append(1.0 - beta)

                    # handle expanded area
                    if self.with_expand:
                        alpha = round(random.uniform(self.expand_ratios[0], self.expand_ratios[1]), 6)
                        expand_w, expand_h = int(new_w_px * alpha), int(new_h_px * alpha)
                        # check if out of bound: get the best offset in GT image
                        x1_expand, y1_expand = x1_src - max(x1_src - expand_w, 0), y1_src - max(y1_src - expand_h, 0)
                        x2_expand, y2_expand = min(x2_src + expand_w, img_width) - x2_src, min(y2_src + expand_h, img_height) - y2_src
                        # check if out of bound: whether the expanded area is out of bound in blend image
                        new_x1_expand, new_y1_expand = x1 - max(x1 - x1_expand, 0), y1 - max(y1 - y1_expand, 0)
                        new_x2_expand, new_y2_expand = min(x2 + x2_expand, img_width) - x2, min(y2 + y2_expand, img_height) - y2
                        # update
                        x1_src, y1_src, x2_src, y2_src = x1_src - new_x1_expand, y1_src - new_y1_expand, x2_src + new_x2_expand, y2_src + new_y2_expand
                        x1, y1, x2, y2 = x1 - new_x1_expand, y1 - new_y1_expand, x2 + new_x2_expand, y2 + new_y2_expand

                    # blend original area first
                    copy_patch_orig = images[source_idx, :, y1_src:y2_src, x1_src:x2_src]
                    if self.copyblend_type == 'blend':
                        blended_patch = updated_images[i, :, y1:y2, x1:x2] * beta + copy_patch_orig * (1 - beta)
                        updated_images[i, :, y1:y2, x1:x2] = blended_patch
                    else:
                        updated_images[i, :, y1:y2, x1:x2] = copy_patch_orig
                    
                # prepare caption merging info if captions exist
                caption_merge = None
                source_index_list = list(source_indices_set)
                if 'captions' in base_target and all('captions' in updated_targets[s] for s in source_index_list):
                    involved_targets = [base_target] + [updated_targets[s] for s in source_index_list]
                    caption_merge = cmh.merge_caption_sets(involved_targets) if cmh else None
                    if caption_merge is not None:
                        unified_captions, unified_mask, mappings = caption_merge
                        base_mapping = mappings[0]
                        src_mappings = {src: mappings[idx + 1] for idx, src in enumerate(source_index_list)}

                        # remap base target before adding blended objects
                        base_target['labels'] = cmh.remap_tensor_labels(base_target['labels'], base_mapping)
                        if 'caption_indices_list' in base_target:
                            base_target['caption_indices_list'] = cmh.remap_caption_indices_list(
                                base_target['caption_indices_list'], base_mapping)
                        base_target['captions'] = unified_captions
                        base_target['caption_padding_mask'] = unified_mask
                        base_target['num_captions'] = len(unified_captions)

                        # remap labels for blended objects into unified caption space
                        remapped_blend_labels = []
                        for blend_label, src_idx in zip(blend_labels, blend_source_indices):
                            mapping = src_mappings.get(src_idx, {})
                            new_label = mapping.get(int(blend_label.item()), int(blend_label.item()))
                            remapped_blend_labels.append(torch.tensor(new_label, dtype=blend_label.dtype))
                        blend_labels = remapped_blend_labels

                # add blended objects to targets
                if len(blend_boxes) > 0:
                    blend_boxes = torch.stack(blend_boxes)
                    blend_labels = torch.stack(blend_labels)
                    blend_areas = torch.stack(blend_areas)

                    # add mixup ratio
                    updated_targets[i]['mixup'] = torch.tensor(
                        [1.0] * len(updated_targets[i]['boxes']) + blend_mixup_ratios,
                        dtype=torch.float32
                    )
                    # update targets
                    updated_targets[i]['boxes'] = torch.cat([updated_targets[i]['boxes'], blend_boxes])
                    updated_targets[i]['labels'] = torch.cat([updated_targets[i]['labels'], blend_labels])
                    updated_targets[i]['area'] = torch.cat([updated_targets[i]['area'], blend_areas])

                    # Update caption_indices_list if exists
                    if 'caption_indices_list' in updated_targets[i]:
                        blend_caption_indices = []
                        for idx, blend_label in enumerate(blend_labels):
                            src_idx = objects_pool['image_idx'][used_indices[idx]]
                            obj_idx = objects_pool['obj_idx'][used_indices[idx]]
                            src_target = targets[src_idx]
                            src_cil = src_target.get('caption_indices_list', [])
                            if caption_merge is not None:
                                mapping = src_mappings.get(src_idx, {})
                                if src_cil and obj_idx < len(src_cil):
                                    mapped_caps = [mapping[c] for c in src_cil[obj_idx] if c in mapping]
                                else:
                                    mapped_caps = [mapping.get(int(blend_label.item()), int(blend_label.item()))]
                            else:
                                if src_cil and obj_idx < len(src_cil):
                                    mapped_caps = src_cil[obj_idx]
                                else:
                                    mapped_caps = [int(blend_label.item())]
                            blend_caption_indices.append(mapped_caps)
                        updated_targets[i]['caption_indices_list'] = updated_targets[i]['caption_indices_list'] + blend_caption_indices

                    # Update caption_indices_onehot if exists
                    if 'caption_indices_onehot' in updated_targets[i]:
                        if caption_merge is None:
                            num_captions = updated_targets[i]['caption_indices_onehot'].shape[1]
                            blend_onehot = torch.zeros((len(blend_labels), num_captions), dtype=torch.float32)
                            for box_idx, label in enumerate(blend_labels):
                                label_idx = int(label.item())
                                if 0 <= label_idx < num_captions:
                                    blend_onehot[box_idx, label_idx] = 1.0
                            updated_targets[i]['caption_indices_onehot'] = torch.cat([updated_targets[i]['caption_indices_onehot'], blend_onehot])

                    # Rebuild caption onehot for merged captions (handles base boxes too)
                    if caption_merge is not None and 'caption_indices_list' in updated_targets[i]:
                        num_captions = len(base_target['captions'])
                        dtype_onehot = updated_targets[i]['caption_indices_onehot'].dtype if 'caption_indices_onehot' in updated_targets[i] else torch.float32
                        updated_targets[i]['caption_indices_onehot'] = cmh.build_onehot_from_cil(
                            updated_targets[i]['caption_indices_list'],
                            num_captions,
                            device=updated_targets[i]['labels'].device,
                            dtype=dtype_onehot
                        )
                elif caption_merge is not None and 'caption_indices_list' in updated_targets[i]:
                    num_captions = len(base_target['captions'])
                    dtype_onehot = updated_targets[i]['caption_indices_onehot'].dtype if 'caption_indices_onehot' in updated_targets[i] else torch.float32
                    updated_targets[i]['caption_indices_onehot'] = cmh.build_onehot_from_cil(
                        updated_targets[i]['caption_indices_list'],
                        num_captions,
                        device=updated_targets[i]['labels'].device,
                        dtype=dtype_onehot
                    )

            images, targets = updated_images, updated_targets
            targets_before_merge = deepcopy(targets)
            stage_tag = 'copyblend'

        if self.data_vis and CopyBlend_flag:
            for i in range(len(updated_targets)):
                image_tensor = images[i]
                if image_tensor.dim() == 4:
                    image_tensor = image_tensor.squeeze(0)
                if image_tensor.min() < 0:  # use normalization
                    image_tensor = image_tensor * torch.tensor([0.229, 0.224, 0.225], device=image_tensor.device).view(3, 1, 1) \
                            + torch.tensor([0.485, 0.456, 0.406], device=image_tensor.device).view(3, 1, 1)
                image_tensor_uint8 = (image_tensor * 255).type(torch.uint8)
                image_numpy = image_tensor_uint8.numpy().transpose((1, 2, 0))
                pilImage = Image.fromarray(image_numpy)
                draw = ImageDraw.Draw(pilImage)
                print('mix_vis:', i, 'boxes.len=', len(updated_targets[i]['boxes']))
                for box in updated_targets[i]['boxes']:
                    draw.rectangle([int(box[0]*640 - (box[2]*640)/2), int(box[1]*640 - (box[3]*640)/2), 
                                    int(box[0]*640 + (box[2]*640)/2), int(box[1]*640 + (box[3]*640)/2)], outline=(255,255,0))
                pilImage.save(self.vis_save + str(i) + "_"+ str(len(updated_targets[i]['boxes'])) +'_out.jpg')
        if self.caption_merge_handler is not None:
            if targets_before_merge is None:
                targets_before_merge = deepcopy(targets)
            targets_after = [self.caption_merge_handler(t) for t in targets]
            # self._debug_caption_visualize(images, targets_before_merge, targets_after, tag=stage_tag)
            targets = targets_after
        return images, targets

    def __call__(self, items):
        images = torch.cat([x[0][None] for x in items], dim=0)
        targets = [x[1] for x in items]

        # Mixup
        images, targets = self.apply_mixup(images, targets)

        if self.scales is not None and self.epoch < self.stop_epoch:
            # sz = random.choice(self.scales)
            # sz = [sz] if isinstance(sz, int) else list(sz)
            # VF.resize(inpt, sz, interpolation=self.interpolation)

            sz = random.choice(self.scales)
            for tg in targets:
                tg['multi_scale_size'] = sz

        return images, targets
