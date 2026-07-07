"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
"""

import torch
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F
import random
from PIL import Image

from .._misc import convert_to_tv_tensor
from ...core import register


@register()
class Mosaic(T.Transform):
    """
    Applies Mosaic augmentation to a batch of images. Combines four randomly selected images
    into a single composite image with randomized transformations.
    """

    def __init__(self, output_size=320, max_size=None, rotation_range=0, translation_range=(0.1, 0.1),
                 scaling_range=(0.5, 1.5), probability=1.0, fill_value=114, use_cache=True, max_cached_images=50,
                 random_pop=True) -> None:
        """
        Args:
            output_size (int): Target size for resizing individual images.
            rotation_range (float): Range of rotation in degrees for affine transformation.
            translation_range (tuple): Range of translation for affine transformation.
            scaling_range (tuple): Range of scaling factors for affine transformation.
            probability (float): Probability of applying the Mosaic augmentation.
            fill_value (int): Fill value for padding or affine transformations.
            use_cache (bool): Whether to use cache. Defaults to True.
            max_cached_images (int): The maximum length of the cache.
            random_pop (bool): Whether to randomly pop a result from the cache.
        """
        super().__init__()
        self.resize = T.Resize(size=output_size, max_size=max_size)
        self.probability = probability
        self.affine_transform = T.RandomAffine(degrees=rotation_range, translate=translation_range,
                                               scale=scaling_range, fill=fill_value)
        self.use_cache = use_cache
        self.mosaic_cache = []
        self.max_cached_images = max_cached_images
        self.random_pop = random_pop

    def load_samples_from_dataset(self, image, target, dataset):
        """Loads and resizes a set of images and their corresponding targets."""
        # Append the main image
        get_size_func = F.get_size if hasattr(F, "get_size") else F.get_spatial_size  # torchvision >=0.17 is get_size
        image, target = self.resize(image, target)
        resized_images, resized_targets = [image], [target]
        max_height, max_width = get_size_func(resized_images[0])

        # randomly select 3 images
        sample_indices = random.choices(range(len(dataset)), k=3)
        for idx in sample_indices:
            # image, target = dataset.load_item(idx)
            image, target = self.resize(dataset.load_item(idx))
            height, width = get_size_func(image)
            max_height, max_width = max(max_height, height), max(max_width, width)
            resized_images.append(image)
            resized_targets.append(target)

        return resized_images, resized_targets, max_height, max_width

    def load_samples_from_cache(self, image, target, cache):
        image, target = self.resize(image, target)
        cache.append(dict(img=image, labels=target))

        if len(cache) > self.max_cached_images:
            if self.random_pop:
                index = random.randint(0, len(cache) - 2)  # do not remove last image
            else:
                index = 0
            cache.pop(index)
        sample_indices = random.choices(range(len(cache)), k=3)
        mosaic_samples = [dict(img=cache[idx]["img"].copy(), labels=self._clone(cache[idx]["labels"])) for idx in
                          sample_indices]  # sample 3 images
        mosaic_samples = [dict(img=image.copy(), labels=self._clone(target))] + mosaic_samples

        get_size_func = F.get_size if hasattr(F, "get_size") else F.get_spatial_size
        sizes = [get_size_func(mosaic_samples[idx]["img"]) for idx in range(4)]
        max_height = max(size[0] for size in sizes)
        max_width = max(size[1] for size in sizes)

        return mosaic_samples, max_height, max_width

    def _visualize_mosaic(self, image, target, captions, save_path='mosaic_debug', max_vis=10):
        """
        可视化 Mosaic 拼接结果

        Args:
            image: PIL Image
            target: 包含 boxes, labels 等的字典
            captions: 类别文本列表
            save_path: 保存路径
            max_vis: 最多可视化的样本数
        """
        import os
        import numpy as np
        from PIL import ImageDraw, ImageFont

        # 创建保存目录
        os.makedirs(save_path, exist_ok=True)

        # 控制可视化数量
        if not hasattr(self, '_vis_count'):
            self._vis_count = 0

        if self._vis_count >= max_vis:
            return

        self._vis_count += 1

        # 复制图像用于绘制
        vis_image = image.copy()
        draw = ImageDraw.Draw(vis_image)

        # 尝试加载字体
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except:
            font = ImageFont.load_default()

        # 获取 boxes 和 labels
        boxes = target['boxes']
        labels = target['labels']

        # 定义颜色
        colors = [
            'red', 'blue', 'green', 'yellow', 'purple',
            'orange', 'pink', 'cyan', 'magenta', 'lime'
        ]

        # 绘制每个 box
        for idx, (box, label_idx) in enumerate(zip(boxes, labels)):
            x1, y1, x2, y2 = box.tolist()
            color = colors[int(label_idx) % len(colors)]

            # 绘制边界框
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

            # 获取类别文本
            if captions is not None:
                label_text = '/'.join(captions[int(label_idx)])
            else:
                label_text = f"class_{int(label_idx)}"

            # 绘制类别标签
            text_bbox = draw.textbbox((x1, y1), label_text, font=font)
            text_bg = [x1, y1 - (text_bbox[3] - text_bbox[1]) - 4, text_bbox[2] + 4, y1]
            draw.rectangle(text_bg, fill=color)
            draw.text((x1 + 2, y1 - (text_bbox[3] - text_bbox[1]) - 2), label_text, fill='white', font=font)

        # 绘制分隔线（显示4个图像的边界）
        width, height = image.size
        mid_w, mid_h = width // 2, height // 2
        draw.line([(mid_w, 0), (mid_w, height)], fill='white', width=2)
        draw.line([(0, mid_h), (width, mid_h)], fill='white', width=2)

        # 保存图像
        save_file = os.path.join(save_path, f'mosaic_{self._vis_count:04d}.jpg')
        vis_image.save(save_file)

        # 打印信息
        print(f"[Mosaic Debug] Saved visualization to {save_file}")
        print(f"  - Image size: {image.size}")
        print(f"  - Num boxes: {len(boxes)}")
        print(f"  - Num classes: {len(captions) if captions else 'N/A'}")
        if captions:
            print(f"  - Classes: {['/'.join(ct) for ct in captions]}")


    def create_mosaic_from_cache(self, mosaic_samples, max_height, max_width):
        placement_offsets = [[0, 0], [max_width, 0], [0, max_height], [max_width, max_height]]
        merged_image = Image.new(mode=mosaic_samples[0]["img"].mode, size=(max_width * 2, max_height * 2), color=0)
        offsets = torch.tensor([[0, 0], [max_width, 0], [0, max_height], [max_width, max_height]]).repeat(1, 2)

        #### wgt_add aim to solve the captions meger in mosaic augment operate
        all_captions = []
        if 'captions' in mosaic_samples[0]['labels']:
            for _, sample in enumerate(mosaic_samples):
                all_captions.extend(sample['labels']['captions'])

            unique_tuples = list(set(tuple(cls) for cls in all_captions))
            unified_captions = sorted([list(t) for t in unique_tuples])
        ####

        mosaic_target = []
        for i, sample in enumerate(mosaic_samples):
            img = sample["img"]
            target = sample["labels"]

            merged_image.paste(img, placement_offsets[i])
            target['boxes'] = target['boxes'] + offsets[i]

            # 重新映射 labels
            if 'captions' in target and 'labels' in target:
                old_captions = target['captions']
                old_labels = target['labels']

                new_labels = torch.tensor([
                    unified_captions.index(old_captions[int(label_idx)])
                    for label_idx in old_labels
                ], dtype=old_labels.dtype)

                target['labels'] = new_labels

            mosaic_target.append(target)

        merged_target = {}
        for key in mosaic_target[0]:
            # merged_target[key] = torch.cat([target[key] for target in mosaic_target])
            if key == 'captions':
                merged_target[key] = unified_captions
            elif key == 'caption_attributes_dict':
                merged_attrs = {}
                for target in mosaic_target:
                    attr_dict = target.get('caption_attributes_dict', {})
                    if not attr_dict:
                        continue
                    for cap, attrs in attr_dict.items():
                        attrs_list = list(attrs) if isinstance(attrs, list) else [attrs]
                        if cap not in merged_attrs:
                            merged_attrs[cap] = list(attrs_list)
                        else:
                            for a in attrs_list:
                                if a not in merged_attrs[cap]:
                                    merged_attrs[cap].append(a)
                merged_target[key] = merged_attrs
            elif key == 'filename':
                merged_target[key] = [target[key] for target in mosaic_target]
            elif key == 'OTA-Det':
                merged_target[key] = mosaic_target[0][key]
            else:
                merged_target[key] = torch.cat([target[key] for target in mosaic_target])

        # self._visualize_mosaic(merged_image, merged_target, unified_captions)

        return merged_image, merged_target

    def create_mosaic_from_dataset(self, images, targets, max_height, max_width):
        """Creates a mosaic image by combining multiple images."""
        placement_offsets = [[0, 0], [max_width, 0], [0, max_height], [max_width, max_height]]
        merged_image = Image.new(mode=images[0].mode, size=(max_width * 2, max_height * 2), color=0)
        for i, img in enumerate(images):
            merged_image.paste(img, placement_offsets[i])

        """Merges targets into a single target dictionary for the mosaic."""
        offsets = torch.tensor([[0, 0], [max_width, 0], [0, max_height], [max_width, max_height]]).repeat(1, 2)
        merged_target = {}
        for key in targets[0]:
            if key == 'boxes':
                values = [target[key] + offsets[i] for i, target in enumerate(targets)]
            else:
                values = [target[key] for target in targets]

            merged_target[key] = torch.cat(values, dim=0) if isinstance(values[0], torch.Tensor) else values

        return merged_image, merged_target

    # @staticmethod
    # def _clone(tensor_dict):
    #     return {key: value.clone() for (key, value) in tensor_dict.items()}
    @staticmethod
    def _clone(tensor_dict):
        def clone_value(val):
            if isinstance(val, torch.Tensor):
                return val.clone()
            if isinstance(val, dict):
                return {k: clone_value(v) for k, v in val.items()}
            if isinstance(val, list):
                return [clone_value(v) for v in val]
            if isinstance(val, tuple):
                return tuple(clone_value(v) for v in val)
            return val

        return {key: clone_value(value) for key, value in tensor_dict.items()}

    def forward(self, *inputs):
        """
        Args:
            inputs (tuple): Input tuple containing (image, target, dataset).

        Returns:
            tuple: Augmented (image, target, dataset).
        """
        if len(inputs) == 1:
            inputs = inputs[0]
        image, target, dataset = inputs

        # Skip mosaic augmentation with probability 1 - self.probability
        if self.probability < 1.0 and random.random() > self.probability:
            return image, target, dataset

        # Prepare mosaic components
        if self.use_cache:
            mosaic_samples, max_height, max_width = self.load_samples_from_cache(image, target, self.mosaic_cache)
            mosaic_image, mosaic_target = self.create_mosaic_from_cache(mosaic_samples, max_height, max_width)
        else:
            resized_images, resized_targets, max_height, max_width = self.load_samples_from_dataset(image, target,dataset)
            mosaic_image, mosaic_target = self.create_mosaic_from_dataset(resized_images, resized_targets, max_height, max_width)

        # Clamp boxes and convert target formats
        if 'boxes' in mosaic_target:
            mosaic_target['boxes'] = convert_to_tv_tensor(mosaic_target['boxes'], 'boxes', box_format='xyxy',
                                                          spatial_size=mosaic_image.size[::-1])
        if 'masks' in mosaic_target:
            mosaic_target['masks'] = convert_to_tv_tensor(mosaic_target['masks'], 'masks')

        # Apply affine transformations
        mosaic_image, mosaic_target = self.affine_transform(mosaic_image, mosaic_target)

        return mosaic_image, mosaic_target, dataset
