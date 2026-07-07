"""
支持多caption多region的VG数据集
每个样本包含一张图像的所有captions和对应的regions
使用one-hot编码表示caption-region映射关系
"""
import os.path
from typing import Callable, Optional
import json
from PIL import Image
import torch
import random
import os, sys
from datasets import load_from_disk

from .._misc import convert_to_tv_tensor
from ...core import register
from ._dataset import DetDataset

__all__ = ['HF_VGDataset']


@register()
class HF_VGDataset(DetDataset): 
    __inject__ = ['transforms', ]
    
    def __init__(
        self,
        img_root: str,
        ann_file: str,  # path to Hugging Face dataset directory
        transforms,
        attributes_align_use = False
    ) -> None:
        self._transforms = transforms
        self.root = img_root
        self.dataset_mode = "VG"
        self.attributes_align_use = attributes_align_use
        self._load_dataset(ann_file)
        self.get_dataset_info()

    def _load_dataset(self, anno):
        """加载 Hugging Face Dataset"""
        if os.path.isdir(anno):
            print(f"Loading Hugging Face dataset from {anno}")
            self.dataset = load_from_disk(anno)
            print(f"Loaded {len(self.dataset)} samples")
        else:
            raise ValueError(f"{anno} is not a valid directory for Hugging Face dataset")

    def __len__(self) -> int:
        return len(self.dataset)

    def get_dataset_info(self):
        print(f"  == total samples: {len(self.dataset)}")

    def __getitem__(self, index):
        img, target = self.load_item(index)
        if self._transforms is not None:
            img, target, _ = self._transforms(img, target, self)
            
        return img, target

    def load_item(self, index: int):
        """
        实现DetDataset要求的load_item方法
        返回transforms之前的原始数据
        """
        item = self.dataset[index]
        
        rel_path = item["filename"]
        abs_path = os.path.join(self.root, rel_path)
        
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"{abs_path} not found.")
        
        image = Image.open(abs_path).convert('RGB')
        w, h = image.size

        # 获取所有captions
        captions = []
        caption_attributes = {}
        for cap_info in item["caption_list"]:
            captions.append(cap_info["caption"])
            if self.attributes_align_use:
                attr_tmp_list = []
                for attr in cap_info["attributes"]:
                    attr_tmp_list.append(attr['description'])
                caption_attributes[cap_info["caption"]] = attr_tmp_list
        
        # num_captions = len(captions)、
        if len(captions) == 0:
            print("captions==0:", abs_path)
        assert len(captions) > 0
        
        # 获取所有regions
        all_boxes = []
        all_caption_indices_list = []
        all_areas = []

        for region in item["regions"]:
            all_boxes.append(region["bbox"])
            all_caption_indices_list.append(region["caption_indices"])
            # 如果HF数据集中有area字段则使用，否则后续计算
            if "area" in region:
                all_areas.append(region["area"])

        # 转换boxes
        if all_boxes:
            boxes = convert_to_tv_tensor(all_boxes, key='boxes', spatial_size=image.size[::-1])
        else:
            raise ValueError('check, no boxes.')

        # 计算area字段（用于mixup等数据增强）
        if all_areas and len(all_areas) == len(boxes):
            # 如果HF数据集提供了area则使用
            area = torch.as_tensor(all_areas, dtype=torch.float32)
        elif len(boxes) > 0:
            # 否则根据boxes计算（boxes 格式为 xyxy，来自 convert_to_tv_tensor）
            area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        else:
            area = torch.empty((0,), dtype=torch.float32)

        # # one-hot 编码
        # caption_indices_onehot = torch.zeros((num_regions, num_captions), dtype=torch.float32)
        # for region_idx, caption_indices in enumerate(all_caption_indices_list):
        #     for cap_idx in caption_indices:
        #         if 0 <= cap_idx < num_captions:
        #             caption_indices_onehot[region_idx, cap_idx] = 1.0

        # 归一化
        # caption_indices_onehot = caption_indices_onehot / (caption_indices_onehot.sum(dim=-1, keepdim=True) + 1e-8)

        # 构建target
        target = {
            "size": torch.as_tensor([int(h), int(w)]),
            "image_id": torch.tensor([index]),
            "filename": rel_path,
            "captions": captions,
            "boxes": boxes,
            "caption_indices_list": all_caption_indices_list,
            "caption_attributes_dict": caption_attributes,
            "labels": None,
            "area": area,
            'iscrowd': torch.zeros(len(boxes), dtype=torch.int64),
            'orig_size': torch.as_tensor([int(w), int(h)]),
            'OTA-Det': True
        }
        # label不用普通的label idx，而是采用caption_indices
        assert len(all_caption_indices_list) == len(boxes)
        return image, target


if __name__ == "__main__":
    print("=== 测试多caption多region数据集 (with DetDataset) ===")
    dataset = HF_VGDataset(
        img_root="../datasets/LLM_Caption_Parse/images/AerialVG/",
        ann_file="../datasets/LLM_Caption_Parse/annotations_hf/AerialVG_train",
    )
    print(f"\nDataset length: {len(dataset)}")
    
    # 测试set_epoch（继承自DetDataset）
    print("\n=== 测试set_epoch方法 ===")
    print(f"Initial epoch: {dataset.epoch}")
    dataset.set_epoch(5)
    print(f"After set_epoch(5): {dataset.epoch}")
    
    # 测试几个样本
    print("\n" + "="*80)
    for i in range(3):
        idx = random.randint(0, min(100, len(dataset)-1))
        print(f"\n{'='*80}")
        print(f"Sample {i+1} (index={idx})")
        print(f"{'='*80}")
        
        image, target = dataset[idx]
        
        print(f"\n基本信息:")
        print(f"  文件名: {target['filename']}")
        print(f"  图像尺寸: {image.size}")
        print(f"  Target尺寸: {target['size']}")
        
        print(f"\nCaption信息 (共{target['num_captions']}个):")
        for cap_idx, caption in enumerate(target['captions']):
            print(f"  [{cap_idx}] {caption}")
        
        print(f"\nRegion信息 (共{target['num_regions']}个):")
        print(f"Boxes shape: {target['boxes'].shape}")
        print(f"Caption indices one-hot shape: {target['caption_indices_onehot'].shape}")
        
        for region_idx in range(min(3, target['num_regions'])):  # 只显示前3个
            bbox = target['boxes'][region_idx].tolist()
            onehot = target['caption_indices_onehot'][region_idx]
            caption_indices = target['caption_indices_list'][region_idx]
            
            print(f"\n  [Region {region_idx}]")
            print(f"    BBox: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")
            print(f"    One-hot: {onehot.tolist()}")
            print(f"    对应的caption索引: {caption_indices}")
            
            for cap_idx in caption_indices:
                print(f"      - Caption {cap_idx}: {target['captions'][cap_idx][:60]}...")
        
        print(f"\n验证one-hot编码:")
        print(f"  Shape: [num_regions={target['num_regions']}, num_captions={target['num_captions']}]")
        print(f"  每行的和: {target['caption_indices_onehot'].sum(dim=1).tolist()}")
        print(f"  每列的和: {target['caption_indices_onehot'].sum(dim=0).tolist()}")

# python -m engine.data.dataset.grounding_dataset