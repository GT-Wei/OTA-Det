"""
Detection-style dataset that outputs the same fields as `otadet_grounding_dataset`.
Supports:
1) Hugging Face serialized datasets (directory path)
2) COCO-format JSON detection datasets

For detection数据集，`captions` 将是数据集类别列表，box 的 `caption_indices_list`/`labels`
对应该类别在该列表中的索引。
"""
import os.path
from typing import Callable, Optional
import json
from PIL import Image
import torch
import random
import os, sys
from datasets import load_from_disk
from faster_coco_eval import COCO

from .._misc import convert_to_tv_tensor
from ...core import register
from ._dataset import DetDataset

__all__ = ['HF_DetDataset']


@register()
class HF_DetDataset(DetDataset): 
    __inject__ = ['transforms', ]
    
    def __init__(
        self,
        img_root: str,
        ann_file: str,  # path to HF dataset directory or COCO json
        transforms,
        category_names: list = None,
        attributes_align_use: bool = False,
        dataset_format: str = "auto"
    ) -> None:
        self._transforms = transforms
        self.root = img_root
        self.dataset_mode = "Det"
        self.attributes_align_use = attributes_align_use
        self.dataset_format = dataset_format if dataset_format != "auto" else ("hf" if os.path.isdir(ann_file) else "coco")

        if self.dataset_format == "hf":
            self._load_hf_dataset(ann_file, category_names)
        elif self.dataset_format == "coco":
            self._load_coco_dataset(ann_file, category_names)
        else:
            raise ValueError(f"Unsupported dataset_format: {self.dataset_format}")

        self.caption_attributes_dict = self._make_caption_attr_dict()
        self.get_dataset_info()

    def _make_caption_attr_dict(self):
        """为每个类别构建简单的属性映射（属性=类别本身）。"""
        attr_dict = {}
        for cap in self.category_names:
            if isinstance(cap, list):
                for c in cap:
                    attr_dict[c] = [c]
            else:
                attr_dict[cap] = [cap]
        return attr_dict

    def _load_hf_dataset(self, anno, category_names):
        """加载 Hugging Face Dataset"""
        if os.path.isdir(anno):
            print(f"Loading Hugging Face dataset from {anno}")
            self.dataset = load_from_disk(anno)
            print(f"Loaded {len(self.dataset)} samples")
            self.ids = list(range(len(self.dataset)))
        else:
            raise ValueError(f"{anno} is not a valid directory for Hugging Face dataset")

        # 构建类别词表：使用 caption_list 中的 caption 作为类别名称
        if category_names is not None:
            self.category_names = category_names
        else:
            raise ValueError("Please provide category list.")

    def _load_coco_dataset(self, ann_file: str, category_names):
        """加载 COCO detection 数据集并构建统一输出。"""
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))

        cat_ids = sorted(self.coco.getCatIds())
        cats = self.coco.loadCats(cat_ids)
        # captions 使用数据集类别名称
        
        if category_names is not None:
            self.category_names = category_names
        else:
            self.category_names = [[c["name"]] for c in cats]
            
        self.category_name_to_idx = {c_id: idx for idx, c_id in enumerate(cat_ids)}
        self.category_id_to_name = {c["id"]: c["name"] for c in cats}

    def __len__(self) -> int:
        return len(self.ids)

    def get_dataset_info(self):
        print(f"  == total samples: {len(self)}")
        if getattr(self, "category_names", None):
            print(f"  == categories: {len(self.category_names)}")

    def __getitem__(self, index):
        img, target = self.load_item(index)
        if self._transforms is not None:
            img, target, _ = self._transforms(img, target, self)
            
        return img, target

    def _build_target_common(self, image, rel_path, image_id, boxes, labels, area=None):
        """组装与 grounding 数据集一致的 target 字段。"""
        w, h = image.size
        # area字段：如果提供则使用（COCO），否则根据boxes计算（HF）
        if area is None:
            if len(boxes) > 0:
                # boxes 格式为 xyxy
                boxes_tensor = boxes if isinstance(boxes, torch.Tensor) else torch.as_tensor(boxes, dtype=torch.float32)
                area = (boxes_tensor[:, 2] - boxes_tensor[:, 0]) * (boxes_tensor[:, 3] - boxes_tensor[:, 1])
            else:
                area = torch.empty((0,), dtype=torch.float32)
        target = {
            "size": torch.as_tensor([int(h), int(w)]),
            "image_id": torch.tensor([image_id]),
            "filename": rel_path,
            "captions": self.category_names,
            # Always wrap boxes as TV BoundingBoxes (even when empty) so downstream transforms expect the right type.
            "boxes": convert_to_tv_tensor(boxes, key='boxes', spatial_size=(h, w)),
            "caption_attributes_dict": self.caption_attributes_dict,
            "labels": labels,  # label这里提供是为了后续数据增强时
            "area": area,
            'iscrowd': torch.zeros(len(boxes), dtype=torch.int64),
            'orig_size': torch.as_tensor([int(w), int(h)]),
            'OTA-Det': True
        }
        return target

    def load_item(self, index: int):
        """
        返回 transforms 之前的原始数据，结构与 grounding 数据集一致：
        {
            size, image_id, filename, captions, boxes, caption_attributes_dict, labels
        }
        """
        if self.dataset_format == "hf":
            item = self.dataset[index]
            
            rel_path = item["filename"]
            abs_path = os.path.join(self.root, rel_path)
            if not os.path.exists(abs_path):
                raise FileNotFoundError(f"{abs_path} not found.")
            image = Image.open(abs_path).convert('RGB')

            all_boxes = []
            all_labels = []

            # 从 regions 提取 bbox 和标签（使用 caption_list 的索引作为类别）
            for region in item["regions"]:
                all_boxes.append(region["bbox"])
                # 默认一个 region 对应一个 caption_idx
                if "caption_indices" in region and len(region["caption_indices"]) > 0:
                    all_labels.append(region["caption_indices"][0])
                else:
                    all_labels.append(0)

            boxes = torch.as_tensor(all_boxes, dtype=torch.float32) if all_boxes else torch.empty((0, 4), dtype=torch.float32)
            labels = torch.as_tensor(all_labels, dtype=torch.int64) if all_labels else torch.empty((0,), dtype=torch.int64)

            return image, self._build_target_common(image, rel_path, index, boxes, labels)

        # COCO detection 分支
        img_id = self.ids[index]
        img_info = self.coco.loadImgs(img_id)[0]
        rel_path = img_info["file_name"]
        abs_path = os.path.join(self.root, rel_path)
        image = Image.open(abs_path).convert("RGB")
        w, h = image.size

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        annos = self.coco.loadAnns(ann_ids)
        # 过滤 crowd / 空框
        annos = [a for a in annos if a.get("iscrowd", 0) == 0 and a.get("bbox", None) is not None]

        boxes, labels, areas = [], [], []
        for ann in annos:
            x, y, bw, bh = ann["bbox"]
            x1, y1, x2, y2 = x, y, x + bw, y + bh
            # clamp
            x1 = max(0, min(x1, w))
            y1 = max(0, min(y1, h))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(self.category_name_to_idx[ann["category_id"]])
            # 直接从COCO annotation中提取area
            areas.append(ann.get("area", (x2 - x1) * (y2 - y1)))

        boxes = torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.empty((0, 4), dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64) if labels else torch.empty((0,), dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.float32) if areas else torch.empty((0,), dtype=torch.float32)

        return image, self._build_target_common(image, rel_path, img_id, boxes, labels, areas)
