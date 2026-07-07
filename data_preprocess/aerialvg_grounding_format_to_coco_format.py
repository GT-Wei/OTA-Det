#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert Grounding JSONL (caption + regions) → COCO JSON format
Each JSONL line is treated as one independent image entry.

python aerialvg_grounding_format_to_coco_format.py \
  --jsonl ../datasets/LLM_Caption_Parse/annotations_grounding/Aerial_VG_JSONL/vg_train_odvg.jsonl \
  --save ../datasets/LLM_Caption_Parse/annotations_grounding/Aerial_VG_COCO/vg_train_odvg_coco.json
  
python aerialvg_grounding_format_to_coco_format.py \
  --jsonl ../datasets/LLM_Caption_Parse/annotations_grounding/Aerial_VG_JSONL/vg_test_odvg.jsonl \
  --save ../datasets/LLM_Caption_Parse/annotations_grounding/Aerial_VG_COCO/vg_test_odvg_coco.json
  
python aerialvg_grounding_format_to_coco_format.py \
  --jsonl ../datasets/LLM_Caption_Parse/annotations_grounding/Aerial_VG_JSONL/vg_val_odvg.jsonl \
  --save ../datasets/LLM_Caption_Parse/annotations_grounding/Aerial_VG_COCO/vg_val_odvg_coco.json
  
python aerialvg_grounding_format_to_coco_format.py \
  --jsonl ../datasets/LLM_Caption_Parse/annotations_grounding/DIOR_RSVG_JSONL/DIOR-RSVG_train.jsonl \
  --save ../datasets/LLM_Caption_Parse/annotations_grounding/DIOR_RSVG_COCO/DIOR-RSVG_train_coco.json
  
python aerialvg_grounding_format_to_coco_format.py \
  --jsonl ../datasets/LLM_Caption_Parse/annotations_grounding/DIOR_RSVG_JSONL/DIOR-RSVG_train.jsonl \
  --save ../datasets/LLM_Caption_Parse/annotations_grounding/DIOR_RSVG_COCO/DIOR-RSVG_train_coco.json
  
python aerialvg_grounding_format_to_coco_format.py \
  --jsonl ../datasets/LLM_Caption_Parse/annotations_grounding/DIOR_RSVG_JSONL/DIOR-RSVG_val.jsonl \
  --save ../datasets/LLM_Caption_Parse/annotations_grounding/DIOR_RSVG_COCO/DIOR-RSVG_val_coco.json

python aerialvg_grounding_format_to_coco_format.py \
  --jsonl ../datasets/LLM_Caption_Parse/annotations_grounding/DIOR_RSVG_JSONL/DIOR-RSVG_test.jsonl \
  --save ../datasets/LLM_Caption_Parse/annotations_grounding/DIOR_RSVG_COCO/DIOR-RSVG_test_coco.json
  
python aerialvg_grounding_format_to_coco_format.py \
  --jsonl ../datasets/LLM_Caption_Parse/annotations_grounding/OPT_RSVG_JSONL/OPT_RSVG_test.jsonl \
  --save ../datasets/LLM_Caption_Parse/annotations_grounding/OPT_RSVG_COCO/OPT_RSVG_test_coco.json
  
python aerialvg_grounding_format_to_coco_format.py \
  --jsonl ../datasets/LLM_Caption_Parse/annotations_grounding/OPT_RSVG_JSONL/OPT_RSVG_train.jsonl \
  --save ../datasets/LLM_Caption_Parse/annotations_grounding/OPT_RSVG_COCO/OPT_RSVG_train_coco.json

python aerialvg_grounding_format_to_coco_format.py \
  --jsonl ../datasets/LLM_Caption_Parse/annotations_grounding/OPT_RSVG_JSONL/OPT_RSVG_val.jsonl \
  --save ../datasets/LLM_Caption_Parse/annotations_grounding/OPT_RSVG_COCO/OPT_RSVG_val_coco.json
  
"""

import os
import json
from tqdm import tqdm
from pathlib import Path

def grounding_to_coco(jsonl_path: str, save_path: str):
    jsonl_path = Path(jsonl_path)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    images, annotations = [], []
    ann_id, img_id = 1, 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Converting grounding → COCO"):
            line = line.strip()
            if not line:
                continue
            meta = json.loads(line)
            fname = meta["filename"]
            height, width = meta["height"], meta["width"]
            grounding = meta["grounding"]
            caption = grounding.get("caption", "")

            # --- 每条样本都作为一个独立 image ---
            images.append({
                "id": img_id,
                "file_name": fname,
                "height": height,
                "width": width,
                "image_caption": caption,
            })

            # --- 每个 region 变成一个 annotation ---
            # 仅要第一个，因为其他都是relation
            region = grounding.get("regions", [])[0]
            # print(region)
            bbox = region["bbox"]
            if len(bbox) != 4:
                continue
            x1, y1, x2, y2 = bbox
            w, h = x2 - x1, y2 - y1
            area = w * h if w > 0 and h > 0 else 0

            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "bbox": [x1, y1, w, h],
                "category_id": 0,        # 单类任务
                "iscrowd": 0,
                "area": area
            })
            ann_id += 1

            img_id += 1

    coco_dict = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 0, "name": "object"}]
    }

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(coco_dict, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Done! Saved COCO JSON to: {save_path}")
    print(f"Total images: {len(images)} | annotations: {len(annotations)}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True, help="path to grounding .jsonl file")
    parser.add_argument("--save", required=True, help="path to save coco .json")
    args = parser.parse_args()

    grounding_to_coco(args.jsonl, args.save)
