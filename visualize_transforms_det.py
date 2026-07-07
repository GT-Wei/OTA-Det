"""
Visualize detection transforms before/after `self._transforms`.

Usage:
    python visualize_transforms_det.py \
        --config configs/OTA-Det/OTA-Det-M/OTADet_dinov3_m_LAE.yml \
        --index 0 \
        --split train \
        --out-dir ./outputs/vis_debug

It will save two images:
    before_<idx>.png  (raw image + raw GT boxes)
    after_<idx>.png   (augmented/normalized image + boxes converted back to xyxy)
"""

import argparse
from pathlib import Path
from typing import Tuple

import torch
import torchvision.transforms.functional as TF
from torchvision.utils import draw_bounding_boxes

from engine.core import YAMLConfig


def denormalize(img: torch.Tensor, mean, std):
    """Undo Normalize to [0, 1] range."""
    mean = torch.tensor(mean, device=img.device)[:, None, None]
    std = torch.tensor(std, device=img.device)[:, None, None]
    return (img * std + mean).clamp(0.0, 1.0)


def cxcywh_to_xyxy(boxes: torch.Tensor, h: int, w: int, normalized: bool = True):
    if boxes.numel() == 0:
        return boxes
    b = boxes.clone()
    if normalized:
        b[:, 0] *= w
        b[:, 1] *= h
        b[:, 2] *= w
        b[:, 3] *= h
    cx, cy, bw, bh = b.unbind(-1)
    x1 = cx - bw / 2
    y1 = cy - bh / 2
    x2 = cx + bw / 2
    y2 = cy + bh / 2
    return torch.stack((x1, y1, x2, y2), dim=-1)


def to_xyxy_after_transforms(boxes: torch.Tensor, h: int, w: int):
    """
    Heuristic: if width/height look <= 2, assume cxcywh normalized,
    otherwise treat as already xyxy absolute.
    """
    if boxes.numel() == 0:
        return boxes
    if torch.max(boxes[:, 2:]) <= 2.0:
        return cxcywh_to_xyxy(boxes, h, w, normalized=True)
    return boxes


def save_image(path: Path, img: torch.Tensor, boxes: torch.Tensor, labels, mean, std):
    img = denormalize(img, mean, std)
    img_uint8 = (img * 255.0).clamp(0, 255).to(torch.uint8)
    if boxes is not None and boxes.numel() > 0:
        drawn = draw_bounding_boxes(img_uint8, boxes, labels=labels, colors="lime", width=2, font_size=14)
    else:
        drawn = img_uint8
    TF.to_pil_image(drawn).save(path)
    print(f"Saved {path}")


def build_label_texts(labels: torch.Tensor, target: dict):
    if labels is None or labels.numel() == 0:
        return []
    labels_list = labels.tolist()
    captions = target.get("captions") or target.get("class_texts")
    label_texts = []
    for l in labels_list:
        name = str(l)
        if captions and len(captions) > l:
            cap = captions[l]
            if isinstance(cap, list):
                name = cap[0] if len(cap) > 0 else str(l)
            else:
                name = str(cap)
        label_texts.append(f"{l}:{name}")
    return label_texts


def main(args):
    cfg = YAMLConfig(args.config)
    dataset = cfg.train_dataloader.dataset if args.split == "train" else cfg.val_dataloader.dataset

    # raw (before transforms)
    img_raw, tgt_raw = dataset.load_item(args.index)
    boxes_raw = tgt_raw.get("boxes")
    labels_raw = tgt_raw.get("labels")
    label_texts_raw = build_label_texts(labels_raw, tgt_raw)
    img_raw_tensor = TF.to_tensor(img_raw)

    # after transforms (Compose might modify inputs, so reload)
    img_aug, tgt_aug, _ = dataset._transforms(*dataset.load_item(args.index), dataset)
    boxes_aug = to_xyxy_after_transforms(tgt_aug.get("boxes"), img_aug.shape[1], img_aug.shape[2])
    labels_aug = tgt_aug.get("labels")
    label_texts_aug = build_label_texts(labels_aug, tgt_aug)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mean = args.mean
    std = args.std

    # Raw image is not normalized yet, so use identity stats.
    save_image(out_dir / f"before_{args.index}.png", img_raw_tensor, boxes_raw, label_texts_raw, [0, 0, 0], [1, 1, 1])
    save_image(out_dir / f"after_{args.index}.png", img_aug.cpu(), boxes_aug.cpu(), label_texts_aug, mean, std)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="YAML config path")
    parser.add_argument("--index", type=int, default=0, help="Sample index to visualize")
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--out-dir", type=str, default="./outputs/vis_debug")
    parser.add_argument(
        "--mean",
        type=float,
        nargs=3,
        default=[0.485, 0.456, 0.406],
        help="Normalization mean used in transforms",
    )
    parser.add_argument(
        "--std",
        type=float,
        nargs=3,
        default=[0.229, 0.224, 0.225],
        help="Normalization std used in transforms",
    )
    main(parser.parse_args())
