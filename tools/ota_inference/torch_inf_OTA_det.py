"""
DEIMv2: Real-Time Object Detection Meets DINOv3
Copyright (c) 2025 The DEIMv2 Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE)
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import json
import os
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFilter, ImageFont

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from engine.core import YAMLConfig

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
FONT_FALLBACKS = [
    os.path.join(REPO_ROOT, "font/static/Roboto-Bold.ttf"),
    os.path.join(REPO_ROOT, "font/static/Roboto-Regular.ttf"),
    os.path.join(REPO_ROOT, "font/Roboto-VariableFont_wdth,wght.ttf"),
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "arialbd.ttf",
    "arial.ttf",
]
COLOR_PALETTE = [
    (255, 107, 107),
    (84, 160, 255),
    (46, 204, 113),
    (255, 159, 67),
    (156, 136, 255),
    (253, 121, 168),
    (26, 188, 156),
    (87, 101, 116),
    (0, 206, 201),
    (232, 67, 147),
]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_font(size: int) -> ImageFont.ImageFont:
    for font_path in FONT_FALLBACKS:
        if font_path and os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
    if hasattr(draw, "textbbox"):
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    return int(draw.textlength(text, font=font)), getattr(font, "size", 14)


def shorten_text(text: str, max_chars: int = 42) -> str:
    text = " ".join(str(text).split())
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)].rstrip() + "…"


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> List[str]:
    text = " ".join(str(text).split())
    if not text:
        return [""]

    words = text.split()
    if len(words) == 1:
        lines, current = [], ""
        for char in text:
            candidate = current + char
            if text_size(draw, candidate, font)[0] <= max_width or not current:
                current = candidate
            else:
                lines.append(current)
                current = char
        if current:
            lines.append(current)
        return lines

    lines, current_words = [], []
    for word in words:
        candidate = " ".join(current_words + [word])
        if text_size(draw, candidate, font)[0] <= max_width or not current_words:
            current_words.append(word)
        else:
            lines.append(" ".join(current_words))
            current_words = [word]
    if current_words:
        lines.append(" ".join(current_words))
    return lines


def normalize_caption(raw: Any) -> str:
    if isinstance(raw, (list, tuple)):
        return " ".join(str(item).strip() for item in raw if str(item).strip())
    return str(raw).strip()


def parse_captions(captions_arg: str) -> List[str]:
    """Parse one or more query captions from JSON, txt, JSON string, |||, or short comma lists."""
    if os.path.isfile(captions_arg):
        if captions_arg.endswith(".json"):
            with open(captions_arg, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                for key in ("queries", "captions", "class_texts", "texts"):
                    if key in data:
                        data = data[key]
                        break
                else:
                    data = list(data.values())
            if not isinstance(data, list):
                raise ValueError(f"Caption JSON should contain a list/dict, got {type(data)}")
            return [normalize_caption(item) for item in data if normalize_caption(item)]

        with open(captions_arg, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    raw = captions_arg.strip()
    if not raw:
        return []

    if raw.startswith(("[", "{")):
        data = json.loads(raw)
        if isinstance(data, dict):
            for key in ("queries", "captions", "class_texts", "texts"):
                if key in data:
                    data = data[key]
                    break
            else:
                data = list(data.values())
        if not isinstance(data, list):
            raise ValueError(f"Caption JSON string should contain a list/dict, got {type(data)}")
        return [normalize_caption(item) for item in data if normalize_caption(item)]

    if "|||" in raw:
        return [text.strip() for text in raw.split("|||") if text.strip()]

    if "," in raw:
        parts = [text.strip() for text in raw.split(",") if text.strip()]
        looks_like_short_list = len(parts) > 1 and all(len(part.split()) <= 4 for part in parts)
        if looks_like_short_list:
            return parts

    return [raw]


def normalize_attr_group(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, dict):
        values = raw.get("attributes", raw.get("attrs", raw.get("texts", list(raw.values()))))
        return normalize_attr_group(values)
    if isinstance(raw, (list, tuple)):
        return [str(item).strip() for item in raw if str(item).strip()]

    text = str(raw).strip()
    if not text:
        return []
    delimiter = "&&" if "&&" in text else ","
    return [item.strip() for item in text.split(delimiter) if item.strip()]


def align_attribute_dict(data: Dict[str, Any], captions: Optional[Sequence[str]]) -> List[List[str]]:
    if not captions:
        return [normalize_attr_group(value) for value in data.values()]

    groups = []
    for idx, caption in enumerate(captions):
        value = data.get(caption, data.get(str(idx), data.get(idx, [])))
        groups.append(normalize_attr_group(value))
    return groups


def parse_attribute_groups(attr_arg: str, captions: Optional[Sequence[str]] = None) -> List[List[str]]:
    """Parse attribute groups aligned to query captions."""
    if os.path.isfile(attr_arg):
        if attr_arg.endswith(".json"):
            with open(attr_arg, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return align_attribute_dict(data, captions)
            if isinstance(data, list):
                return [normalize_attr_group(item) for item in data]
            return [normalize_attr_group(data)]

        with open(attr_arg, "r", encoding="utf-8") as f:
            return [normalize_attr_group(line.strip()) for line in f if line.strip()]

    raw = attr_arg.strip()
    if not raw:
        return []

    if raw.startswith(("[", "{")):
        data = json.loads(raw)
        if isinstance(data, dict):
            return align_attribute_dict(data, captions)
        if isinstance(data, list):
            return [normalize_attr_group(item) for item in data]
        return [normalize_attr_group(data)]

    if "|||" in raw:
        return [normalize_attr_group(block) for block in raw.split("|||") if block.strip()]

    return [normalize_attr_group(raw)]


def parse_index_list(idx_arg: Optional[str]) -> List[int]:
    if idx_arg is None:
        return []
    raw = idx_arg.strip()
    if not raw:
        return []
    if raw.startswith("["):
        data = json.loads(raw)
        return [int(item) for item in data]
    for sep in ("|||", "||", "|", ","):
        if sep in raw:
            return [int(item.strip()) for item in raw.split(sep) if item.strip()]
    return [int(raw)]


def build_attribute_inputs(
    captions: Optional[Sequence[str]],
    attribute_groups: Optional[Sequence[Sequence[str]]],
    max_attrs_per_caption: int = 10,
) -> Tuple[Optional[List[str]], Optional[torch.Tensor], Optional[torch.Tensor]]:
    if not captions or not attribute_groups:
        return None, None, None

    attr_to_idx: Dict[str, int] = {}
    attributes: List[str] = []
    cap_to_attr_map = torch.full((len(captions), max_attrs_per_caption), -1, dtype=torch.long)

    for cap_idx in range(len(captions)):
        group = attribute_groups[cap_idx] if cap_idx < len(attribute_groups) else []
        insert_pos = 0
        for attr in group:
            if insert_pos >= max_attrs_per_caption:
                break
            attr = str(attr).strip()
            if not attr:
                continue
            if attr not in attr_to_idx:
                attr_to_idx[attr] = len(attributes)
                attributes.append(attr)
            cap_to_attr_map[cap_idx, insert_pos] = attr_to_idx[attr]
            insert_pos += 1

    if not attributes:
        return None, None, None

    attr_padding_mask = torch.ones(len(attributes), dtype=torch.bool)
    return attributes, attr_padding_mask, cap_to_attr_map


def prepare_text_inputs(
    captions: Optional[List[str]],
    attribute_groups: Optional[List[List[str]]],
    max_attrs_per_caption: int,
    device: torch.device,
    captions_placeholder: bool,
    attr_priority_indices: Optional[Iterable[int]],
) -> Dict[str, Any]:
    captions_batch = [captions] if captions else None
    caption_padding_mask = [torch.ones(len(captions), dtype=torch.bool, device=device)] if captions else None

    caption_attributes_batch = None
    caption_attributes_padding_mask = None
    cap_to_attr_map_batch = None

    attr_list, attr_pad_mask, cap_to_attr_map = build_attribute_inputs(captions, attribute_groups, max_attrs_per_caption)
    if attr_list is not None:
        caption_attributes_batch = [attr_list]
        caption_attributes_padding_mask = [attr_pad_mask.to(device)]
        cap_to_attr_map_batch = cap_to_attr_map.unsqueeze(0).to(device)

    caption_score_pref = None
    if captions and attr_priority_indices:
        pref_mask = torch.zeros(len(captions), dtype=torch.bool, device=device)
        for idx in attr_priority_indices:
            if 0 <= idx < len(captions):
                pref_mask[idx] = True
        if pref_mask.any():
            caption_score_pref = pref_mask
    elif captions_placeholder and captions:
        caption_score_pref = torch.ones(len(captions), dtype=torch.bool, device=device)

    return {
        "captions": captions,
        "attribute_groups": attribute_groups,
        "captions_batch": captions_batch,
        "caption_padding_mask": caption_padding_mask,
        "caption_attributes_batch": caption_attributes_batch,
        "caption_attributes_padding_mask": caption_attributes_padding_mask,
        "cap_to_attr_map_batch": cap_to_attr_map_batch,
        "caption_score_pref": caption_score_pref,
        "captions_placeholder": captions_placeholder,
    }


def aggregate_boxes(labels: torch.Tensor, boxes: torch.Tensor, scores: torch.Tensor):
    """Aggregate detections with identical coordinates and merge query labels."""
    if len(boxes) == 0:
        return [], boxes, []

    labels_cpu = labels.detach().cpu().tolist()
    boxes_cpu = boxes.detach().cpu().tolist()
    scores_cpu = scores.detach().cpu().tolist()

    merged: Dict[Tuple[float, ...], List[Tuple[int, float]]] = {}
    for label, box, score in zip(labels_cpu, boxes_cpu, scores_cpu):
        rounded_box = tuple(round(float(value), 3) for value in box)
        merged.setdefault(rounded_box, []).append((int(label), float(score)))

    aggregated_labels, aggregated_scores, aggregated_boxes = [], [], []
    for box, pairs in merged.items():
        pairs.sort(key=lambda item: item[1], reverse=True)
        aggregated_labels.append([item[0] for item in pairs])
        aggregated_scores.append([item[1] for item in pairs])
        aggregated_boxes.append(torch.tensor(box, device=boxes.device, dtype=boxes.dtype))

    return aggregated_labels, torch.stack(aggregated_boxes), aggregated_scores


def query_tag(label: int, score: float, captions: Optional[Sequence[str]]) -> str:
    if captions and 0 <= label < len(captions):
        return f"Q{label} {shorten_text(captions[label], 22)} {score:.2f}"
    return f"Q{label} {score:.2f}"


def build_hud_card(
    captions: Optional[Sequence[str]],
    attr_groups: Optional[Sequence[Sequence[str]]],
    image_width: int,
    font: ImageFont.ImageFont,
    header: str = "MULTI-QUERY INPUT",
) -> Optional[Image.Image]:
    if not captions:
        return None

    dummy = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    max_text_width = max(260, int(image_width * 0.58))
    pad_x, pad_y = 18, 16
    line_gap = 7
    header_h = text_size(dummy, header, font)[1] + 8

    rows = []
    for idx, caption in enumerate(captions):
        color = COLOR_PALETTE[idx % len(COLOR_PALETTE)]
        content = f"Q{idx}: {caption}"
        lines = wrap_text(dummy, content, font, max_text_width)
        if attr_groups and idx < len(attr_groups) and attr_groups[idx]:
            attr_text = "Attr: " + ", ".join(str(item) for item in attr_groups[idx] if str(item).strip())
            lines.extend(wrap_text(dummy, attr_text, font, max_text_width))
        rows.append((color, lines))

    line_h = max(text_size(dummy, "Ag", font)[1] + line_gap, getattr(font, "size", 16) + 8)
    width = max(text_size(dummy, header, font)[0], max_text_width) + pad_x * 2
    height = pad_y * 2 + header_h + sum(max(1, len(lines)) * line_h + 5 for _, lines in rows)

    shadow = 12
    card = Image.new("RGBA", (width + shadow * 2, height + shadow * 2), (0, 0, 0, 0))
    shadow_layer = Image.new("RGBA", card.size, (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow_layer)
    shadow_draw.rounded_rectangle(
        [shadow + 3, shadow + 5, shadow + width + 3, shadow + height + 5],
        radius=18,
        fill=(0, 0, 0, 48),
    )
    shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(6))
    card.alpha_composite(shadow_layer)

    draw = ImageDraw.Draw(card)
    draw.rounded_rectangle(
        [shadow, shadow, shadow + width, shadow + height],
        radius=18,
        fill=(255, 255, 255, 220),
        outline=(255, 255, 255, 180),
        width=1,
    )

    x = shadow + pad_x
    y = shadow + pad_y
    draw.text((x, y), header, fill=(42, 52, 65), font=font)
    y += header_h
    draw.line((x, y, shadow + width - pad_x, y), fill=(180, 190, 200, 170), width=1)
    y += 8

    for color, lines in rows:
        draw.rounded_rectangle([x, y + 4, x + 10, y + 14], radius=3, fill=color + (255,))
        text_x = x + 18
        for line in lines:
            draw.text((text_x, y), line, fill=color, font=font)
            y += line_h
        y += 5

    return card


def render_detections(
    image: Image.Image,
    labels: List[List[int]],
    boxes: torch.Tensor,
    scores: List[List[float]],
    captions: Optional[Sequence[str]],
    attr_groups: Optional[Sequence[Sequence[str]]],
    score_threshold: float,
    hud_header: str = "MULTI-QUERY INPUT",
) -> Image.Image:
    image_rgba = image.convert("RGBA")
    overlay = Image.new("RGBA", image_rgba.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    measure_draw = ImageDraw.Draw(image_rgba)

    tag_font_size = max(14, min(28, min(image_rgba.size) // 28))
    hud_font_size = max(14, min(24, min(image_rgba.size) // 34))
    tag_font = load_font(tag_font_size)
    hud_font = load_font(hud_font_size)

    for box_idx, box in enumerate(boxes):
        valid_items = [
            (label, score)
            for label, score in zip(labels[box_idx], scores[box_idx])
            if score >= score_threshold
        ]
        if not valid_items:
            continue

        valid_items.sort(key=lambda item: item[1], reverse=True)
        visible_items = valid_items[:3]
        first_label = visible_items[0][0]
        color = COLOR_PALETTE[first_label % len(COLOR_PALETTE)]
        x1, y1, x2, y2 = [float(value) for value in box.detach().cpu().tolist()]

        draw.rectangle([x1, y1, x2, y2], fill=color + (42,))
        draw.rectangle([x1, y1, x2, y2], outline=color + (255,), width=3)

        tag_text = " / ".join(query_tag(label, score, captions) for label, score in visible_items)
        if len(valid_items) > len(visible_items):
            tag_text += f" +{len(valid_items) - len(visible_items)}"
        tag_text = shorten_text(tag_text, 72)
        text_w, text_h = text_size(measure_draw, tag_text, tag_font)
        pad_x, pad_y = 9, 5
        tag_w, tag_h = text_w + pad_x * 2, text_h + pad_y * 2
        tag_x = max(0, min(x1, image_rgba.width - tag_w - 1))
        tag_y = y1 - tag_h - 5
        if tag_y < 0:
            tag_y = min(image_rgba.height - tag_h - 1, y1 + 5)

        draw.rounded_rectangle(
            [tag_x, tag_y, tag_x + tag_w, tag_y + tag_h],
            radius=9,
            fill=color + (245,),
        )
        draw.text((tag_x + pad_x, tag_y + pad_y - 1), tag_text, fill=(255, 255, 255, 255), font=tag_font)

    image_rgba.alpha_composite(overlay)
    hud = build_hud_card(captions, attr_groups, image_rgba.width, hud_font, header=hud_header)
    if hud is not None:
        margin = max(12, min(image_rgba.size) // 50)
        image_rgba.alpha_composite(hud, (margin, margin))

    return image_rgba.convert("RGB")


def save_detections(
    images: Sequence[Image.Image],
    labels: Sequence[List[List[int]]],
    boxes: Sequence[torch.Tensor],
    scores: Sequence[List[List[float]]],
    captions: Optional[Sequence[str]],
    attr_groups: Optional[Sequence[Sequence[str]]],
    score_threshold: float,
    output_path: str,
    hud_header: str = "MULTI-QUERY INPUT",
) -> None:
    rendered = render_detections(
        images[0], labels[0], boxes[0], scores[0], captions, attr_groups, score_threshold, hud_header=hud_header
    )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    rendered.save(output_path)
    print(f"Result saved to: {output_path}")


def process_image(
    model: nn.Module,
    device: torch.device,
    file_path: str,
    text_inputs: Dict[str, Any],
    size: Tuple[int, int] = (640, 640),
    vit_backbone: bool = False,
    output_path: str = "torch_results.jpg",
    score_threshold: float = 0.45,
) -> None:
    image = Image.open(file_path).convert("RGB")
    width, height = image.size
    orig_size = torch.tensor([[width, height]], device=device)

    transforms = T.Compose([
        T.Resize(size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if vit_backbone else T.Lambda(lambda x: x),
    ])
    image_data = transforms(image).unsqueeze(0).to(device)

    labels, boxes, scores = model(
        image_data,
        orig_size,
        captions_batch=text_inputs.get("captions_batch"),
        caption_attributes_batch=text_inputs.get("caption_attributes_batch"),
        caption_padding_mask=text_inputs.get("caption_padding_mask"),
        caption_attributes_padding_mask=text_inputs.get("caption_attributes_padding_mask"),
        cap_to_attr_map_batch=text_inputs.get("cap_to_attr_map_batch"),
        score_source=text_inputs.get("score_source"),
        caption_score_pref=text_inputs.get("caption_score_pref"),
    )

    save_detections(
        [image],
        labels,
        boxes,
        scores,
        captions=text_inputs.get("display_captions") or text_inputs.get("captions"),
        attr_groups=text_inputs.get("display_attribute_groups"),
        score_threshold=score_threshold,
        output_path=output_path,
        hud_header=text_inputs.get("hud_header", "MULTI-QUERY INPUT"),
    )


def process_video(
    model: nn.Module,
    device: torch.device,
    file_path: str,
    text_inputs: Dict[str, Any],
    size: Tuple[int, int] = (640, 640),
    vit_backbone: bool = False,
    output_path: str = "torch_results.mp4",
    score_threshold: float = 0.45,
) -> None:
    cap = cv2.VideoCapture(file_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (orig_width, orig_height))

    transforms = T.Compose([
        T.Resize(size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if vit_backbone else T.Lambda(lambda x: x),
    ])

    frame_count = 0
    print(f"Processing video frames... Total: {total_frames}")
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        width, height = frame_pil.size
        orig_size = torch.tensor([[width, height]], device=device)
        image_data = transforms(frame_pil).unsqueeze(0).to(device)

        labels, boxes, scores = model(
            image_data,
            orig_size,
            captions_batch=text_inputs.get("captions_batch"),
            caption_attributes_batch=text_inputs.get("caption_attributes_batch"),
            caption_padding_mask=text_inputs.get("caption_padding_mask"),
            caption_attributes_padding_mask=text_inputs.get("caption_attributes_padding_mask"),
            cap_to_attr_map_batch=text_inputs.get("cap_to_attr_map_batch"),
            score_source=text_inputs.get("score_source"),
            caption_score_pref=text_inputs.get("caption_score_pref"),
        )

        rendered = render_detections(
            frame_pil,
            labels[0],
            boxes[0],
            scores[0],
            captions=text_inputs.get("display_captions") or text_inputs.get("captions"),
            attr_groups=text_inputs.get("display_attribute_groups"),
            score_threshold=score_threshold,
            hud_header=text_inputs.get("hud_header", "MULTI-QUERY INPUT"),
        )
        writer.write(cv2.cvtColor(np.array(rendered), cv2.COLOR_RGB2BGR))
        frame_count += 1

        if frame_count % 10 == 0:
            pct = frame_count / total_frames * 100 if total_frames else 0
            print(f"Processed {frame_count}/{total_frames} frames ({pct:.1f}%)...")

    cap.release()
    writer.release()
    print(f"Video processing complete. Result saved to: {output_path}")


def build_model(cfg: YAMLConfig, topk: int, score_source: str) -> nn.Module:
    class Model(nn.Module):
        def __init__(self, max_detections: int, default_score_source: str):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
            self.max_detections = max_detections
            self.postprocessor._force_score_source = default_score_source

        @staticmethod
        def _limit_detections(agg_labels, agg_boxes, agg_scores, max_detections):
            if max_detections <= 0 or len(agg_boxes) <= max_detections:
                return agg_labels, agg_boxes, agg_scores
            order = sorted(
                range(len(agg_boxes)),
                key=lambda idx: max(agg_scores[idx]) if agg_scores[idx] else float("-inf"),
                reverse=True,
            )[:max_detections]
            order_tensor = torch.tensor(order, device=agg_boxes.device)
            return (
                [agg_labels[idx] for idx in order],
                agg_boxes.index_select(0, order_tensor),
                [agg_scores[idx] for idx in order],
            )

        def forward(
            self,
            images,
            orig_target_sizes,
            captions_batch=None,
            caption_attributes_batch=None,
            caption_padding_mask=None,
            caption_attributes_padding_mask=None,
            cap_to_attr_map_batch=None,
            score_source=None,
            caption_score_pref=None,
        ):
            if score_source:
                self.postprocessor._force_score_source = score_source
            self.postprocessor._per_caption_score_pref = caption_score_pref

            outputs = self.model(
                images,
                captions_batch=captions_batch,
                caption_attributes_batch=caption_attributes_batch,
                caption_padding_mask=caption_padding_mask,
                caption_attributes_padding_mask=caption_attributes_padding_mask,
                cap_to_attr_map_batch=cap_to_attr_map_batch,
            )
            labels, boxes, scores = self.postprocessor(
                outputs,
                orig_target_sizes,
                cap_to_attr_map_batch=cap_to_attr_map_batch,
            )

            batch_aggregated_labels, batch_aggregated_boxes, batch_aggregated_scores = [], [], []
            for batch_idx in range(labels.shape[0]):
                agg_labels, agg_boxes, agg_scores = aggregate_boxes(labels[batch_idx], boxes[batch_idx], scores[batch_idx])
                agg_labels, agg_boxes, agg_scores = self._limit_detections(
                    agg_labels,
                    agg_boxes,
                    agg_scores,
                    self.max_detections,
                )
                batch_aggregated_labels.append(agg_labels)
                batch_aggregated_boxes.append(agg_boxes)
                batch_aggregated_scores.append(agg_scores)

            return batch_aggregated_labels, batch_aggregated_boxes, batch_aggregated_scores

    return Model(topk, score_source)


def main(args) -> None:
    cfg = YAMLConfig(args.config, resume=args.resume)

    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    if not args.resume:
        raise AttributeError("Only support resume to load model.state_dict by now.")

    checkpoint = torch.load(args.resume, map_location="cpu")
    state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
    cfg.model.load_state_dict(state)

    device = torch.device(args.device)
    model = build_model(cfg, topk=args.topk, score_source=args.score_source).to(device)
    model.eval()

    img_size = cfg.yaml_cfg["eval_spatial_size"]
    vit_backbone = cfg.yaml_cfg.get("Multi_modality_Backbone", False)

    captions = None
    if args.class_texts:
        captions = parse_captions(args.class_texts)
        print(f"Loaded {len(captions)} query text(s):")
        for idx, text in enumerate(captions):
            print(f"  Q{idx}: {shorten_text(text, 100)}")

    attribute_groups = None
    if args.attr_texts:
        attribute_groups = parse_attribute_groups(args.attr_texts, captions)
        print(f"Loaded {len(attribute_groups)} attribute group(s):")
        for idx, group in enumerate(attribute_groups):
            print(f"  Attr{idx}: {shorten_text(', '.join(group), 100)}")

    captions_placeholder = False
    if (not captions) and attribute_groups:
        captions = [" ".join(group) if group else f"query_{idx}" for idx, group in enumerate(attribute_groups)]
        captions_placeholder = True
        print(f"Derived {len(captions)} query text(s) from attribute groups.")

    if captions and attribute_groups and len(attribute_groups) != len(captions):
        print(
            f"Note: {len(attribute_groups)} attribute group(s) provided for {len(captions)} query text(s); "
            "extra groups are ignored and missing groups are empty."
        )

    if not captions:
        print("No query text provided; running model default behavior.")

    attr_priority_indices = parse_index_list(args.attr_priority)
    text_inputs = prepare_text_inputs(
        captions,
        attribute_groups,
        args.max_attrs_per_caption,
        device,
        captions_placeholder,
        attr_priority_indices,
    )
    if args.score_source == "auto":
        effective_score_source = "attr" if captions_placeholder else "caption"
    else:
        effective_score_source = args.score_source
    text_inputs["score_source"] = effective_score_source
    if captions_placeholder and attribute_groups:
        # Attribute-only input: the visualization should reflect the attribute detection head,
        # so show the attribute set itself as the query instead of a synthesized sentence.
        text_inputs["display_captions"] = [
            ", ".join(str(a) for a in group if str(a).strip()) or f"query_{idx}"
            for idx, group in enumerate(attribute_groups)
        ]
        text_inputs["display_attribute_groups"] = None
        text_inputs["hud_header"] = "ATTRIBUTE-SET INPUT"
    else:
        text_inputs["display_captions"] = captions
        text_inputs["display_attribute_groups"] = attribute_groups if args.show_attrs_in_hud else None
        text_inputs["hud_header"] = "MULTI-QUERY INPUT"
    print(f"Score source: {effective_score_source}")
    if attribute_groups and not captions_placeholder and not args.show_attrs_in_hud:
        print("HUD attribute lines are hidden by default; add --show-attrs-in-hud to display them.")

    file_path = args.input
    file_ext = os.path.splitext(file_path)[-1].lower()
    if args.output:
        output_path = args.output
    else:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_path = f"{base_name}_result.jpg" if file_ext in IMAGE_EXTS else f"{base_name}_result.mp4"

    with torch.no_grad():
        if file_ext in IMAGE_EXTS:
            print(f"Processing image: {file_path}")
            process_image(model, device, file_path, text_inputs, img_size, vit_backbone, output_path, args.score_threshold)
            print("Image processing complete.")
        else:
            print(f"Processing video: {file_path}")
            process_video(model, device, file_path, text_inputs, img_size, vit_backbone, output_path, args.score_threshold)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OTA-Det image/video inference with multi-query and attribute-set prompts")
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to config file")
    parser.add_argument("-r", "--resume", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to input image or video")
    parser.add_argument("-d", "--device", type=str, default="cuda", help="Device to run inference on, e.g. cpu/cuda/cuda:0")
    parser.add_argument(
        "-t",
        "--class-texts",
        type=str,
        default=None,
        help=(
            "Query texts. Supports JSON file/list string, txt file, 'q1|||q2', short comma lists, "
            "single words, and full sentences. Use JSON or ||| for sentences containing commas."
        ),
    )
    parser.add_argument(
        "-a",
        "--attr-texts",
        type=str,
        default=None,
        help=(
            "Attribute groups aligned to queries. Supports JSON list-of-lists/dict, txt one group per line, "
            "'red,car|||white,building', or a single attribute set. If -t is omitted, queries are derived from attributes."
        ),
    )
    parser.add_argument("-o", "--output", type=str, default=None, help="Output file path")
    parser.add_argument("-s", "--score-threshold", type=float, default=0.4, help="Detection score threshold")
    parser.add_argument(
        "--topk",
        type=int,
        default=0,
        help="Keep top K aggregated detections per image; 0 keeps all detections above threshold. Use --topk 1 for REC-style RSVG evaluation.",
    )
    parser.add_argument("--max-attrs-per-caption", type=int, default=10, help="Maximum attributes to use per query")
    parser.add_argument("--show-attrs-in-hud", action="store_true", help="Display aligned attribute groups in the visualization HUD. Hidden by default to keep inference figures clean.")
    parser.add_argument(
        "--score-source",
        type=str,
        default="auto",
        choices=["auto", "caption", "attr"],
        help="Ranking score source. auto uses attr when queries are derived from attribute-only input, otherwise caption.",
    )
    parser.add_argument(
        "--attr-priority",
        type=str,
        default=None,
        help="Comma/pipe/JSON list of 0-based query indices that should use attribute scores.",
    )

    main(parser.parse_args())
