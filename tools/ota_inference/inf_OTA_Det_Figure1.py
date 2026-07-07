"""
DEIMv2: Real-Time Object Detection Meets DINOv3
Copyright (c) 2025 The DEIMv2 Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE)
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import os
import sys
import json
import traceback

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont, ImageFilter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from engine.core import YAMLConfig

# --- 1. Nano Banana Style Configuration ---

# 字体路径配置
FONT_PATH_TAG = "./font/static/Roboto-Bold.ttf"
FONT_PATH_LEGEND = "./font/static/Roboto-Medium.ttf"
# 备用字体路径
FONT_FALLBACK_PATHS = [
    "./font/static/Roboto-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "arialbd.ttf",
    "arial.ttf"
]

# Nano Banana Palette
COLOR_PALETTE = [
    (255, 107, 107),  # [0] Pastel Red
    (84, 160, 255),   # [1] Soft Blue
    (46, 204, 113),   # [2] Emerald Green
    (255, 159, 67),   # [3] Orange
    (156, 136, 255),  # [4] Soft Purple
    (253, 121, 168),  # [5] Sakura Pink
    (26, 188, 156),   # [6] Turquoise
]

HUD_BG_COLOR = (255, 255, 255, 217)    # rgba(255,255,255,0.85)
HUD_STROKE_COLOR = (255, 255, 255, 153) # 0.6 alpha
HUD_TEXT_COLOR = (68, 68, 68)          # #444
HUD_HEADER_COLOR = (85, 85, 85)        # #555
HUD_MAX_WIDTH_RATIO = 0.618             # HUD width ratio relative to image

# --- 2. Helper Functions (Visualization) ---

def load_font(specific_path, fallback_size=18):
    """Load font with fallbacks"""
    candidates = [specific_path] + FONT_FALLBACK_PATHS
    for path in candidates:
        if path and os.path.exists(path):
            try:
                return ImageFont.truetype(path, fallback_size)
            except Exception:
                continue
    return ImageFont.load_default()

def split_text(draw, text, font, max_width):
    """Text wrapping helper"""
    words = text.split()
    lines = []
    current_line = []
    for word in words:
        test_line = ' '.join(current_line + [word])
        w = 0
        if hasattr(draw, "textbbox"):
            bbox = draw.textbbox((0, 0), test_line, font=font)
            w = bbox[2] - bbox[0]
        else:
            w = font.getlength(test_line)
            
        if w <= max_width:
            current_line.append(word)
        else:
            if current_line: lines.append(' '.join(current_line))
            current_line = [word]
    if current_line: lines.append(' '.join(current_line))
    return lines

def draw_rounded_rect(draw, box, radius, fill=None, outline=None, width=1):
    x1, y1, x2, y2 = box
    draw.rounded_rectangle([x1, y1, x2, y2], radius=radius, fill=fill, outline=outline, width=width)

def create_glass_hud_nano(lines_data, font, padding=20):
    """Create the Glass HUD image"""
    dummy_draw = ImageDraw.Draw(Image.new("RGB", (1,1)))
    
    # Calculate Line Height
    if hasattr(dummy_draw, "textbbox"):
        bbox = dummy_draw.textbbox((0,0), "Ag", font=font)
        line_height = bbox[3] - bbox[1] + 10 
    else:
        line_height = 28

    header_text = "Input:"
    header_font = font 
    header_h = line_height + 8 
    
    # Calculate Max Width
    max_w = 0
    if hasattr(dummy_draw, "textlength"):
        header_w = dummy_draw.textlength(header_text, font=header_font)
    else:
        header_w = header_font.getlength(header_text)
    max_w = max(max_w, header_w)

    parsed_lines = []
    for item in lines_data:
        prefix = item['prefix']
        content_lines = item['lines']
        color = item['color']
        
        for i, line in enumerate(content_lines):
            if i == 0:
                prefix_w = dummy_draw.textlength(prefix, font=font)
                text_w = dummy_draw.textlength(line, font=font)
                total_w = prefix_w + text_w
                parsed_lines.append({
                    "type": "content", 
                    "prefix": prefix, 
                    "text": line, 
                    "color": color,  # apply palette color to prefix & text
                    "prefix_w": prefix_w
                })
            else:
                indent_str = "   "
                indent_w = dummy_draw.textlength(indent_str, font=font)
                text_w = dummy_draw.textlength(line, font=font)
                total_w = indent_w + text_w
                parsed_lines.append({
                    "type": "continuation", 
                    "indent": indent_str,
                    "text": line, 
                    "indent_w": indent_w,
                    "color": color
                })
            max_w = max(max_w, total_w)
            
    w = max_w + padding * 2
    h = (padding) + (header_h) + (len(parsed_lines) * line_height) + (padding)
    
    # Draw
    shadow_blur = 10
    card_img = Image.new("RGBA", (int(w + shadow_blur*2), int(h + shadow_blur*2)), (0,0,0,0))
    
    # Shadow
    shadow_draw = ImageDraw.Draw(card_img)
    shadow_box = [shadow_blur+2, shadow_blur+4, shadow_blur+w+2, shadow_blur+h+4]
    shadow_draw.rounded_rectangle(shadow_box, radius=12, fill=(0,0,0,40)) 
    card_img = card_img.filter(ImageFilter.GaussianBlur(5))
    
    # Glass Body
    card_draw = ImageDraw.Draw(card_img)
    main_box = [shadow_blur, shadow_blur, shadow_blur+w, shadow_blur+h]
    card_draw.rounded_rectangle(main_box, radius=12, fill=HUD_BG_COLOR, outline=HUD_STROKE_COLOR, width=1)
    
    # Content
    curr_x = shadow_blur + padding
    curr_y = shadow_blur + padding
    
    card_draw.text((curr_x, curr_y), header_text.upper(), fill=HUD_HEADER_COLOR, font=header_font)
    
    line_y = curr_y + line_height
    card_draw.line([(curr_x, line_y), (curr_x + max_w, line_y)], fill=(200, 200, 200, 180), width=1)
    
    curr_y += header_h + 4
    
    for line_item in parsed_lines:
        if line_item["type"] == "content":
            # Use palette color for both prefix and text to distinguish per-caption
            text_color = line_item.get("color", HUD_TEXT_COLOR)
            card_draw.text((curr_x, curr_y), line_item["prefix"], fill=text_color, font=font, stroke_width=0)
            card_draw.text((curr_x + line_item["prefix_w"], curr_y), line_item["text"], fill=text_color, font=font)
        else:
            text_color = line_item.get("color", HUD_TEXT_COLOR)
            card_draw.text((curr_x + line_item["indent_w"], curr_y), line_item["text"], fill=text_color, font=font)
        curr_y += line_height
        
    return card_img

# --- 3. Core Logic (DEIMv2 + Processing) ---

def parse_captions(captions_arg):
    """
    Parse captions from argument string or file.
    """
    if os.path.isfile(captions_arg):
        if captions_arg.endswith('.json'):
            with open(captions_arg, 'r', encoding='utf-8') as f:
                captions = json.load(f)
                if not isinstance(captions, list):
                    raise ValueError(f"JSON file should contain a list, got {type(captions)}")
                return captions
        else:
            with open(captions_arg, 'r', encoding='utf-8') as f:
                captions = [line.strip() for line in f if line.strip()]
                return captions
    else:
        if captions_arg.strip().startswith('['):
            try:
                captions = json.loads(captions_arg)
                if not isinstance(captions, list):
                    raise ValueError(f"JSON string should be a list, got {type(captions)}")
                return captions
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON string: {e}")
        
        if '|||' in captions_arg:
            captions = [text.strip() for text in captions_arg.split('|||')]
            return captions
        
        captions = [text.strip() for text in captions_arg.split(',')]
        return captions

def parse_attribute_groups(attr_arg):
    """
    Parse attribute groups for captions.
    Supported formats:
      1) JSON file (list of lists / list of strings / dict of caption->list)
      2) Text file (one attribute group per line, comma or && separated)
      3) JSON string
      4) Groups separated by '|||', attributes separated by ',' or '&&'
    Returns: List[List[str]]
    """
    def normalize_group(raw):
        if raw is None:
            return []
        if isinstance(raw, (list, tuple)):
            return [str(x).strip() for x in raw if str(x).strip()]
        raw = str(raw)
        if '&&' in raw:
            parts = raw.split('&&')
        else:
            parts = raw.split(',')
        return [p.strip() for p in parts if p.strip()]

    def to_groups(data):
        if isinstance(data, dict):
            return [normalize_group(v) for v in data.values()]
        if isinstance(data, (list, tuple)):
            return [normalize_group(v) for v in data]
        return [normalize_group(data)]

    if os.path.isfile(attr_arg):
        if attr_arg.endswith('.json'):
            with open(attr_arg, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return to_groups(data)
        with open(attr_arg, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        return [normalize_group(line) for line in lines]

    if attr_arg.strip().startswith(('{', '[')):
        data = json.loads(attr_arg)
        return to_groups(data)

    if '|||' in attr_arg:
        return [normalize_group(block) for block in attr_arg.split('|||') if block.strip()]

    return [normalize_group(attr_arg)]

def parse_index_list(idx_arg):
    """
    Parse indices from comma/pipe separated string or JSON list.
    """
    if idx_arg is None:
        return []
    idx_arg = idx_arg.strip()
    if not idx_arg:
        return []
    if idx_arg[0] in '[{':
        try:
            data = json.loads(idx_arg)
            if isinstance(data, (list, tuple)):
                return [int(x) for x in data]
        except Exception:
            pass
    separators = [',', '|', '||', '|||']
    for sep in separators:
        if sep in idx_arg:
            try:
                return [int(x.strip()) for x in idx_arg.split(sep) if x.strip()!='']
            except Exception:
                break
    try:
        return [int(idx_arg)]
    except Exception:
        return []

def build_attribute_inputs(captions, attribute_groups, max_attrs_per_caption=10):
    """
    Build flattened attribute list, padding mask, and caption->attribute map.
    """
    if not captions or not attribute_groups:
        return None, None, None

    num_captions = len(captions)
    attr_to_idx = {}
    attributes = []
    cap_to_attr_map = torch.full((num_captions, max_attrs_per_caption), -1, dtype=torch.long)

    for cap_idx in range(num_captions):
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

    attr_padding_mask = torch.ones(len(attributes), dtype=torch.bool) if attributes else torch.zeros(0, dtype=torch.bool)
    return attributes, attr_padding_mask, cap_to_attr_map

def prepare_text_prompts(captions, attribute_groups, max_attrs_per_caption, device, captions_placeholder=False, attr_priority_indices=None):
    """
    Package captions/attributes and corresponding masks for model forward.
    """
    captions_batch = [captions] if captions else None
    caption_padding_mask = [torch.ones(len(captions), dtype=torch.bool, device=device)] if captions else None

    caption_attributes_batch = None
    caption_attributes_padding_mask = None
    cap_to_attr_map_batch = None
    cap_to_attr_map = None

    if captions and attribute_groups:
        attr_list, attr_pad_mask, cap_to_attr_map = build_attribute_inputs(captions, attribute_groups, max_attrs_per_caption)
        if attr_list is not None and len(attr_list) > 0:
            caption_attributes_batch = [attr_list]
            caption_attributes_padding_mask = [attr_pad_mask.to(device)]
    if cap_to_attr_map is not None:
        cap_to_attr_map_batch = cap_to_attr_map.unsqueeze(0).to(device)

    caption_score_pref = None
    if captions and attribute_groups and attr_priority_indices:
        mask = torch.zeros(len(captions), dtype=torch.bool, device=device)
        for idx in attr_priority_indices:
            if 0 <= idx < len(captions):
                mask[idx] = True
        if mask.any():
            caption_score_pref = mask
    elif captions_placeholder and captions:
        # If captions were auto-generated placeholders, prefer attribute scores for all
        caption_score_pref = torch.ones(len(captions), dtype=torch.bool, device=device)

    return {
        "captions": captions,
        "attribute_groups": attribute_groups,
        "captions_batch": captions_batch,
        "caption_padding_mask": caption_padding_mask,
        "caption_attributes_batch": caption_attributes_batch,
        "caption_attributes_padding_mask": caption_attributes_padding_mask,
        "cap_to_attr_map_batch": cap_to_attr_map_batch,
        "captions_placeholder": captions_placeholder,
        "caption_score_pref": caption_score_pref,
    }

def aggregate_boxes(labels, boxes, scores):
    """Aggregate detections with identical coordinates and merge their labels."""
    if len(boxes) == 0:
        return [], boxes, []

    labels_cpu = labels.detach().cpu().tolist()
    boxes_cpu = boxes.detach().cpu().tolist()
    scores_cpu = scores.detach().cpu().tolist()

    merged = {}
    for label, box, score in zip(labels_cpu, boxes_cpu, scores_cpu):
        merged.setdefault(tuple(box), []).append((int(label), float(score)))

    aggregated_labels = []
    aggregated_scores = []
    aggregated_boxes = []

    for box, pairs in merged.items():
        pairs.sort(key=lambda item: item[1], reverse=True)
        aggregated_labels.append([item[0] for item in pairs])
        aggregated_scores.append([item[1] for item in pairs])
        aggregated_boxes.append(torch.tensor(box, device=boxes.device, dtype=boxes.dtype))

    aggregated_boxes = torch.stack(aggregated_boxes) if aggregated_boxes else boxes[:0]
    return aggregated_labels, aggregated_boxes, aggregated_scores


def draw_nano_style(images, labels, boxes, scores, captions=None, attr_groups=None, thrh=0.45,
                    output_path='torch_results.jpg', score_source='caption', caption_score_pref=None):
    """
    Draw using Nano Banana Style:
    1. Glass HUD for captions.
    2. Semi-transparent filled BBoxes.
    3. Pill-shaped tags (Index or Multi-index).
    """
    # Load fonts
    font_tag = load_font(FONT_PATH_TAG, 30)
    font_hud = load_font(FONT_PATH_LEGEND, 30)
    dummy_draw = ImageDraw.Draw(Image.new("RGB", (1,1)))

    for i, im in enumerate(images):
        im = im.convert("RGBA")
        overlay = Image.new("RGBA", im.size, (0,0,0,0))
        draw_overlay = ImageDraw.Draw(overlay)
        draw_img = ImageDraw.Draw(im) # for text measurement fallback

        # --- Draw Boxes and Pills ---
        box_list = boxes[i]
        label_list = labels[i]
        score_list = scores[i]
        
        for j, b in enumerate(box_list):
            raw_labels = label_list[j]
            raw_scores = score_list[j]
            
            # Filter by threshold
            valid_items = [(l, s) for l, s in zip(raw_labels, raw_scores) if s >= thrh]
            if not valid_items:
                continue
            
            # Sort by score desc for stable display
            valid_items = sorted(valid_items, key=lambda x: x[1], reverse=True)
            box_labels, box_scores = zip(*valid_items)
            best_score = box_scores[0]
            
            # Use color of the first label
            first_label_idx = box_labels[0]
            base_color = COLOR_PALETTE[first_label_idx % len(COLOR_PALETTE)]
            
            # Nano Style Colors
            fill_color = base_color + (46,)   # ~18% alpha
            stroke_color = base_color + (255,) # 100% alpha
            
            # Coordinates
            x1, y1, x2, y2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
            
            # 1. Draw Box (Fill + Stroke)
            draw_overlay.rectangle([x1, y1, x2, y2], fill=fill_color)
            draw_overlay.rectangle([x1, y1, x2, y2], outline=stroke_color, width=3)
            
            # 2. Draw Nano Pill Tag
            # Text content with per-label score: "0:0.88/1:0.76"
            tag_pairs = [f"{lid}:{score:.2f}" for lid, score in valid_items]
            tag_txt = '/'.join(tag_pairs)
            
            # Calculate size
            if hasattr(draw_img, "textbbox"):
                bbox = draw_img.textbbox((0, 0), tag_txt, font=font_tag)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            else:
                tw = font_tag.getlength(tag_txt)
                th = 14
            
            tag_pad_x, tag_pad_y = 8, 4
            tag_w = tw + tag_pad_x * 2
            tag_h = th + tag_pad_y * 2
            
            tag_x = x1
            tag_y = max(0, y1 - tag_h - 4) # 4px gap above box
            
            # Draw Pill Background
            draw_rounded_rect(draw_overlay, [tag_x, tag_y, tag_x+tag_w, tag_y+tag_h], radius=8, fill=stroke_color)
            
            # Draw Text
            text_draw_y = tag_y + tag_pad_y - 1 
            draw_overlay.text((tag_x + tag_pad_x, text_draw_y), tag_txt, fill="white", font=font_tag)

        # Composite overlay onto image
        im = Image.alpha_composite(im, overlay)
        
        # --- Draw Glass HUD ---
        # Only draw HUD if we have captions to show
        pref_mask = None
        if caption_score_pref is not None:
            if torch.is_tensor(caption_score_pref):
                pref_mask = caption_score_pref.detach().cpu().to(torch.bool).tolist()
            else:
                pref_mask = [bool(x) for x in caption_score_pref]

        if captions and len(captions) > 0:
            hud_lines_data = []
            hud_max_width = int(im.width * HUD_MAX_WIDTH_RATIO)
            
            for idx, caption in enumerate(captions):
                color = COLOR_PALETTE[idx % len(COLOR_PALETTE)]
                prefix = f"[{idx}] "
                
                # Calculate available width for text
                prefix_w = dummy_draw.textlength(prefix, font=font_hud)
                avail_w = hud_max_width - prefix_w - 40
                
                prefer_attr = pref_mask[idx] if pref_mask and idx < len(pref_mask) else False
                mode = score_source

                show_caption = False
                show_attr = False
                if mode == 'avg':
                    show_caption = True
                    show_attr = True
                else:
                    if prefer_attr or mode == 'attr':
                        show_attr = True
                    else:
                        show_caption = True

                hud_lines = []
                if show_caption:
                    hud_lines.extend(split_text(dummy_draw, caption, font_hud, avail_w))
                if show_attr and attr_groups and idx < len(attr_groups):
                    attrs = [a for a in attr_groups[idx] if a]
                    if attrs:
                        attr_text = "Attr: " + ", ".join(attrs)
                        hud_lines.extend(split_text(dummy_draw, attr_text, font_hud, avail_w))
                # Fallback: if no lines were added, still show caption text
                if not hud_lines:
                    hud_lines.extend(split_text(dummy_draw, caption, font_hud, avail_w))

                hud_lines_data.append({"prefix": prefix, "lines": hud_lines, "color": color})
                
            # if hud_lines_data:
            #     hud_card = create_glass_hud_nano(hud_lines_data, font_hud)
            #     im.paste(hud_card, (20, 20), hud_card)
        
        # Save
        im.convert("RGB").save(output_path)
        print(f"Result saved to: {output_path}")

def process_image(model, device, file_path, text_inputs, size=(640, 640), 
                 vit_backbone=False, output_path='torch_results.jpg', score_threshold=0.45):
    """Process a single image"""
    im_pil = Image.open(file_path).convert('RGB')
    w, h = im_pil.size
    orig_size = torch.tensor([[w, h]]).to(device)

    transforms = T.Compose([
        T.Resize(size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
                if vit_backbone else T.Lambda(lambda x: x)
    ])
    im_data = transforms(im_pil).unsqueeze(0).to(device)

    captions = text_inputs.get("captions")
    captions_batch = text_inputs.get("captions_batch")
    caption_padding_mask = text_inputs.get("caption_padding_mask")
    caption_attributes_batch = text_inputs.get("caption_attributes_batch")
    caption_attributes_padding_mask = text_inputs.get("caption_attributes_padding_mask")
    cap_to_attr_map_batch = text_inputs.get("cap_to_attr_map_batch")
    caption_score_pref = text_inputs.get("caption_score_pref")
    
    output = model(
        im_data,
        orig_size,
        captions_batch,
        caption_attributes_batch=caption_attributes_batch,
        caption_padding_mask=caption_padding_mask,
        caption_attributes_padding_mask=caption_attributes_padding_mask,
        cap_to_attr_map_batch=cap_to_attr_map_batch,
        score_source=text_inputs.get("score_source"),
        caption_score_pref=caption_score_pref
    )
    labels, boxes, scores = output

    # Pass captions to draw function for HUD
    draw_nano_style([im_pil], labels, boxes, scores, captions=captions, attr_groups=text_inputs.get("attribute_groups"),
                    thrh=score_threshold, output_path=output_path, score_source=text_inputs.get("score_source"),
                    caption_score_pref=text_inputs.get("caption_score_pref"))


def process_video(model, device, file_path, text_inputs, size=(640, 640), 
                 vit_backbone=False, output_path='torch_results.mp4', score_threshold=0.45):
    """Process a video file"""
    cap = cv2.VideoCapture(file_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (orig_w, orig_h))

    transforms = T.Compose([
        T.Resize(size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
                if vit_backbone else T.Lambda(lambda x: x)
    ])

    captions_batch = text_inputs.get("captions_batch")
    caption_padding_mask = text_inputs.get("caption_padding_mask")
    caption_attributes_batch = text_inputs.get("caption_attributes_batch")
    caption_attributes_padding_mask = text_inputs.get("caption_attributes_padding_mask")
    cap_to_attr_map_batch = text_inputs.get("cap_to_attr_map_batch")
    caption_score_pref = text_inputs.get("caption_score_pref")

    frame_count = 0
    print(f"Processing video frames... Total: {total_frames}")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        w, h = frame_pil.size
        orig_size = torch.tensor([[w, h]]).to(device)

        im_data = transforms(frame_pil).unsqueeze(0).to(device)

        output = model(
            im_data,
            orig_size,
            captions_batch,
            caption_attributes_batch=caption_attributes_batch,
            caption_padding_mask=caption_padding_mask,
            caption_attributes_padding_mask=caption_attributes_padding_mask,
            cap_to_attr_map_batch=cap_to_attr_map_batch,
            score_source=text_inputs.get("score_source"),
            caption_score_pref=caption_score_pref
        )
        labels, boxes, scores = output

        # Temporary path for drawing
        temp_path = f'temp_frame_{frame_count}.jpg'
        draw_nano_style([frame_pil], labels, boxes, scores, captions=text_inputs.get("captions"),
                        attr_groups=text_inputs.get("attribute_groups"), thrh=score_threshold, output_path=temp_path,
                        score_source=text_inputs.get("score_source"), caption_score_pref=caption_score_pref)
        
        frame_drawn = Image.open(temp_path)
        os.remove(temp_path)

        frame_cv = cv2.cvtColor(np.array(frame_drawn), cv2.COLOR_RGB2BGR)
        out.write(frame_cv)
        frame_count += 1

        if frame_count % 10 == 0:
            print(f"Processed {frame_count}/{total_frames} frames ({frame_count/total_frames*100:.1f}%)...")

    cap.release()
    out.release()
    print(f"Video processing complete. Result saved to: {output_path}")


def main(args):
    """Main function"""
    cfg = YAMLConfig(args.config, resume=args.resume)

    if 'HGNetv2' in cfg.yaml_cfg:
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
    else:
        raise AttributeError('Only support resume to load model.state_dict by now.')

    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self, score_source="caption"):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
            # Default score source (caption | attr | avg), can be overridden per-call
            self.default_score_source = score_source
            self.postprocessor._force_score_source = score_source

        def forward(self, images, orig_target_sizes, captions_batch=None, caption_attributes_batch=None,
                    caption_padding_mask=None, caption_attributes_padding_mask=None, cap_to_attr_map_batch=None,
                    score_source=None, caption_score_pref=None):
            # Update score source per call if provided
            if score_source:
                self.postprocessor._force_score_source = score_source
            if caption_score_pref is not None:
                self.postprocessor._per_caption_score_pref = caption_score_pref
            else:
                self.postprocessor._per_caption_score_pref = None
            outputs = self.model(
                images,
                captions_batch=captions_batch,
                caption_attributes_batch=caption_attributes_batch,
                caption_padding_mask=caption_padding_mask,
                caption_attributes_padding_mask=caption_attributes_padding_mask,
                cap_to_attr_map_batch=cap_to_attr_map_batch
            )
            outputs = self.postprocessor(outputs, orig_target_sizes, cap_to_attr_map_batch)
            
            labels, boxes, scores = outputs
            
            batch_aggregated_labels = []
            batch_aggregated_boxes = []
            batch_aggregated_scores = []
            
            for b in range(labels.shape[0]):
                agg_labels, agg_boxes, agg_scores = aggregate_boxes(
                    labels[b], boxes[b], scores[b]
                )
                batch_aggregated_labels.append(agg_labels)
                batch_aggregated_boxes.append(agg_boxes)
                batch_aggregated_scores.append(agg_scores)
            
            return batch_aggregated_labels, batch_aggregated_boxes, batch_aggregated_scores

    device = torch.device(args.device)
    model = Model(score_source=args.score_source).to(device)
    model.eval()
    
    img_size = cfg.yaml_cfg["eval_spatial_size"]
    vit_backbone = cfg.yaml_cfg.get('Multi_modality_Backbone', False)

    # Parse captions
    if args.class_texts: 
        try:
            captions = parse_captions(args.class_texts) 
            print(f"Successfully loaded {len(captions)} class texts:")
            for idx, text in enumerate(captions):
                display_text = text if len(text) <= 100 else text[:100] + "..."
                print(f"  [{idx}] {display_text}")
        except Exception as e:
            print(f"Error parsing class texts: {e}")
            raise
    else:
        captions = None
        print("No class texts provided, using default behavior")

    # Parse attribute groups
    if args.attr_texts:
        try:
            attribute_groups = parse_attribute_groups(args.attr_texts)
            print(f"Loaded attribute groups for {len(attribute_groups)} caption(s).")
            for idx, group in enumerate(attribute_groups):
                preview = ', '.join(group)
                preview = preview if len(preview) <= 100 else preview[:100] + "..."
                print(f"  Attr[{idx}]: {preview}")
        except Exception as e:
            print(f"Error parsing attribute texts: {e}")
            raise
    else:
        attribute_groups = None
        print("No attribute texts provided.")

    captions_placeholder = False
    attr_priority_indices = parse_index_list(args.attr_priority)
    # If no captions provided but attributes exist, derive simple captions from attributes
    if (captions is None or len(captions) == 0) and attribute_groups:
        captions = [' '.join([a for a in group if a]) or f'caption_{idx}' for idx, group in enumerate(attribute_groups)]
        print(f"Derived {len(captions)} caption(s) from attribute groups for inference.")
        captions_placeholder = True

    if captions and attribute_groups and len(attribute_groups) != len(captions):
        print(f"Note: {len(attribute_groups)} attribute group(s) provided for {len(captions)} caption(s). "
              f"Extra groups will be ignored; missing groups will be treated as empty.")

    # Build text inputs (captions + attributes)
    text_inputs = prepare_text_prompts(captions, attribute_groups, args.max_attrs_per_caption, device, captions_placeholder, attr_priority_indices)
    if args.score_source == 'auto':
        effective_score_source = 'attr' if text_inputs.get("captions_placeholder") else 'caption'
    else:
        effective_score_source = args.score_source
    text_inputs["score_source"] = effective_score_source

    # Determine output path
    file_path = args.input
    file_ext = os.path.splitext(file_path)[-1].lower()
    
    if args.output:
        output_path = args.output
    else:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        if file_ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            output_path = f'{base_name}_nano.jpg'
        else:
            output_path = f'{base_name}_nano.mp4'

    if file_ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        print(f"Processing image: {file_path}")
        process_image(model, device, file_path, text_inputs, img_size, 
                     vit_backbone, output_path, args.score_threshold)
        print("Image processing complete.")
    else:
        print(f"Processing video: {file_path}")
        process_video(model, device, file_path, text_inputs, img_size, 
                     vit_backbone, output_path, args.score_threshold)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='DEIMv2 Inference with Nano Banana Style Visualization')
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to config file')
    parser.add_argument('-r', '--resume', type=str, required=True, help='Path to checkpoint file')
    parser.add_argument('-i', '--input', type=str, required=True, help='Path to input image or video')
    parser.add_argument('-d', '--device', type=str, default='cuda', help='Device (cpu or cuda:0)')
    parser.add_argument('-t', '--class-texts', type=str, default=None, help='Class text descriptions')
    parser.add_argument('-o', '--output', type=str, default=None, help='Output file path')
    parser.add_argument('-s', '--score-threshold', type=float, default=0.4, help='Detection score threshold')
    parser.add_argument('-a', '--attr-texts', type=str, default=None, help='Attribute text groups (aligned to captions)')
    parser.add_argument('--max-attrs-per-caption', type=int, default=10, help='Max attributes to use per caption')
    parser.add_argument('--score-source', type=str, default='caption', choices=['caption', 'attr', 'avg', 'auto'],
                        help='Score source for ranking boxes: caption (default), attr, avg, or auto (use attr when captions are placeholders)')
    parser.add_argument('--attr-priority', type=str, default=None,
                        help='Comma/pipe/JSON list of caption indices that should use attribute scores (0-based)')
    
    args = parser.parse_args()
    main(args)
