"""
Benchmark FPS for the OTA-Det deployment model using the dataset defined in a YAML config.
The script reuses the deploy path from `torch_inf_OTA_det.py` and iterates the eval
DataLoader to time model forward + postprocess. Data loading and host-to-device transfer
are intentionally excluded from the timed region.
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont # 用于自定义可视化绘制

# Make project modules importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from engine.core import YAMLConfig


def aggregate_boxes(labels: torch.Tensor, boxes: torch.Tensor, scores: torch.Tensor):
    """
    Aggregate boxes with identical coordinates and sort their labels by score.
    This mirrors the logic in torch_inf_OTA_det.py but avoids a pandas dependency.
    """
    if len(boxes) == 0:
        return [], boxes, []

    labels_cpu = labels.detach().cpu().tolist()
    boxes_cpu = boxes.detach().cpu().tolist()
    scores_cpu = scores.detach().cpu().tolist()

    merged = {}
    for lbl, box, score in zip(labels_cpu, boxes_cpu, scores_cpu):
        key = tuple(box)
        merged.setdefault(key, []).append((lbl, score))

    aggregated_labels: List[List[int]] = []
    aggregated_boxes: List[torch.Tensor] = []
    aggregated_scores: List[List[float]] = []

    for key, pairs in merged.items():
        pairs.sort(key=lambda x: x[1], reverse=True)
        aggregated_labels.append([p[0] for p in pairs])
        aggregated_scores.append([p[1] for p in pairs])
        aggregated_boxes.append(torch.tensor(key, device=boxes.device))

    aggregated_boxes_tensor = torch.stack(aggregated_boxes) if aggregated_boxes else boxes[:0]
    return aggregated_labels, aggregated_boxes_tensor, aggregated_scores


def get_color_palette(num_colors: int = 80):
    """Generate distinct colors for different classes."""
    colors = []
    for i in range(num_colors):
        hue = i / num_colors
        h = hue * 360
        s = 0.8
        v = 0.95

        c = v * s
        x = c * (1 - abs((h / 60) % 2 - 1))
        m = v - c

        if h < 60:
            r, g, b = c, x, 0
        elif h < 120:
            r, g, b = x, c, 0
        elif h < 180:
            r, g, b = 0, c, x
        elif h < 240:
            r, g, b = 0, x, c
        elif h < 300:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x

        colors.append((int((r + m) * 255), int((g + m) * 255), int((b + m) * 255)))
    return colors


COLOR_PALETTE = get_color_palette()


class DeployModel(nn.Module):
    """Thin wrapper to switch the training model/postprocessor into deploy mode."""

    def __init__(self, cfg: YAMLConfig, aggregate: bool = True):
        super().__init__()
        self.model = cfg.model.deploy()
        self.postprocessor = cfg.postprocessor.deploy()
        self.aggregate = aggregate

    def forward(self, images: torch.Tensor, orig_target_sizes: torch.Tensor, captions_batch=None):
        outputs = self.model(images, captions_batch=captions_batch)
        labels, boxes, scores = self.postprocessor(outputs, orig_target_sizes)

        if not self.aggregate:
            return labels, boxes, scores

        batch_labels, batch_boxes, batch_scores = [], [], []
        for b in range(labels.shape[0]):
            agg_labels, agg_boxes, agg_scores = aggregate_boxes(labels[b], boxes[b], scores[b])
            batch_labels.append(agg_labels)
            batch_boxes.append(agg_boxes)
            batch_scores.append(agg_scores)
        return batch_labels, batch_boxes, batch_scores


def build_dataloader(cfg: YAMLConfig, batch_size: Optional[int], num_workers: Optional[int]):
    """Override dataloader params from CLI while keeping the config defaults."""
    val_cfg = cfg.yaml_cfg.get("val_dataloader", {})
    if batch_size is not None:
        # Avoid conflict between total_batch_size and batch_size
        val_cfg.pop("total_batch_size", None)
        val_cfg["batch_size"] = batch_size
    if num_workers is not None:
        val_cfg["num_workers"] = num_workers
        if num_workers == 0:
            val_cfg["persistent_workers"] = False
    cfg.yaml_cfg["val_dataloader"] = val_cfg
    return cfg.val_dataloader


def load_model(
    cfg: YAMLConfig,
    checkpoint_path: str,
    device: torch.device,
    aggregate: bool,
    text_cache_mode: str = "off",
    text_cache_path: Optional[str] = None,
    text_cache_max: Optional[int] = None,
):
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "ema" in checkpoint:
        state = checkpoint["ema"]["module"]
    else:
        state = checkpoint["model"]
    cfg.model.load_state_dict(state)

    model = DeployModel(cfg, aggregate=aggregate).to(device)
    model.eval()
    backbone = getattr(model.model, "backbone", None)
    if backbone is not None and hasattr(backbone, "set_text_cache"):
        backbone.set_text_cache(text_cache_path, mode=text_cache_mode, max_entries=text_cache_max)
    return model


def parse_override_texts(raw: Optional[str]) -> Optional[List[str]]:
    if raw is None:
        return None
    texts = [item.strip() for item in raw.replace("|||", ",").split(",") if item.strip()]
    return texts or None


def extract_captions(targets: List[dict]) -> List[Optional[List[str]]]:
    """Keep only valid captions to avoid padding overhead."""
    captions_batch: List[Optional[List[str]]] = []
    for target in targets:
        captions = target.get("captions", None)
        num_valid = target.get("num_captions", None)
        if captions is None:
            captions_batch.append(None)
            continue
        if num_valid is not None:
            captions = captions[:num_valid]
        captions_batch.append(captions)
    return captions_batch


def denormalize(img: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
    """Undo Normalize to [0, 1] range for visualization."""
    return (img * std[:, None, None] + mean[:, None, None]).clamp(0.0, 1.0)


def get_font(size: int = 14) -> Optional[ImageFont.ImageFont]:
    """尝试加载一个可用的字体文件，否则返回 None。"""
    try:
        # 尝试加载一个常用的字体
        return ImageFont.truetype("arial.ttf", size)
    except IOError:
        try:
            # 尝试加载另一个常见的字体
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
        except IOError:
            return ImageFont.load_default() # 使用 PIL 默认字体


def build_vis_entries(labels, boxes, scores, score_thr: float, aggregate: bool, captions=None):
    """
    润色后的构建可视化条目函数。只保留类别文本和边界框，并移除分数信息以实现简洁标签。
    """
    def label_to_text(lbl: int):
        if captions and len(captions) > lbl:
            name = captions[lbl]
            if isinstance(name, list):
                name = name[0] if len(name) > 0 else str(lbl)
            return str(name)
        return str(lbl)

    kept_boxes = []
    kept_texts = [] 

    for idx, box in enumerate(boxes):
        if aggregate:
            raw_labels = labels[idx]
            raw_scores = scores[idx]
            # 找到最高分且高于阈值的检测结果
            valid = [(l, s) for l, s in zip(raw_labels, raw_scores) if s >= score_thr]
            if not valid:
                continue
            lbl, _ = valid[0]
            # 简化文本：只显示最高分的类别名称，模仿图片中的简洁样式
            text = label_to_text(lbl)
        else:
            scr = scores[idx]
            if scr < score_thr:
                continue
            lbl = labels[idx]
            # 简化文本：只显示类别名称
            text = label_to_text(lbl)

        kept_boxes.append(box)
        kept_texts.append(text)

    if not kept_boxes:
        # 返回 None 和两个空列表
        return None, [], []
        
    return torch.stack(kept_boxes).to(torch.float32), kept_texts, kept_texts # 最后一个 kept_texts 是一个占位符


def visualize_predictions(
    model: nn.Module,
    dataloader,
    device: torch.device,
    aggregate: bool,
    out_dir: str,
    score_thr: float,
    max_batches: int,
    mean: List[float],
    std: List[float],
    use_text: bool,
):
    """
    润色后的可视化函数：使用 PIL 绘制带白色背景的标签和红色的边界框。
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    mean_tensor = torch.tensor(mean, dtype=torch.float32)
    std_tensor = torch.tensor(std, dtype=torch.float32)

    iterator = iter(dataloader)
    saved = 0
    font = get_font(size=14) # 获取字体

    dataset = getattr(dataloader, "dataset", None)

    def resolve_image_path(target: dict):
        fname = target.get("filename") if isinstance(target, dict) else None
        if not fname:
            return None
        candidates = []
        if dataset is not None and hasattr(dataset, "root"):
            candidates.append(Path(dataset.root) / fname)
        if dataset is not None and hasattr(dataset, "datasets"):
            for ds in dataset.datasets:
                root = getattr(ds, "root", None)
                if root:
                    candidates.append(Path(root) / fname)
        candidates.append(Path(fname))
        for p in candidates:
            if p.exists():
                return p
        return None

    with torch.no_grad():
        for batch_idx in range(max_batches):
            try:
                images, targets = next(iterator)
            except StopIteration:
                if batch_idx == 0:
                    print("No batches available for visualization.")
                break

            images = images.to(device, non_blocking=True)
            orig_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
            captions_batch = extract_captions(targets) if use_text else None

            outputs = model(images, orig_sizes.to(device), captions_batch=captions_batch)
            labels_batch, boxes_batch, scores_batch = outputs

            if aggregate:
                labels_list = labels_batch
                boxes_list = [b.detach().cpu() for b in boxes_batch] 
                scores_list = scores_batch
            else:
                labels_list = [l.detach().cpu().tolist() for l in labels_batch]
                boxes_list = [b.detach().cpu() for b in boxes_batch]
                scores_list = [s.detach().cpu().tolist() for s in scores_batch]


            for i in range(len(images)):
                labels = labels_list[i]
                scores = scores_list[i]
                captions = targets[i].get("captions") if isinstance(targets[i], dict) else None

                boxes = boxes_list[i]
                # 获取简化后的可视化数据
                boxes_tensor, texts, _ = build_vis_entries(labels, boxes, scores, score_thr, aggregate, captions=captions)

                image_path = resolve_image_path(targets[i])

                # 尝试加载原始图片
                if image_path is not None:
                    try:
                        raw_img_pil = Image.open(image_path).convert("RGB")
                    except Exception:
                        img_tensor = denormalize(images[i].detach().cpu(), mean_tensor, std_tensor).clamp(0.0, 1.0)
                        raw_img_pil = TF.to_pil_image(img_tensor)
                else:
                    img_tensor = denormalize(images[i].detach().cpu(), mean_tensor, std_tensor).clamp(0.0, 1.0)
                    raw_img_pil = TF.to_pil_image(img_tensor)


                # 使用 PIL 进行绘制
                draw = ImageDraw.Draw(raw_img_pil)

                if boxes_tensor is not None and boxes_tensor.numel() > 0:
                    for box, text in zip(boxes_tensor.tolist(), texts):
                        x_min, y_min, x_max, y_max = box
                        
                        # 1. 绘制红色的边界框 (线宽设置为 2)
                        draw.rectangle([(x_min, y_min), (x_max, y_max)], outline="red", width=2)
                        
                        # 2. 绘制白色背景的标签气泡
                        if font:
                            # 获取文本框尺寸 (W, H)
                            try:
                                # (left, top, right, bottom)
                                text_bbox = draw.textbbox((0, 0), text, font=font) 
                                bbox_w = text_bbox[2] - text_bbox[0]
                                bbox_h = text_bbox[3] - text_bbox[1]
                            except AttributeError:
                                # 旧版本 PIL 使用 textsize
                                bbox_w, bbox_h = draw.textsize(text, font=font)
                        else:
                            # 使用默认字体估算尺寸
                            bbox_w = len(text) * 8 
                            bbox_h = 16 
                            
                        padding = 5 # 标签内边距
                        
                        label_x = x_min
                        label_y = y_min - bbox_h - 2 * padding 
                        
                        # 确保标签不超出图片顶部
                        if label_y < 0: 
                            label_y = y_min + 2 * padding # 如果超出，则放在框内顶部下方
                            if label_y + bbox_h + padding > y_max:
                                label_y = 0

                        # 绘制白色背景矩形
                        draw.rectangle(
                            [
                                (label_x, label_y), 
                                (label_x + bbox_w + 2 * padding, label_y + bbox_h + 2 * padding)
                            ], 
                            fill="white"
                        )
                        
                        # 绘制黑色文本（加上内边距）
                        draw.text(
                            (label_x + padding, label_y + padding), 
                            text, 
                            font=font, 
                            fill="black"
                        )
                        
                save_path = out_path / f"batch{batch_idx}_img{i}.jpg"
                raw_img_pil.save(save_path)
                saved += 1

    if saved > 0:
        print(f"Saved {saved} visualization image(s) to {out_path}")


def measure_fps(
    model: nn.Module,
    dataloader,
    device: torch.device,
    warmup_iters: int,
    max_iters: Optional[int],
    aggregate: bool,
    use_text: bool,
    override_texts: Optional[List[str]] = None,
):
    times: List[float] = []
    total_images = 0

    with torch.no_grad():
        for step, (images, targets) in enumerate(dataloader):
            # 核心逻辑：如果达到 max_iters + warmup_iters，则停止
            if max_iters is not None and step >= warmup_iters + max_iters:
                break

            images = images.to(device, non_blocking=True)
            orig_sizes = torch.stack([t["orig_size"] for t in targets], dim=0).to(device)
            captions_batch = extract_captions(targets) if use_text else None
            if captions_batch is not None and override_texts is not None:
                captions_batch = [override_texts for _ in targets]

            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.time()
            _ = model(images, orig_sizes, captions_batch=captions_batch)
            if device.type == "cuda":
                torch.cuda.synchronize()
            iter_time = time.time() - start

            if step >= warmup_iters:
                times.append(iter_time)
                total_images += images.shape[0]

            if (step + 1) % 10 == 0 or step < warmup_iters:
                print(
                    f"[iter {step + 1}] "
                    f"{'warmup' if step < warmup_iters else 'measure'} "
                    f"time: {iter_time * 1000:.2f} ms"
                )

    if not times:
        raise RuntimeError("No iterations were measured; check warmup/max-iters settings.")

    total_time = sum(times)
    avg_latency = (total_time / len(times)) * 1000.0
    fps = total_images / total_time
    print("\n=== FPS Report ===")
    print(f"Measured batches : {len(times)} (warmup {warmup_iters} skipped)")
    print(f"Images processed : {total_images}")
    print(f"Avg latency/batch: {avg_latency:.2f} ms")
    print(f"Throughput       : {fps:.2f} FPS")
    if aggregate:
        print("Note: timings include postprocess + box aggregation.")
    else:
        print("Note: timings include postprocess only (aggregation skipped).")


def parse_args():
    parser = argparse.ArgumentParser(description="Measure OTA-Det torch inference FPS on a dataset.")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="configs/OTA-Det/OTA-Det-M/OTADet_dinov3_m_AerialVG.yml",
        help="Path to YAML config (dataset + model).",
    )
    parser.add_argument(
        "-r", "--resume", type=str, default='', help="Checkpoint path (expects model.state_dict format)."
    )
    parser.add_argument(
        "-d", "--device", type=str, default="cuda", help="Device string, e.g. cuda:0 or cpu."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override val dataloader batch size (per rank). Defaults to config.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override num_workers for the val dataloader. Use 0 to force single-process loading.",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=10,
        help="Batches to warm up (not timed).",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=1000, # 限制测量迭代次数为 1000 轮
        help="Max measured batches after warmup. None = exhaust the dataloader.",
    )
    parser.add_argument(
        "--skip-aggregation",
        action="store_true",
        help="Skip box aggregation to measure pure model + postprocess speed.",
    )
    parser.add_argument(
        "--disable-text",
        action="store_true",
        help="Do not pass captions_batch to the model (measure FPS without text branch).",
    )
    parser.add_argument(
        "--disable-cudnn-benchmark",
        action="store_true",
        help="Disable cudnn benchmark (enabled by default for stable input size).",
    )
    parser.add_argument(
        "--visualize-batches",
        type=int,
        default=0,
        help="Save predictions for the first N batches after FPS test (0 disables visualization).",
    )
    parser.add_argument(
        "--vis-dir",
        type=str,
        default="./outputs/fps_vis",
        help="Directory to save visualization results.",
    )
    parser.add_argument(
        "--vis-threshold",
        type=float,
        default=0.45,
        help="Score threshold for drawing boxes when visualization is enabled.",
    )
    parser.add_argument(
        "--vis-mean",
        type=float,
        nargs=3,
        default=[0.485, 0.456, 0.406],
        help="Normalization mean used to de-normalize images for visualization.",
    )
    parser.add_argument(
        "--vis-std",
        type=float,
        nargs=3,
        default=[0.229, 0.224, 0.225],
        help="Normalization std used to de-normalize images for visualization.",
    )
    parser.add_argument(
        "--text-cache-mode",
        type=str,
        choices=["off", "record", "reuse"],
        default="off",
        help="Text feature cache behavior: off | record | reuse.",
    )
    parser.add_argument(
        "--text-cache-path",
        type=str,
        default="./outputs/text_cache.pt",
        help="Path to save/load cached text features.",
    )
    parser.add_argument(
        "--text-cache-max",
        type=int,
        default=100,
        help="Max unique caption batches to cache when mode=record.",
    )
    parser.add_argument(
        "--override-texts",
        type=str,
        default=None,
        help="Override every sample's captions with fixed text prompts, e.g. 'vehicle' or 'car,ship'.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.disable_cudnn_benchmark:
        torch.backends.cudnn.benchmark = True

    device = torch.device(args.device)
    cfg = YAMLConfig(args.config, resume=args.resume)

    dataloader = build_dataloader(cfg, args.batch_size, args.num_workers)
    model = load_model(
        cfg,
        args.resume,
        device,
        aggregate=not args.skip_aggregation,
        text_cache_mode=args.text_cache_mode,
        text_cache_path=args.text_cache_path,
        text_cache_max=args.text_cache_max,
    )
    use_text = not args.disable_text
    override_texts = parse_override_texts(args.override_texts)

    print(
        f"Running FPS test on device={device} | batches={len(dataloader)} "
        f"| batch_size={dataloader.batch_size} | "
        f"text={'on' if use_text else 'off'} | aggregation={'off' if args.skip_aggregation else 'on'} | "
        f"override_texts={len(override_texts) if override_texts else 'off'}"
    )
    measure_fps(
        model,
        dataloader,
        device,
        warmup_iters=args.warmup_iters,
        max_iters=args.max_iters,
        aggregate=not args.skip_aggregation,
        use_text=use_text,
        override_texts=override_texts,
    )

    if args.visualize_batches > 0:
        visualize_predictions(
            model,
            dataloader,
            device,
            aggregate=not args.skip_aggregation,
            out_dir=args.vis_dir,
            score_thr=args.vis_threshold,
            max_batches=args.visualize_batches,
            mean=args.vis_mean,
            std=args.vis_std,
            use_text=use_text,
        )

    # Persist cached text features when recording.
    backbone = getattr(model.model, "backbone", None)
    if backbone is not None and hasattr(backbone, "save_text_cache"):
        backbone.save_text_cache()


if __name__ == "__main__":
    main()
