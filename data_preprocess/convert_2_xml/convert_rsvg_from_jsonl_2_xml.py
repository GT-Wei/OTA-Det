#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
convert_rsvg_from_jsonl_2_xml.py

从 AerialVG 风格 JSONL 归并为 Grounding XML（无 relation_object）。
严格校验：每条 JSON 的 regions 必须且只能有 1 个。
违规处理策略 --on-violation: error(默认) | drop | split | first

用法：
python convert_rsvg_from_jsonl_2_xml.py \
  --input ../datasets/LLM_Caption_Parse/annotations/JSONL/OPT_RSVG_JSONL/OPT_RSVG_train.jsonl \
  --outdir ../datasets/LLM_Caption_Parse/annotations/OPT_RSVG_train \
  --database OPT_RSVG \
  --depth 3 \
  --prefix '' \
  --on-violation error
python convert_rsvg_from_jsonl_2_xml.py \
  --input ../datasets/LLM_Caption_Parse/annotations/JSONL/OPT_RSVG_JSONL/OPT_RSVG_val.jsonl \
  --outdir ../datasets/LLM_Caption_Parse/annotations/OPT_RSVG_val \
  --database OPT_RSVG \
  --depth 3 \
  --prefix '' \
  --on-violation error
python convert_rsvg_from_jsonl_2_xml.py \
  --input ../datasets/LLM_Caption_Parse/annotations/JSONL/OPT_RSVG_JSONL/OPT_RSVG_test.jsonl \
  --outdir ../datasets/LLM_Caption_Parse/annotations/OPT_RSVG_test \
  --database OPT_RSVG \
  --depth 3 \
  --prefix '' \
  --on-violation error

python convert_rsvg_from_jsonl_2_xml.py \
  --input ../datasets/LLM_Caption_Parse/annotations/JSONL/DIOR_RSVG_JSONL/DIOR-RSVG_test.jsonl \
  --outdir ../datasets/LLM_Caption_Parse/annotations/DIOR_RSVG_test \
  --database DIOR_RSVG \
  --depth 3 \
  --prefix '' \
  --on-violation error
python convert_rsvg_from_jsonl_2_xml.py \
  --input ../datasets/LLM_Caption_Parse/annotations/JSONL/DIOR_RSVG_JSONL/DIOR-RSVG_train.jsonl \
  --outdir ../datasets/LLM_Caption_Parse/annotations/DIOR_RSVG_train \
  --database DIOR_RSVG \
  --depth 3 \
  --prefix '' \
  --on-violation error
python convert_rsvg_from_jsonl_2_xml.py \
  --input ../datasets/LLM_Caption_Parse/annotations/JSONL/DIOR_RSVG_JSONL/DIOR-RSVG_val.jsonl \
  --outdir ../datasets/LLM_Caption_Parse/annotations/DIOR_RSVG_val \
  --database DIOR_RSVG \
  --depth 3 \
  --prefix '' \
  --on-violation error
"""

import os
import json
import argparse
from pathlib import Path
from collections import defaultdict
import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import Any, Dict, List, Tuple

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def prettify(elem: ET.Element) -> str:
    rough = ET.tostring(elem, encoding="utf-8")
    return minidom.parseString(rough).toprettyxml(indent="\t", encoding="utf-8").decode("utf-8")


def add_text(parent: ET.Element, tag: str, text: Any) -> ET.Element:
    node = ET.SubElement(parent, tag)
    node.text = "" if text is None else str(text)
    return node


def to_bbox_list(b):
    if isinstance(b, list) and len(b) == 4:
        return b
    return [0, 0, 0, 0]


def region_to_target(parent: ET.Element, region: Dict[str, Any]) -> None:
    tgt = ET.SubElement(parent, "target_object")
    add_text(tgt, "phrase", region.get("phrase", ""))  # 类别名
    xmin, ymin, xmax, ymax = to_bbox_list(region.get("bbox", []))
    bnd = ET.SubElement(tgt, "bndbox")
    add_text(bnd, "xmin", xmin); add_text(bnd, "ymin", ymin)
    add_text(bnd, "xmax", xmax); add_text(bnd, "ymax", ymax)


def grounding_block(parent: ET.Element, caption: str, region: Dict[str, Any]) -> None:
    g = ET.SubElement(parent, "grounding")
    add_text(g, "caption", caption if caption is not None else "")
    region_to_target(g, region)


def build_root(filename_out: str, width: int, height: int, depth: int, database: str) -> ET.Element:
    root = ET.Element("annotation")
    add_text(root, "filename", filename_out)
    src = ET.SubElement(root, "source"); add_text(src, "database", database)
    size = ET.SubElement(root, "size")
    add_text(size, "width", width); add_text(size, "height", height); add_text(size, "depth", depth)
    add_text(root, "segmented", 0)
    return root


def handle_regions(rec: Dict[str, Any], policy: str) -> List[Tuple[str, Dict[str, Any]]]:
    """返回 [(caption, region), ...]；按 policy 处理 regions != 1 的情况。"""
    g = rec.get("grounding", {}) or {}
    caption = g.get("caption", "")
    regions = g.get("regions", []) or []

    if len(regions) == 1:
        return [(caption, regions[0])]

    # 违规
    if policy == "error":
        raise ValueError(f"regions count != 1 (got {len(regions)}); filename={rec.get('filename')}")
    if policy == "drop":
        return []
    if policy == "first":
        return [(caption, regions[0])] if regions else []
    if policy == "split":
        return [(caption, r) for r in regions]
    raise ValueError(f"Unknown policy: {policy}")


def convert_jsonl(
    input_jsonl: Path,
    outdir: Path,
    database: str,
    depth: int,
    prefix: str,
    policy: str,
    violations_log: Path
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    buckets = defaultdict(list)  # filename -> [records]
    total_lines = 0
    violations = 0

    # 可选：统计总行数以显示百分比（不想多读一次可删掉）
    total_est = None
    try:
        with input_jsonl.open("r", encoding="utf-8", errors="ignore") as f:
            total_est = sum(1 for _ in f)
    except Exception:
        pass

    it = tqdm(total=total_est, desc="读取JSONL并分组", unit="行", dynamic_ncols=True) if tqdm else None
    log_fp = violations_log.open("w", encoding="utf-8")

    with input_jsonl.open("r", encoding="utf-8", errors="ignore") as f:
        for ln, line in enumerate(f, start=1):
            if it is not None:
                it.update(1)
            s = line.strip()
            if not s:
                continue
            total_lines += 1
            try:
                rec = json.loads(s)
            except json.JSONDecodeError as e:
                violations += 1
                log_fp.write(f"[JSON-ERROR] line={ln}: {e}\n")
                if policy == "error":
                    if it is not None: it.close()
                    log_fp.close()
                    raise
                continue

            regions = ((rec.get("grounding") or {}).get("regions") or [])
            if len(regions) != 1:
                violations += 1
                log_fp.write(f"[VIOLATION] line={ln} filename={rec.get('filename')} regions={len(regions)}\n")

            fname = rec.get("filename")
            if not fname:
                violations += 1
                log_fp.write(f"[VIOLATION] line={ln} missing filename\n")
                if policy == "error":
                    if it is not None: it.close()
                    log_fp.close()
                    raise ValueError(f"Missing filename at line {ln}")
                continue
            buckets[fname].append(rec)

    if it is not None:
        it.close()

    if not buckets:
        # 读到了文件但没有任何有效样本，给出预览帮助排查
        preview = ""
        try:
            with input_jsonl.open("r", encoding="utf-8", errors="ignore") as f:
                from itertools import islice
                preview = "".join(list(islice(f, 3)))
        except Exception:
            pass
        log_fp.close()
        raise RuntimeError(f"No valid records parsed from: {input_jsonl}\n"
                           f"Preview first lines:\n{preview}")

    # 写出 XML
    items = list(buckets.items())
    it2 = tqdm(items, desc="写出XML", unit="图", dynamic_ncols=True) if tqdm else items
    out_count = 0

    for orig_fn, recs in it2:
        first = recs[0]
        width = int(first.get("width", 0) or 0)
        height = int(first.get("height", 0) or 0)

        base = os.path.basename(orig_fn)
        prefixed_base = f"{prefix}{base}" if prefix else base
        filename_out = prefixed_base

        root = build_root(filename_out, width, height, depth, database)

        for rec in recs:
            try:
                pairs = handle_regions(rec, policy)  # [(caption, region), ...]
            except Exception as e:
                log_fp.write(f"[ABORT] filename={rec.get('filename')} reason={e}\n")
                log_fp.close()
                if tqdm and hasattr(it2, "close"): it2.close()
                raise
            for caption, region in pairs:
                grounding_block(root, caption, region)

        xml_name = os.path.splitext(prefixed_base)[0] + ".xml"
        (outdir / xml_name).write_text(prettify(root), encoding="utf-8")
        out_count += 1

    if tqdm and hasattr(it2, "close"):
        it2.close()
    log_fp.close()

    print(f"✅ 写出 {out_count} 个 XML 到：{outdir}")
    print(f"📊 统计：读入非空行 {total_lines}，违规样本 {violations}（详见 {violations_log}）。")
    if policy == "error" and violations > 0:
        print("⚠️ 启用严格模式（error），建议先根据 violations.log 清洗数据，或改用 --on-violation split/first/drop。")


def main():
    ap = argparse.ArgumentParser(description="AerialVG JSONL → Grounding XML（严格仅1个region）")
    ap.add_argument("--input", required=True, type=Path, help="输入 .jsonl 文件")
    ap.add_argument("--outdir", required=True, type=Path, help="输出 XML 目录")
    ap.add_argument("--database", default="DIOR_RSVG", type=str, help="<database> 字段")
    ap.add_argument("--depth", default=3, type=int, help="<depth> 通道数")
    ap.add_argument("--prefix", default=None, type=str, help="给 <filename> 与输出文件名加的前缀")
    ap.add_argument("--on-violation", choices=["error", "drop", "split", "first"], default="error",
                    help="遇到 regions!=1 时的处理策略")
    ap.add_argument("--violations-log", default="violations.log", type=Path, help="违规样本日志")
    return ap.parse_args()


if __name__ == "__main__":
    args = main()
    convert_jsonl(args.input, args.outdir, args.database, args.depth,
                  args.prefix, args.on_violation, args.violations_log)
