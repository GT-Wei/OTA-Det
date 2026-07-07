#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
conver_AerialVG_to_xml.py

将相同 filename 的多条 JSONL 标注归并到一个 VOC 风格 XML 中。
每条 caption 生成一个 <grounding> 节点；第一个 region 作为 <target_object>，
其余作为 <realation_object>（沿用数据里的键名 realation）。
支持给最终保存时的文件名与 <filename> 加前缀（如 AerialVG_）。

用法：
  python conver_AerialVG_to_xml.py \
    --input ../datasets/AerialVG/annotation/vg_test_odvg.jsonl \
    --outdir ../datasets/LLM_Caption_Parse/annotations/AerialVG_test \
    --database AerialVG \
    --depth 3 \
    --prefix AerialVG_
  python conver_AerialVG_to_xml.py \
    --input ../datasets/AerialVG/annotation/vg_train_odvg.jsonl \
    --outdir ../datasets/LLM_Caption_Parse/annotations/AerialVG_train \
    --database AerialVG \
    --depth 3 \
    --prefix AerialVG_
  python conver_AerialVG_to_xml.py \
    --input ../datasets/AerialVG/annotation/vg_val_odvg.jsonl \
    --outdir ../datasets/LLM_Caption_Parse/annotations/AerialVG_val \
    --database AerialVG \
    --depth 3 \
    --prefix AerialVG_
"""

import os
import json
import argparse
from pathlib import Path
from collections import defaultdict
import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import List, Dict, Any
from tqdm import tqdm

def prettify(elem: ET.Element) -> str:
    rough = ET.tostring(elem, encoding='utf-8')
    return minidom.parseString(rough).toprettyxml(indent="\t", encoding="utf-8").decode("utf-8")

def add_text_child(parent: ET.Element, tag: str, text: Any) -> ET.Element:
    node = ET.SubElement(parent, tag)
    node.text = "" if text is None else str(text)
    return node

def region_to_xml(parent: ET.Element, tag: str, region: Dict[str, Any]) -> None:
    node = ET.SubElement(parent, tag)
    add_text_child(node, "phrase", region.get("phrase", ""))
    if "realation" in region and region["realation"] is not None:
        add_text_child(node, "realation", region["realation"])
    bbox = region.get("bbox", [])
    if not (isinstance(bbox, list) and len(bbox) == 4):
        bbox = [0, 0, 0, 0]
    bnd = ET.SubElement(node, "bndbox")
    add_text_child(bnd, "xmin", bbox[0])
    add_text_child(bnd, "ymin", bbox[1])
    add_text_child(bnd, "xmax", bbox[2])
    add_text_child(bnd, "ymax", bbox[3])

def grounding_block(parent: ET.Element, caption: str, regions: List[Dict[str, Any]]) -> None:
    g = ET.SubElement(parent, "grounding")
    add_text_child(g, "caption", caption if caption is not None else "")
    if not regions:
        return
    region_to_xml(g, "target_object", regions[0])
    for r in regions[1:]:
        region_to_xml(g, "realation_object", r)

def build_root_xml(filename_out: str, width: int, height: int, database: str, depth: int) -> ET.Element:
    root = ET.Element("annotation")
    add_text_child(root, "filename", filename_out)
    src = ET.SubElement(root, "source")
    add_text_child(src, "database", database)
    size = ET.SubElement(root, "size")
    add_text_child(size, "width", width)
    add_text_child(size, "height", height)
    add_text_child(size, "depth", depth)
    add_text_child(root, "segmented", 0)
    return root

def convert_with_aggregation(input_jsonl: Path, outdir: Path, database: str, depth: int, prefix: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    with input_jsonl.open("r", encoding="utf-8", errors="ignore") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] 第 {ln} 行 JSON 解析失败：{e}")
                continue
            fname = rec.get("filename")
            if not fname:
                print(f"[WARN] 第 {ln} 行缺少 filename，已跳过")
                continue
            buckets[fname].append(rec)

    for orig_fn, recs in tqdm(buckets.items()):
        first = recs[0]
        width = int(first.get("width", 0) or 0)
        height = int(first.get("height", 0) or 0)

        # 仅在“输出”时加前缀：既用于 <filename>，也用于 .xml 文件名
        base = os.path.basename(orig_fn)
        prefixed_base = f"{prefix}{base}" if prefix else base
        filename_out = prefixed_base  # XML <filename>

        root = build_root_xml(filename_out, width, height, database, depth)

        for rec in recs:
            g = rec.get("grounding", {}) or {}
            caption = g.get("caption", "")
            regions = g.get("regions", []) or []
            grounding_block(root, caption, regions)

        xml_stem = os.path.splitext(prefixed_base)[0]  # AerialVG_xxx
        xml_name = f"{xml_stem}.xml"
        xml_path = outdir / xml_name
        with xml_path.open("w", encoding="utf-8") as xf:
            xf.write(prettify(root))

    print(f"✅ 总计 {len(buckets)} 张图片已生成 XML 到：{outdir}")

def parse_args():
    ap = argparse.ArgumentParser(description="将包含多条 caption 的 JSONL 归并为单 XML（按 filename 分组），并为输出文件名加前缀")
    ap.add_argument("--input", required=True, type=Path, help="输入 .jsonl 文件路径")
    ap.add_argument("--outdir", required=True, type=Path, help="输出 XML 目录")
    ap.add_argument("--database", default="DIOR", type=str, help="XML <database> 字段")
    ap.add_argument("--depth", default=3, type=int, help="图像通道数，写入 <depth>")
    ap.add_argument("--prefix", default="AerialVG_", type=str, help="给输出文件名与 <filename> 加的前缀，留空则不加")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    convert_with_aggregation(args.input, args.outdir, args.database, args.depth, args.prefix)
