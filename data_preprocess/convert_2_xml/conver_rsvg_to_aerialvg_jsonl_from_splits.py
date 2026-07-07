#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_dior_rsvg_jsonl_from_splits.py
# 先转换为grounding格式
用法示例：
python conver_rsvg_to_aerialvg_jsonl_from_splits.py \
  --anno_dir ../datasets/RSVG-OPT/Annotations \
  --split_dir ../datasets/RSVG-OPT/split \
  --out_dir  ../datasets/LLM_Caption_Parse/OPT_RSVG \
  --prefix OPT_RSVG_ \
  --train_name OPT_RSVG_train.jsonl \
  --val_name   OPT_RSVG_val.jsonl \
  --test_name  OPT_RSVG_test.jsonl
"""

import os
import json
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

def filelist(root: Path, suffix: str = ".xml") -> list[Path]:
    files = [p for p in root.rglob(f"*{suffix}") if p.is_file()]
    files.sort()  # 保持与官方 loader 一致的稳定顺序
    return files

def read_index_set(txt_path: Path) -> set[int]:
    s = set()
    if not txt_path.is_file():
        return s
    with txt_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            t = line.strip()
            if not t:
                continue
            try:
                s.add(int(t))
            except ValueError:
                # 允许出现 "00003" 这类零填充字符串
                try:
                    s.add(int(t.lstrip("0") or "0"))
                except Exception:
                    pass
    return s

def geti(elem: ET.Element, name: str, default: int = 0) -> int:
    x = elem.findtext(name)
    try:
        return int(x)
    except Exception:
        return default

def text(elem: ET.Element, default: str = "") -> str:
    return elem.text.strip() if (elem is not None and elem.text) else default

def main():
    ap = argparse.ArgumentParser(description="DIOR-RSVG → AerialVG JSONL 按 split 导出（0-based 全局 object 索引）")
    ap.add_argument("--anno_dir", required=True, type=Path, help="VOC 注释目录（含 <object>/<description>）")
    ap.add_argument("--split_dir", required=True, type=Path, help="包含 train.txt/val.txt/test.txt 的目录")
    ap.add_argument("--out_dir",  required=True, type=Path, help="输出目录（写出 train/val/test.jsonl）")

    ap.add_argument("--prefix", default="", type=str, help="可选：给 JSONL 的 filename 加前缀（如 DIOR_RSVG_）")
    ap.add_argument("--train_name", default="train.jsonl", type=str)
    ap.add_argument("--val_name",   default="val.jsonl",   type=str)
    ap.add_argument("--test_name",  default="test.jsonl",  type=str)
    args = ap.parse_args()

    # 读取 split 索引集合（0-based）
    train_idx = read_index_set(args.split_dir / "train.txt")
    val_idx   = read_index_set(args.split_dir / "val.txt")
    test_idx  = read_index_set(args.split_dir / "test.txt")

    # 打开输出文件（流式写入）
    args.out_dir.mkdir(parents=True, exist_ok=True)
    f_train = (args.out_dir / args.train_name).open("w", encoding="utf-8")
    f_val   = (args.out_dir / args.val_name).open("w", encoding="utf-8")
    f_test  = (args.out_dir / args.test_name).open("w", encoding="utf-8")

    annos = filelist(args.anno_dir, ".xml")
    iterator = tqdm(annos, desc="遍历注释文件", unit="份", dynamic_ncols=True) if tqdm else annos

    count = 0  # 全局 object 计数器，严格 0-based（与官方 data_loader 一致）
    wrote_train = wrote_val = wrote_test = 0

    for apath in iterator:
        try:
            root = ET.parse(apath).getroot()
        except Exception as e:
            print(f"[WARN] 解析失败 {apath}: {e}")
            continue

        fname = text(root.find("filename"), "")
        if args.prefix and fname:
            fname_out = f"{args.prefix}{os.path.basename(fname)}"
        else:
            fname_out = fname

        size = root.find("size")
        width  = geti(size, "width",  0) if size is not None else 0
        height = geti(size, "height", 0) if size is not None else 0
        # depth 可不需要；AerialVG 例子里只要 width/height

        # 先尝试 VOC 结构
        voc_objs = root.findall("object")
        if voc_objs:
            for obj in voc_objs:
                # 取 bbox 与 description、name
                b = obj.find("bndbox")
                if b is None:
                    count += 1
                    continue
                xmin = geti(b, "xmin", 0); ymin = geti(b, "ymin", 0)
                xmax = geti(b, "xmax", 0); ymax = geti(b, "ymax", 0)
                cls  = text(obj.find("name"), "object")
                desc = text(obj.find("description"), "").strip()
                if not desc:
                    desc = f"a {cls}"

                rec = {
                    "filename": fname_out,
                    "height": height,
                    "width":  width,
                    "grounding": {
                        "caption": desc,
                        "regions": [
                            {
                                "bbox": [xmin, ymin, xmax, ymax],
                                "phrase": cls
                            }
                        ]
                    }
                }

                line = json.dumps(rec, ensure_ascii=False)
                # 根据全局索引写入对应 split
                if count in train_idx:
                    f_train.write(line + "\n"); wrote_train += 1
                if count in val_idx:
                    f_val.write(line + "\n"); wrote_val += 1
                if count in test_idx:
                    f_test.write(line + "\n"); wrote_test += 1

                count += 1
            continue

        # 再兼容 Grounding 结构（若你的注释已经转成了 <grounding>）
        for g in root.findall("grounding"):
            cap = text(g.find("caption"), "").strip()
            tgt = g.find("target_object")
            bnd = tgt.find("bndbox") if tgt is not None else None
            phr = text(tgt.find("phrase"), "object") if tgt is not None else "object"
            if bnd is None:
                count += 1
                continue
            xmin = geti(bnd, "xmin", 0); ymin = geti(bnd, "ymin", 0)
            xmax = geti(bnd, "xmax", 0); ymax = geti(bnd, "ymax", 0)
            if not cap:
                cap = f"a {phr}"

            rec = {
                "filename": fname_out,
                "height": height,
                "width":  width,
                "grounding": {
                    "caption": cap,
                    "regions": [
                        {
                            "bbox": [xmin, ymin, xmax, ymax],
                            "phrase": phr
                        }
                    ]
                }
            }

            line = json.dumps(rec, ensure_ascii=False)
            if count in train_idx:
                f_train.write(line + "\n"); wrote_train += 1
            if count in val_idx:
                f_val.write(line + "\n"); wrote_val += 1
            if count in test_idx:
                f_test.write(line + "\n"); wrote_test += 1

            count += 1

    f_train.close(); f_val.close(); f_test.close()
    print(f"✅ 完成：train {wrote_train} 条，val {wrote_val} 条，test {wrote_test} 条。总枚举 object 数：{count}")
    print(f"输出：\n  {args.out_dir / args.train_name}\n  {args.out_dir / args.val_name}\n  {args.out_dir / args.test_name}")

if __name__ == "__main__":
    main()
