#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 复制图片/images, 并加数据集名称前缀

"""
python cp_images_add_prefix.py \
    --src ../datasets/AerialVG/images \
    --dst ../datasets/LLM_Caption_Parse/images/AerialVG \
    --prefix AerialVG_
    
python cp_images_add_prefix.py \
    --src ../datasets/DIOR-RSVG/images \
    --dst ../datasets/LLM_Caption_Parse/images/DIOR-RSVG \
    --prefix DIOR-RSVG_

python cp_images_add_prefix.py \
    --src ../datasets/RSVG-OPT/Image \
    --dst ../datasets/LLM_Caption_Parse/images/RSVG-OPT \
    --prefix RSVG-OPT_
"""

import os
import re
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from shutil import copy2
from tqdm import tqdm

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def should_prefix(name: str, prefix: str) -> bool:
    return not name.startswith(prefix)

def build_dest_path(src_root: Path, dst_root: Path, src_path: Path, prefix: str) -> Path:
    rel = src_path.relative_to(src_root)
    stem, ext = rel.stem, rel.suffix
    if ext.lower() not in IMG_EXTS:
        return None
    new_name = (prefix + rel.name) if should_prefix(rel.name, prefix) else rel.name
    return dst_root / rel.parent / new_name

def main():
    ap = argparse.ArgumentParser(description="拷贝图片并在文件名前加前缀")
    ap.add_argument("--src", required=True, type=Path, help="源目录")
    ap.add_argument("--dst", required=True, type=Path, help="目标目录")
    ap.add_argument("--prefix", default="AerialVG_", type=str, help="文件名前缀")
    ap.add_argument("--workers", type=int, default=8, help="并行线程数")
    ap.add_argument("--dry-run", action="store_true", help="演练模式，不实际复制")
    args = ap.parse_args()

    src_root, dst_root, prefix = args.src, args.dst, args.prefix
    dst_root.mkdir(parents=True, exist_ok=True)

    files = []
    for p in src_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            files.append(p)

    iterator = tqdm(files, desc="复制中", unit="张", dynamic_ncols=True) if tqdm else files

    def task(p: Path):
        dst_path = build_dest_path(src_root, dst_root, p, prefix)
        if dst_path is None:
            return False, p, "skip-nonimage"
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        if not args.dry_run:
            copy2(p, dst_path)
        return True, p, "ok"

    ok, skip, err = 0, 0, 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(task, p): p for p in files}
        for fut in as_completed(futures):
            if tqdm:
                iterator.update(1)
            try:
                done, p, status = fut.result()
                if status == "ok":
                    ok += 1
                elif status == "skip-nonimage":
                    skip += 1
            except Exception:
                err += 1
        if tqdm:
            iterator.close()

    print(f"完成：复制 {ok} 张，跳过非图片 {skip}，错误 {err}。输出目录：{dst_root}")

if __name__ == "__main__":
    main()
