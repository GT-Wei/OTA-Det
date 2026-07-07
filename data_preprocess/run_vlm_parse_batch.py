#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch LLM Caption Attribute Parser for Grounding XML (v2)
- 按批处理（--batch-size 可控）
- 多 GPU 并行（所有端口同时开跑）
- 每卡线程池并发请求（--per-gpu-concurrency）
- 扁平化本批所有 <grounding> 后负载均衡到各 GPU
- 本批全部完成后立即写回本批 XML 到 --out-dir（不覆盖输入目录）
- 跳过逻辑：若 <reasoning_caption_parse> 存在且不是 "Error:" 开头 -> 跳过该 grounding
- 写回使用 xml.dom.minidom 进行漂亮缩进，结构稳定
- 每批生成统计 JSON：{out_dir}/batch_stats/batch_{batch_id}.json

python run_vlm_parse_batch.py \
  --input-dir ../datasets/LLM_Caption_Parse/annotations/DIOR_RSVG_train \
  --out-dir   ../datasets/LLM_Caption_Parse/annotations/DIOR_RSVG_train_parsed_batch \
  --api-base  http://localhost \
  --start-port 18000 \
  --api-key   EMPTY \
  --model     openai/gpt-oss-20b \
  --gpus      4 \
  --batch-size 16 \
  --per-gpu-concurrency 8 \
  --delay 0.2 \
  --max-retries 5 \
  --skip-existing-out
"""

import os
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import traceback
from datetime import datetime

import xml.etree.ElementTree as ET
from xml.dom import minidom
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import torch
except Exception:
    torch = None

from openai import OpenAI

# ----------------- PROMPT（保留原版，不改） -----------------
PROMPT_SYSTEM = (
    "Reasoning: high\n"
    "You are a precise grounding-caption analyzer. Your task is to transform any given caption into a TARGET-CENTRIC, strictly factual, structured summary.\n\n"
    "OBJECTIVE\n"
    "- Identify the PRIMARY TARGET (the main grounded entity) and restate ALL information about the target using 'it/its'.\n"
    "- Keep every statement target-centric. Other objects may appear only as context for relations to the target.\n"
    "- Extract as many faithful details as the caption explicitly states; do NOT invent or infer.\n\n"
    "CORE RULES\n"
    "1) Target-centric rewriting only: describe properties/relations of the target; never give standalone descriptions of secondary objects.\n"
    "2) Convert relative mentions into explicit relations to the target (e.g., 'There is a car to its left').\n"
    "3) If uncertain, omit. No hallucination, no synonyms not grounded in the caption.\n"
    "4) Quote minimal verbatim evidence spans for each attribute in 'caption_evidence'.\n"
    "5) Use natural, concise English for 'description' with 'it/its'.\n"
    "6) Output MUST be a single valid JSON object. No markdown fences, no extra text.\n\n"
    "ASPECT VOCABULARY \n"
    "- category, color, size, shape, material, texture, number, state, part, text, brand\n"
    "- action, activity, pose, status\n"
    "- position, orientation, direction, relative_position, spatial_relation, distance\n"
    "- environment, weather, time, context, purpose\n"
    "- Or ANY other observable information\n\n"
    "Be comprehensive but accurate - extract all stated information but don't infer unstated facts."
)

OUTPUT_SCHEMA = (
    "{"
    "\"primary_target\": \"Main object/entity being grounded (string)\","
    "\"attributes\": ["
      "{"
        "\"aspect\": \"What aspect this describes (e.g., color, category, position, action, spatial relation)\","
        "\"description\": \"Target-centric statement using it/its (string)\","
        "\"caption_evidence\": [\"minimal verbatim phrase\", \"...\"],"
        "\"confidence\": 0.0"
      "}"
    "],"
    "\"analysis\": \"Brief rationale of what was extracted and how target-centric rewriting was enforced (string)\""
    "}"
)

def build_user_message(caption: str) -> str:
    return (
        f"Now analyze this caption comprehensively (Note: Include category and avoid redundant parsing of the same attribute under different types.):\n"
        f"Caption: {caption}\n\n"
        f"Output JSON with ALL extractable information:\n"
        f"{OUTPUT_SCHEMA}"
    )

# ----------------- LLM 调用（带重试） -----------------
def _normalize_base(api_base: str, port: int) -> str:
    # 输入形如 http://localhost 或 http://127.0.0.1:18000/v1
    # 输出为完整 /v1 基础地址
    if api_base.endswith("/v1"):
        # 调用者已带端口：直接返回
        return api_base
    # 未带 /v1：拼接端口
    if "://" in api_base and api_base.count(":") == 1:
        # e.g., http://localhost
        return f"{api_base}:{port}/v1"
    if "://" in api_base and api_base.count(":") >= 2:
        # 已经带了端口但没 /v1
        return f"{api_base}/v1"
    # 兜底
    return f"http://localhost:{port}/v1"

def call_llm_parser(api_base_with_port_v1: str, api_key: str, model: str, caption_text: str,
                    temperature=0.15, top_p=0.9, max_tokens=8192, retries=5, delay=0.5) -> Dict[str, Any]:
    client = OpenAI(api_key=api_key, base_url=api_base_with_port_v1)
    messages = [
        {"role": "system", "content": [{"type": "text", "text": PROMPT_SYSTEM}]},
        {"role": "user", "content": [{"type": "text", "text": build_user_message(caption_text)}]},
    ]
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            content = resp.choices[0].message.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            return {"ok": True, "data": json.loads(content)}
        except Exception as e:
            last_err = str(e)
            if attempt < retries:
                time.sleep(delay * attempt)
                continue
            return {"ok": False, "error": last_err or "unknown"}

# ----------------- 工具：批切分 -----------------
def batched(seq: List[Path], batch_size: int):
    for i in range(0, len(seq), batch_size):
        yield seq[i:i+batch_size]

# ----------------- 扫描一批文件 -> 任务扁平化 -----------------
def collect_tasks_for_batch(files: List[Path]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    对传入的这批 XML 文件：
      - 若 <reasoning_caption_parse> 存在且不是 'Error:' 开头 -> 跳过
      - 其他情况加入任务
    返回:
      tasks: [{"xml_path": str, "grounding_index": int, "caption": str}]
      stats: {"in_files": N, "total_groundings": M, "todo": K}
    """
    tasks = []
    total_groundings = 0
    for xml_path in files:
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            g_list = root.findall("grounding")
            total_groundings += len(g_list)
            for idx, g in enumerate(g_list):
                cap = g.find("caption")
                if cap is None or not (cap.text or "").strip():
                    continue
                caption = cap.text.strip()
                existing_reason = g.find("reasoning_caption_parse")
                if existing_reason is not None:
                    reason_text = (existing_reason.text or "").strip()
                    if reason_text and not reason_text.startswith("Error:"):
                        # 已成功，跳过
                        continue
                tasks.append({"xml_path": str(xml_path), "grounding_index": idx, "caption": caption})
        except Exception:
            traceback.print_exc()
            continue
    stats = {
        "in_files": len(files),
        "total_groundings": total_groundings,
        "todo": len(tasks)
    }
    return tasks, stats

# ----------------- 扁平任务 -> 按 GPU 均衡分配 -----------------
def split_tasks_balanced(tasks: List[Dict[str, Any]], n_buckets: int) -> List[List[Dict[str, Any]]]:
    """
    以 caption 长度作为粗略复杂度，做贪心装箱。
    """
    if n_buckets <= 1:
        return [tasks]
    scored = [(len(t["caption"]), t) for t in tasks]
    scored.sort(key=lambda x: x[0], reverse=True)
    buckets = [[] for _ in range(n_buckets)]
    loads = [0] * n_buckets
    for cost, task in scored:
        i = loads.index(min(loads))
        buckets[i].append(task)
        loads[i] += cost
    return buckets

# ----------------- XML 写入相关（minidom 格式化） -----------------
# def prettify(elem: ET.Element) -> str:
#     rough_str = ET.tostring(elem, encoding="utf-8")
#     reparsed = minidom.parseString(rough_str)
#     return reparsed.toprettyxml(indent="\t", encoding="utf-8").decode("utf-8")
# ----------------- XML 处理 -----------------
def prettify(elem: ET.Element) -> str:
    """格式化XML - 生成干净的输出，没有多余空行"""
    def indent_xml(elem, level=0):
        i = "\n" + level * "\t"
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "\t"
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for child in elem:
                indent_xml(child, level + 1)
            if not child.tail or not child.tail.strip():
                child.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i
    
    indent_xml(elem)
    xml_str = ET.tostring(elem, encoding='utf-8').decode('utf-8')
    
    # 添加 XML 声明
    if not xml_str.startswith('<?xml'):
        xml_str = "<?xml version='1.0' encoding='utf-8'?>\n" + xml_str
    
    return xml_str

def insert_caption_attributes_node(attributes: List[Dict[str, Any]]) -> ET.Element:
    cap_attr = ET.Element("caption_attributes")
    for attr in attributes or []:
        one = ET.SubElement(cap_attr, "attributes")
        t = ET.SubElement(one, "type")
        t.text = str(attr.get("aspect", "") or "")
        d = ET.SubElement(one, "description")
        d.text = str(attr.get("description", "") or "")
    return cap_attr

def insert_reasoning_node_text(text: str) -> ET.Element:
    node = ET.Element("reasoning_caption_parse")
    node.text = (text or "").strip()
    return node

def apply_results_to_xml(xml_path: Path, results_for_this_xml: List[Dict[str, Any]], out_dir: Path):
    """
    将一个 XML 的多条 grounding 结果回写：
      - 定位到各 grounding_index
      - 清理旧的 caption_attributes / reasoning_caption_parse
      - 在 caption 后插入新的 caption_attributes + reasoning_caption_parse
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        g_list = root.findall("grounding")

        # 保序写入：按 index 排序
        items = sorted(results_for_this_xml, key=lambda r: r["grounding_index"])

        for r in items:
            i = r["grounding_index"]
            if i < 0 or i >= len(g_list):
                continue
            g = g_list[i]
            # 清理旧节点（如果存在）
            for old in g.findall("caption_attributes"):
                g.remove(old)
            for old in g.findall("reasoning_caption_parse"):
                g.remove(old)

            # 找 caption 的插入点
            cap = g.find("caption")
            children = list(g)
            insert_idx = 0
            if cap is not None:
                try:
                    insert_idx = children.index(cap) + 1
                except ValueError:
                    insert_idx = 0

            # 构造新节点
            if r.get("ok"):
                attr_node = insert_caption_attributes_node(r.get("attributes", []))
                rea_node = insert_reasoning_node_text(r.get("analysis", ""))
            else:
                attr_node = insert_caption_attributes_node([])
                rea_node = insert_reasoning_node_text(f"Error: {r.get('error','unknown')}")

            # 按顺序插入（caption 后）
            g.insert(insert_idx, attr_node)
            g.insert(insert_idx + 1, rea_node)

        # 输出
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / xml_path.name
        out_file.write_text(prettify(root), encoding="utf-8")

    except Exception:
        print(f"✗ Failed to write back: {xml_path.name}")
        traceback.print_exc()

# ----------------- 单 GPU（单端口）线程池并发请求 -----------------
def run_tasks_on_one_gpu(rank: int, port: int, api_base: str, api_key: str, model: str,
                         tasks: List[Dict[str, Any]],
                         per_gpu_concurrency: int,
                         delay: float, max_retries: int) -> Dict[str, Any]:
    """
    返回该 GPU 处理的结果字典：
      {
        "rank": rank,
        "port": port,
        "results": [ {...}, ... ],
        "success": n,
        "failed": m
      }
    其中 results 每项形如：
      {"xml_path": str, "grounding_index": int, "ok": bool, "attributes": [...], "analysis": "...", "error": "..."}
    """
    base_with_port_v1 = _normalize_base(api_base, port)
    out_results: List[Dict[str, Any]] = []
    success = 0
    failed = 0

    def _do_one(t):
        res = call_llm_parser(base_with_port_v1, api_key, model, t["caption"],
                              retries=max_retries, delay=delay)
        if res.get("ok"):
            return {
                "xml_path": t["xml_path"],
                "grounding_index": t["grounding_index"],
                "ok": True,
                "attributes": res["data"].get("attributes", []),
                "analysis": res["data"].get("analysis", "")
            }
        else:
            return {
                "xml_path": t["xml_path"],
                "grounding_index": t["grounding_index"],
                "ok": False,
                "error": res.get("error", "unknown")
            }

    with ThreadPoolExecutor(max_workers=max(per_gpu_concurrency, 1)) as ex:
        futs = [ex.submit(_do_one, t) for t in tasks]
        for _ in tqdm(as_completed(futs), total=len(futs),
                      desc=f"[GPU{rank}@{port}] 处理groundings", dynamic_ncols=True):
            try:
                r = _.result()
                out_results.append(r)
                if r.get("ok"):
                    success += 1
                else:
                    failed += 1
            except Exception as e:
                out_results.append({
                    "xml_path": "<unknown>",
                    "grounding_index": -1,
                    "ok": False,
                    "error": f"executor_error: {e}"
                })
                failed += 1

    return {
        "rank": rank,
        "port": port,
        "results": out_results,
        "success": success,
        "failed": failed
    }

# ----------------- 批次统计落盘 -----------------
def write_batch_stats(out_dir: Path, batch_id: int,
                      gpu_reports: List[Dict[str, Any]],
                      all_results: List[Dict[str, Any]]):
    stats_dir = out_dir / "batch_stats"
    stats_dir.mkdir(parents=True, exist_ok=True)

    # GPU 汇总
    gpu_summary = {
        f"GPU{r['rank']}@{r['port']}": {"success": r["success"], "failed": r["failed"]}
        for r in gpu_reports
    }

    # 文件维度汇总
    file_summary: Dict[str, Dict[str, int]] = defaultdict(lambda: {"success": 0, "failed": 0})
    failed_captions: List[Dict[str, str]] = []
    for r in all_results:
        xml_name = Path(r.get("xml_path", "<unknown>")).name
        if r.get("ok"):
            file_summary[xml_name]["success"] += 1
        else:
            file_summary[xml_name]["failed"] += 1
            failed_captions.append({
                "xml": xml_name,
                "caption_index": r.get("grounding_index", -1),
                "error": str(r.get("error", "unknown"))
            })

    payload = {
        "batch_id": batch_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "gpu_summary": gpu_summary,
        "file_summary": file_summary,
        "failed_captions": failed_captions
    }

    out_json = stats_dir / f"batch_{batch_id}.json"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

# ----------------- 主流程（按批处理 + 批内写回） -----------------
def main():
    ap = argparse.ArgumentParser(description="Batch-by-file XML parsing with multi-GPU vLLM and per-GPU thread pools (v2).")
    ap.add_argument("--input-dir", required=True, type=Path, help="输入 XML 目录")
    ap.add_argument("--out-dir",   required=True, type=Path, help="输出 XML 目录（不会覆盖原目录）")
    ap.add_argument("--api-base",  default="http://localhost", type=str, help="vLLM 服务基础地址（不含端口或已带端口）")
    ap.add_argument("--start-port", default=18000, type=int, help="首端口（每卡 +1）")
    ap.add_argument("--api-key",   default=os.getenv("SILICONFLOW_API_KEY", ""), type=str, help="API key")
    ap.add_argument("--model",     default="openai/gpt-oss-20b", type=str, help="模型名称")
    ap.add_argument("--gpus",      default=None, type=int, help="GPU 数（默认自动探测）")
    ap.add_argument("--batch-size", default=None, type=int, help="每批 XML 数（默认等于 GPU 数）")
    ap.add_argument("--per-gpu-concurrency", default=8, type=int, help="每个 GPU 线程池并发请求数")
    ap.add_argument("--delay",     default=0.2, type=float, help="请求失败的指数退避基础延时（秒）")
    ap.add_argument("--max-retries", default=5, type=int, help="失败重试次数")
    ap.add_argument("--skip-existing-out", action="store_true",
                    help="若输出目录已存在同名 XML 则跳过该文件（适合断点续跑）")
    args = ap.parse_args()

    # GPU 数
    n_gpu = args.gpus
    if n_gpu is None:
        if torch is not None:
            try:
                n_gpu = torch.cuda.device_count()
            except Exception:
                n_gpu = 1
        else:
            n_gpu = 1
    n_gpu = max(1, n_gpu)

    # batch 大小
    batch_size = args.batch_size or n_gpu

    # XML 列表
    all_xmls = sorted(args.input_dir.glob("*.xml"))
    print(f"🔍 Found {len(all_xmls)} XML files. GPUs={n_gpu}, batch_size={batch_size}, per_gpu_concurrency={args.per_gpu_concurrency}")
    if not all_xmls:
        print("没有可处理的 XML。")
        return

    # 批处理
    batch_id = 0
    processed_files = 0
    total_files = len(all_xmls)
    
    with tqdm(total=total_files, desc="全部文件进度", dynamic_ncols=True) as pbar:
        for batch_files in batched(all_xmls, batch_size):
            batch_id += 1
            print(f"\n=== 批次 {batch_id} | 文件数: {len(batch_files)} ===")

            # 如果全都已在 out-dir 存在且选择了 skip-existing-out，则跳过本批
            if args.skip_existing_out and all((args.out_dir / f.name).exists() for f in batch_files):
                print("本批文件在输出目录已存在，跳过。")
                continue

            # 扁平化当前批次的任务
            tasks, stats = collect_tasks_for_batch(batch_files)
            print(f"  批内统计：XML={stats['in_files']} | total_groundings={stats['total_groundings']} | 待处理={stats['todo']}")

            # 若这批完全无需处理（都已成功），则直接复制原 XML 到 out-dir
            if not tasks:
                print("  本批无需新增解析，直接复制原 XML 到目标目录。")
                args.out_dir.mkdir(parents=True, exist_ok=True)
                for f in batch_files:
                    out_p = args.out_dir / f.name
                    if out_p.exists() and args.skip_existing_out:
                        continue
                    out_p.write_text(Path(f).read_text(encoding="utf-8"), encoding="utf-8")
                # 也输出空统计
                write_batch_stats(args.out_dir, batch_id, gpu_reports=[], all_results=[])
                continue

            # 负载均衡到各 GPU
            buckets = split_tasks_balanced(tasks, n_gpu)

            # 多 GPU 同时并行：每卡开一个 Future，卡内再开线程池并发
            gpu_reports: List[Dict[str, Any]] = []
            all_results: List[Dict[str, Any]] = []
            with ThreadPoolExecutor(max_workers=n_gpu) as ex:
                futures = []
                for rank in range(n_gpu):
                    bucket = buckets[rank]
                    if not bucket:
                        continue
                    port = args.start_port + rank
                    print(f"  -> GPU{rank} @ {port}: {len(bucket)} groundings")
                    futures.append(ex.submit(
                        run_tasks_on_one_gpu,
                        rank, port, args.api_base, args.api_key, args.model,
                        bucket, args.per_gpu_concurrency, args.delay, args.max_retries
                    ))
                for fut in as_completed(futures):
                    rep = fut.result()
                    gpu_reports.append(rep)
                    all_results.extend(rep["results"])

            # 将结果映射回这批 XML 并立即写回
            by_xml: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            for r in all_results:
                xp = r.get("xml_path")
                if not xp or xp == "<unknown>":
                    continue
                by_xml[xp].append(r)

            print(f"  回写本批 XML：{len(by_xml)} 个文件")
            for f in batch_files:
                f_path_str = str(f)
                if f_path_str in by_xml:
                    apply_results_to_xml(f, by_xml[f_path_str], args.out_dir)
                else:
                    # 该文件在本批没有任务（比如都已成功），复制原文件
                    out_p = args.out_dir / f.name
                    if out_p.exists() and args.skip_existing_out:
                        continue
                    out_p.write_text(Path(f).read_text(encoding="utf-8"), encoding="utf-8")

            # 批次统计落盘
            write_batch_stats(args.out_dir, batch_id, gpu_reports, all_results)
            print(f"=== 批次 {batch_id} 完成，已写回 {len(batch_files)} 个 XML ===")

            processed_files += len(batch_files)
            pbar.update(len(batch_files))
    print(f"\n✅ 全部完成。输出目录：{args.out_dir}")

if __name__ == "__main__":
    main()
