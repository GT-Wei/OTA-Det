#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LLM Caption Attribute Parser for Grounding XML

Usage:
python run_vlm_parse_xml_gptoss.py \
      --input-dir ../datasets/LLM_Caption_Parse/annotations/AerialVG_train \
      --out-dir ./xml_parsed \
      --api-key $SILICONFLOW_API_KEY \
      --api-base http://localhost:18000/v1 \
      --model openai/gpt-oss-20b
python run_vlm_parse-gpt-oss.py \
  --input-dir ../datasets/LLM_Caption_Parse/annotations_parse/AerialVG_train \
  --out-dir ../datasets/LLM_Caption_Parse/annotations_parse/AerialVG_train \
  --api-base http://localhost:18000/v1 \
  --api-key EMPTY \
  --model openai/gpt-oss-20b \
  --delay 0.5
python run_vlm_parse-gpt-oss.py \
  --input-dir ../datasets/LLM_Caption_Parse/annotations_parse/AerialVG_test \
  --out-dir ../datasets/LLM_Caption_Parse/annotations_parse/AerialVG_test \
  --api-base http://localhost:18000/v1 \
  --api-key EMPTY \
  --model openai/gpt-oss-20b \
  --delay 0.5
python run_vlm_parse-gpt-oss.py \
  --input-dir ../datasets/LLM_Caption_Parse/annotations_parse/AerialVG_val \
  --out-dir ../datasets/LLM_Caption_Parse/annotations_parse/AerialVG_val \
  --api-base http://localhost:18001/v1 \
  --api-key EMPTY \
  --model openai/gpt-oss-20b \
  --delay 0.5
"""

import os
import json
import time
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET
from xml.dom import minidom
from tqdm import tqdm
from openai import OpenAI

# ----------------- LLM PROMPT -----------------
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

# ----------------- LLM 调用 -----------------
def call_llm_parser(client: OpenAI, caption_text: str, model: str,
                    temperature=0.15, top_p=0.9, max_tokens=8192, retries=5, delay=0.5) -> dict:
    messages = [
        {"role": "system", "content": [{"type": "text", "text": PROMPT_SYSTEM}]},
        {"role": "user", "content": [{"type": "text", "text": build_user_message(caption_text)}]},
    ]
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
            return json.loads(content)
        except Exception as e:
            if attempt < retries:
                time.sleep(delay * attempt)
                continue
            return {"error": str(e)}

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

def insert_caption_attributes(grounding: ET.Element, attributes: list):
    """添加 <caption_attributes> 节点"""
    cap_attr = ET.Element("caption_attributes")
    for attr in attributes:
        one = ET.SubElement(cap_attr, "attributes")
        ET.SubElement(one, "type").text = attr.get("aspect", "")
        ET.SubElement(one, "description").text = attr.get("description", "")
    grounding.append(cap_attr)

def insert_reasoning_node(grounding: ET.Element, reasoning_text: str):
    """添加 <reasoning_caption_parse> 节点"""
    node = ET.Element("reasoning_caption_parse")
    node.text = reasoning_text.strip() if reasoning_text else ""
    grounding.append(node)

def process_xml(xml_path: Path, out_dir: Path, client: OpenAI, model: str, delay: float = 0.5):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        modified = False

        for grounding in root.findall("grounding"):
            caption_node = grounding.find("caption")
            if caption_node is None or not caption_node.text or not caption_node.text.strip():
                continue
            caption = caption_node.text.strip()

            # 检查是否已存在节点
            existing_reason_node = grounding.find("reasoning_caption_parse")
            if existing_reason_node is not None:
                reason_text = (existing_reason_node.text or "").strip()
                # reasoning 不是 Error 开头则跳过
                if reason_text and not reason_text.startswith("Error:"):
                    continue

                # reasoning 是 Error 开头，则清除旧节点准备重新解析
                for old_node in grounding.findall("caption_attributes"):
                    grounding.remove(old_node)
                for old_node in grounding.findall("reasoning_caption_parse"):
                    grounding.remove(old_node)
                    
            result = call_llm_parser(client, caption, model, delay=delay)
            time.sleep(delay)

            if "error" in result:
                insert_caption_attributes(grounding, [])
                insert_reasoning_node(grounding, f"Error: {result['error']}")
            else:
                insert_caption_attributes(grounding, result.get("attributes", []))
                insert_reasoning_node(grounding, result.get("analysis", ""))

            modified = True

        if modified:
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / xml_path.name
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(prettify(root))

    except Exception as e:
        print(f"✗ Error processing {xml_path.name}: {e}")

# ----------------- 主函数 -----------------
def main():
    parser = argparse.ArgumentParser(description="Parse captions in XMLs with LLM and insert structured <caption_attributes> and <reasoning_caption_parse>.")
    parser.add_argument("--input-dir", type=str, required=True, help="输入 XML 文件夹路径")
    parser.add_argument("--out-dir", type=str, required=True, help="输出文件夹路径")
    parser.add_argument("--api-base", type=str, default="http://localhost:18000/v1", help="API base URL")
    parser.add_argument("--api-key", type=str, default=os.getenv("SILICONFLOW_API_KEY", ""), help="API key")
    parser.add_argument("--model", type=str, default="openai/gpt-oss-20b", help="模型名称")
    parser.add_argument("--delay", type=float, default=0.5, help="每条 caption 调用间隔秒数")
    args = parser.parse_args()

    client = OpenAI(api_key=args.api_key, base_url=args.api_base)
    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    xml_files = sorted(input_dir.glob("*.xml"))

    print(f"🔍 Found {len(xml_files)} XML files to process.")
    for xml_path in tqdm(xml_files, desc="Processing XMLs", dynamic_ncols=True):
        process_xml(xml_path, out_dir, client, args.model, delay=args.delay)

    print(f"✅ Completed. Parsed XMLs saved to {out_dir}")

if __name__ == "__main__":
    main()
