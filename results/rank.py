import json
import re
from pathlib import Path

input_path = "glm_outputs.jsonl"  # 原始文件（含 prompt/response）
output_path = "glm_ranked.jsonl"  # 新生成，供 parse_result.py 用
prompt_id = "PromptA"  # 可自定义，如你后续有多个 prompt 版本

def extract_ranked_items(response):
    """从 LLM response 中提取排序列表"""
    match = re.search(r"排序[:：]?\s*\[(.*?)\]", response, re.DOTALL)
    if not match:
        return []
    items = match.group(1).split(",")
    return [i.strip().strip('"').strip("'") for i in items]

with open(input_path, 'r', encoding='utf-8') as fin, open(output_path, 'w', encoding='utf-8') as fout:
    for line in fin:
        obj = json.loads(line)
        user_id = obj.get("user_id")
        response = obj.get("response", "")
        ranked_items = extract_ranked_items(response)

        converted = {
            "user_id": user_id,
            "prompt_id": prompt_id,
            "ranked_items": ranked_items
        }
        fout.write(json.dumps(converted, ensure_ascii=False) + "\n")

print(f"已生成融合格式文件：{output_path}")
