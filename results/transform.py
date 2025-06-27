import json
import re

input_file = "glm_outputs.jsonl"
output_file = "glm_outputs.jsonl2"
prompt_id = "PromptA"  # 如你后续有多个 prompt，可动态赋值


def extract_ranking(response_text):
    match = re.search(r'排序：?\[(.*?)\]', response_text, re.DOTALL)
    if not match:
        return []
    items = match.group(1).split(",")
    return [item.strip().strip('"').strip("'") for item in items]


with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
    for line in fin:
        obj = json.loads(line)
        user_id = obj["user_id"]
        response = obj["response"]
        ranked_items = extract_ranking(response)

        out = {
            "user_id": user_id,
            "prompt_id": prompt_id,
            "ranked_items": ranked_items
        }
        fout.write(json.dumps(out, ensure_ascii=False) + "\n")

print(f"✅ 已生成：{output_file}")
