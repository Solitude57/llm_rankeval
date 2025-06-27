import pandas as pd
import json
import time
from prompt_builder import build_prompt
from glm_api import query_glm

df = pd.read_csv("../data/llm_eval_samples.csv")
output_path = "../results/glm_outputs.jsonl"

with open(output_path, "w", encoding="utf-8") as f_out:
    for idx, row in df.iterrows():
        try:
            history = eval(row["history"])
            candidates = eval(row["candidates"])
            prompt = build_prompt(history, candidates)
            response = query_glm(prompt)

            f_out.write(json.dumps({
                "user_id": row["user_id"],
                "prompt": prompt,
                "response": response,
                "positive": row["positive"]
            }, ensure_ascii=False) + "\n")
            print(f"[✓] 第 {idx + 1} 条完成")
        except Exception as e:
            print(f"[×] 第 {idx + 1} 条失败: {e}")

        time.sleep(0.5)