import json
import numpy as np
import pandas as pd
from collections import defaultdict
from itertools import combinations
from sklearn.metrics import ndcg_score

# 配置路径
INPUT_JSONL_PATH = "../results/glm_ranked.jsonl"  # LLM 排序输出
POSITIVE_MAP_PATH = "../data/llm_eval_samples.csv"  # 含正例信息
OUTPUT_CSV_PATH = "../results/eval_metrics_extended3.csv"  # 输出结果

# 加载 ground-truth 正例映射
positive_map = pd.read_csv(POSITIVE_MAP_PATH)[["user_id", "positive"]].set_index("user_id").to_dict()["positive"]

# 加载 LLM 排序结果
records = defaultdict(list)
with open(INPUT_JSONL_PATH, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        uid = str(obj["user_id"])
        prompt_id = obj.get("prompt_id", "PromptA")
        ranked_items = obj.get("ranked_items", [])
        records[uid].append((prompt_id, ranked_items))


# 融合排序函数：平均名次
def fuse_rankings(rank_lists):
    score = defaultdict(float)
    for rank in rank_lists:
        for i, item in enumerate(rank):
            score[item] += i + 1
    return sorted(score, key=lambda x: score[x])


# ILD 多样性计算（基于 Jaccard 距离）
def compute_ild(ranked_list):
    def jaccard_distance(a, b):
        s1, s2 = set(a.lower().split()), set(b.lower().split())
        return 1 - len(s1 & s2) / len(s1 | s2) if s1 | s2 else 0

    return np.mean([jaccard_distance(a, b) for a, b in combinations(ranked_list, 2)]) if len(ranked_list) >= 2 else 0


# 个性化计算：不同用户推荐列表的差异程度
def personalization_matrix(user_topk_dict):
    users = list(user_topk_dict.keys())
    scores = []
    for u1, u2 in combinations(users, 2):
        s1, s2 = set(user_topk_dict[u1]), set(user_topk_dict[u2])
        if s1 or s2:
            scores.append(1 - len(s1 & s2) / len(s1 | s2))
    return np.mean(scores) if scores else 0


# 主循环：为每个用户计算指标
results = []
user_topk_dict = {}

for user_id, prompt_ranks in records.items():
    fused_topk = fuse_rankings([rank for _, rank in prompt_ranks])
    top5 = fused_topk[:5]

    # 空推荐跳过
    if not top5:
        print(f"⚠️ 用户 {user_id} 的推荐列表为空，跳过")
        continue

    user_topk_dict[user_id] = top5
    positive = positive_map.get(int(user_id))

    if not positive:
        print(f"⚠️ 用户 {user_id} 没有正例标签，跳过")
        continue

    hit1 = int(positive == top5[0])
    hit5 = int(positive in top5)
    rel = [1 if item == positive else 0 for item in top5]
    ndcg = ndcg_score([rel], [list(reversed(range(len(rel))))])
    ild = compute_ild(top5)

    # 用户级个性化（与其他用户的推荐列表差异）
    others = {k: v for k, v in user_topk_dict.items() if k != user_id}
    if others:
        per_user_score = np.mean([
            1 - len(set(top5) & set(o)) / len(set(top5) | set(o))
            for o in others.values()
        ])
    else:
        per_user_score = 0

    results.append({
        "user_id": user_id,
        "final_top5": top5,
        "Hit@1": hit1,
        "Hit@5": hit5,
        "NDCG@5": round(ndcg, 4),
        "ILD@5": round(ild, 4),
        "PerUserPersonalization": round(per_user_score, 4)
    })

# 全局个性化指标
global_score = personalization_matrix(user_topk_dict)
for r in results:
    r["Personalization"] = round(global_score, 4)

# 写入 CSV 文件
df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV_PATH, index=False)
print(f"✅ 已保存所有评估结果：{OUTPUT_CSV_PATH}")
