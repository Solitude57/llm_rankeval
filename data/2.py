import pandas as pd
import random

# 读取数据
df = pd.read_csv("movielens.csv")
df.dropna(inplace=True)

# 每个用户看过的电影
user_groups = df.groupby("userId")

samples = []

for user_id, group in user_groups:
    if len(group) < 10:
        continue  # 跳过行为太少的用户

    user_history_all = group['title'].tolist()
    history = random.sample(user_history_all, 5)

    remaining = list(set(user_history_all) - set(history))
    if not remaining:
        continue
    positive = random.choice(remaining)

    # 负例采样：从全集中取出未看过的
    all_items = df['title'].unique().tolist()
    user_watched = user_history_all + [positive]
    negative_items = list(set(all_items) - set(user_watched))

    if len(negative_items) < 9:
        continue

    negatives = random.sample(negative_items, 9)
    candidates = negatives + [positive]
    random.shuffle(candidates)

    samples.append({
        "user_id": user_id,
        "history": history,
        "candidates": candidates,
        "positive": positive
    })

# 保存为CSV
sample_df = pd.DataFrame(samples)
sample_df.to_csv("llm_eval_samples.csv", index=False)

print(f"生成了 {len(samples)} 条 LLM 评估样本")
