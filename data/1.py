import pandas as pd

# 读取 u.data 文件（tab 分隔）
ratings = pd.read_csv("u.data", sep="\t", names=["userId", "movieId", "rating", "timestamp"])

# 读取 u.item 文件（| 分隔）
items = pd.read_csv("u.item", sep="|", encoding="latin-1", header=None, usecols=[0, 1])
items.columns = ["movieId", "title"]

# 合并成一个完整的数据集
df = ratings.merge(items, on="movieId")

# 只保留我们需要的字段
df = df[['userId', 'title', 'rating']]
df.to_csv("movielens.csv", index=False)

print(df.head())
