def build_prompt(history, candidates):
    return f"""
你是一个推荐系统专家。以下是一个用户的历史观看记录和一个推荐候选列表。

用户历史记录：
{', '.join(history)}

候选电影列表（顺序随机）：
{', '.join(candidates)}

请你完成以下任务：
1. 根据用户的兴趣，从高到低对候选电影进行排序；
2. 简要说明你的排序理由；
3. 指出你认为最不相关的一项。

输出格式如下：
排序：[title1, title2, ..., title10]
理由：xxxxx
最不相关项：titleX
"""
