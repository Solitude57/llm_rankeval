
ZHIPU_API_KEY = "7bfd63929f1747b8b128c4b6e34a97e7.j2Mg0duKlSsQKhRn"
import os
import zhipuai

zhipuai.api_key = "7bfd63929f1747b8b128c4b6e34a97e7.j2Mg0duKlSsQKhRn"  # 或用 os.getenv("ZHIPU_API_KEY")

def query_glm(prompt):
    response = zhipuai.model_api.sse_invoke(
        model="glm-4",
        prompt=prompt,
        top_p=0.7,
        temperature=0.9,
        return_type="text"
    )
    result = ""
    for event in response.events():
        if event.event == "add":
            result += event.data
        elif event.event == "error" or event.event == "interrupted":
            raise Exception("GLM API 错误中断")
    return result
