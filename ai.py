from openai import OpenAI
import requests
import json
api_key = "1bc3aca311f155f00ad7a33d2eb5b86c472e558b"
client = OpenAI(api_key=api_key
                ,base_url="https://aistudio.baidu.com/llm/lmapi/v3")
# 这是最简单的API调用方式，适合单轮对话场景
response = client.chat.completions.create(
    # messages是一个列表，包含对话历史
    messages=[
        {
            'role': 'user',  # 角色可以是user(用户)、assistant(AI)或system(系统提示)
            'content': '讲述一个令人毛骨悚然的恐怖故事'  # 具体的对话内容
        }
    ],
    model="ernie-3.5-8k",  # 选择使用的模型，这里使用的是ernie-3.5-8k,也可以选择表格上的其他文生文模型
)

# 从响应中获取生成的内容
print("AI创作的恐怖故事:\n")
print(response.choices[0].message.content)  # choices[0]表示第一个（也是唯一的）回复