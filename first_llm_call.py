# Please install OpenAI SDK first: `pip3 install openai`

from openai import OpenAI

client = OpenAI(api_key="sk-lyczdgelbgsbknxvlbqaxpeodwkpzaxsnyqdilfnfizkbtif", base_url="https://api.siliconflow.cn/v1/")

response = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-V3",
    messages=[
        # {"role": "system", "content": "You are a helpful assistant"},
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello!"},

    ],
    stream=False
)

message = response.choices[0].message
if hasattr(message, 'reasoning_content'):
    print(message.reasoning_content)
print(message.content)


