# Please install OpenAI SDK first: `pip3 install openai`

from openai import OpenAI

client = OpenAI(api_key="sk-lyczdgelbgsbknxvlbqaxpeodwkpzaxsnyqdilfnfizkbtif", base_url="https://api.siliconflow.cn/v1/")

response = client.chat.completions.create(
    # model="deepseek-ai/DeepSeek-V3",
    model="Qwen/Qwen3-235B-A22B",
    messages=[
        # {"role": "system", "content": "You are a helpful assistant"},
        # {"role": "system", "content": "/no_think"},
        {"role": "user", "content": "Hello!"},

    ],
    extra_body={"enable_thinking": True, "thinking_budget": 32768},
    stream=False
)

message = response.choices[0].message
if hasattr(message, 'reasoning_content'):
    print(message.reasoning_content)
print(message.content)


