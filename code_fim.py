from openai import OpenAI

print("Code Fim by:")
print("siliconflow + v3")
client = OpenAI(api_key="sk-lyczdgelbgsbknxvlbqaxpeodwkpzaxsnyqdilfnfizkbtif", base_url="https://api.siliconflow.cn/v1/")

response = client.completions.create(
    model="Pro/deepseek-ai/DeepSeek-V3",
    prompt="def fib(a):",
    suffix="    return fib(a-1) + fib(a-2)",
    # prompt="朝辞白帝彩云间，",
    # suffix="两岸猿声啼不住，轻舟已过万重山。",
    max_tokens=128
)
print(response.choices[0].text)

print("===============")
print("dashscope + qwen-coder")
client = OpenAI(api_key="sk-3f13933b10bb4992b16b667f5ea26124", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

response = client.completions.create(
    model="qwen2.5-coder-7b-instruct",
    prompt="<|fim_prefix|>def fib(a):<|fim_suffix|>    return fib(a-1) + fib(a-2)<|fim_middle|>",
    max_tokens=128
)
print(response.choices[0].text)