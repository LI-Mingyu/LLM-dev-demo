import os
import argparse
from openai import OpenAI

# 设置命令行参数解析
parser = argparse.ArgumentParser(description='AI对话脚本')
parser.add_argument('--model', type=str, default='deepseek-ai/DeepSeek-R1',
                    help='指定使用的模型名称（默认：deepseek-ai/DeepSeek-R1）')

args = parser.parse_args()

client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url="https://api.siliconflow.cn/v1/",
)

# 初始化空对话历史
messages = []

# 直接进入对话循环
while True:
    user_input = input("\n请输入（直接回车退出对话）: ").strip()
    
    # 退出条件检测
    if not user_input or user_input.lower() == 'exit':
        break
    
    # 添加用户输入到对话历史
    messages.append({'role': 'user', 'content': user_input})
    
    # 生成回复
    completion = client.chat.completions.create(
        model=args.model,
        messages=messages
    )
    # 打印模型的思考过程和回复
    if hasattr(completion.choices[0].message, 'reasoning_content'):
        print(f"\n{args.model} think:", completion.choices[0].message.reasoning_content)
    print(f"\n{args.model}:", completion.choices[0].message.content)
    messages.append({'role': 'assistant', 'content': completion.choices[0].message.content})

print("对话已结束")