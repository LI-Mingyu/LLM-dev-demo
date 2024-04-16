import numpy as np
import streamlit as st
import tiktoken
import openai 
import logging
# Milvus 相关的库导入
from pymilvus import (
    connections,  # 用于连接 Milvus 数据库
    Collection,  # 用于操作 Milvus 集合
    CollectionSchema,  # 定义集合的 schema
    FieldSchema,  # 定义字段
    DataType,  # 数据类型
    utility  # 用于执行特定的数据库工具函数，比如检查集合是否存在
)

import os
from datetime import date, datetime, timedelta
import json

# Define a function to get the session id and the remote ip 
# Caution: this function is implemented in a hacky way and may break in the future
from streamlit import runtime
from streamlit.runtime.scriptrunner import get_script_run_ctx

GPT_MODEL = "gpt-4"
VECTOR_DIM = 1536 
DISTANCE_METRIC = "COSINE"  
INDEX_NAME = "SangforWP"

# Helper functions
def json_gpt(input: str):
    completion = openai.ChatCompletion.create(
        model=GPT_MODEL,
        messages=[
            {"role": "system", "content": "Output only valid JSON"},
            {"role": "user", "content": input},
        ],
        temperature=0.2,
    )

    text = completion.choices[0].message.content
    parsed = json.loads(text)

    return parsed

# 定义Milvus的相关函数

# The streamlit script starts here
logging.info("Starting Streamlit script ...")

# 连接Milvus
# 从环境变量获取 Milvus 服务器配置，或使用默认值
milvus_host = os.getenv('MILVUS_HOST', 'localhost')       # 默认为 localhost
milvus_port = os.getenv('MILVUS_PORT', '19530')           # Milvus 默认端口为 19530

# 建立连接
connections.connect(alias="default", host=milvus_host, port=milvus_port)


st.set_page_config(
    page_title="会说话的白皮书",
    page_icon="📚",
)
st.subheader("📚会说话的白皮书")
st.write("您好！我是《深信服超融合可靠性白皮书》，有什么我可以帮到您的？")

if "messages" not in st.session_state.keys():
    # Initialize the session_state.messages
    st.session_state.messages = [{"role": "system", "content": "你是一个有IT系统专业知识的人工智能助手，尤其精通超融合的可靠性技术。你的主要职责是回答用户关于深信服超融合系统的可靠性方面的问题。"}]
else: # Since streamlit script is executed every time a widget is changed, this "else" is not necessary, but improves readability
    # Display chat messages
    for message in st.session_state.messages:
        if message["role"] == "user" or message["role"] == "assistant":
            with st.chat_message(message["role"]):
                st.write(message["content"])

# Prepare the api_key to call OpenAI
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    logging.error("KEY is not set.")
    st.error('KEY未设置，请联系我的主人。')
    st.stop() 
else:
    openai.api_key = openai_api_key

# User-provided prompt
if user_prompt := st.chat_input('在此输入您的问题'):
    logging.info(f"User's prompt: {user_prompt}")
    with st.chat_message("user"):
        st.write(user_prompt)
        st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("assistant"):
        with st.spinner("人工智能正在思考..."):
            QUERY_GEN_PROMPT = f"""
You are an AI assistant that answers questions about Sangfor HCI reliability.
You have access to a search API that returns troubleshooting articles.
Generate search query extracting key words from the user's question.

User question: {user_prompt}
Format: {{"searchQuery": "search query"}}"""
            
            query_str = json_gpt(QUERY_GEN_PROMPT)["searchQuery"] # 使用GPT来生成查询字符串
            logging.info(f"Generated query: {query_str}")
            try:
                # 生成查询的向量表示
                query_embedding = openai.Embedding.create(input=query_str, model="text-embedding-ada-002")["data"][0]["embedding"]
                query_vec = np.array(query_embedding).astype(np.float32).tolist()

                # 使用 Milvus 执行向量搜索
                collection = Collection(name="pdf_page_collection")  # 加载已存在的集合
                search_params = {"metric_type": "L2", "params": {"nprobe": 16}}
                search_results = collection.search([query_vec], "page_embedding", search_params, limit=2, output_fields=["page_num", "content"])

                # 处理搜索结果
                result_contents = [doc.entity.get('content') for doc in search_results[0]]
                result_str = "\n\n------another page------\n\n".join(result_contents)
                st.session_state.messages.append({"role": "user", "content": "请根据搜索结果回答用户在前面遇到的问题……\n" + result_str})

            except Exception as e:
                logging.error(f"Error during Milvus processing: {e}")
                st.error("无法搜索答案，这很可能是系统故障导致，请联系我的主人。")
                st.stop()

            # 确保搜索结果不为空并且至少有一项结果
            if len(search_results) > 0:
                first_hits = search_results[0]  # 获取第一个查询的结果（这里假设只有一个查询）
                # 记录日志，显示最多前两个结果的信息
                if len(first_hits) > 1:
                    logging.info(f"Search results: ({first_hits[0].score}){first_hits[0].entity.get('content')[:20]}..." +
                                f"({first_hits[1].score}){first_hits[1].entity.get('content')[:20]}...")
                elif len(first_hits) == 1:
                    logging.info(f"Search result: ({first_hits[0].score}){first_hits[0].entity.get('content')[:20]}...")
                else:
                    logging.info("No results found.")
            else:
                logging.error("No results from Milvus search.")
            
            st.session_state.messages.append({"role": "user", "content": f"""请根据搜索结果回答user在前面遇到的问题。注意，请务必首先依赖搜索结果，而不是你自己已有的知识。如果搜索结果中包含了具体操作步骤，也请据此给用户具体操作指引。
搜索结果：\n{result_str}"""})
            
            test_messages = st.session_state.messages.copy() # make a copy of the messages for testing & debugging
            try:
                gpt_response = openai.ChatCompletion.create(
                        model=GPT_MODEL,
                        messages=st.session_state.messages,
                        temperature=0.5,
                        stream=True)
                resp_display = st.empty()
                collected_resp_content = ""
                for chunk in gpt_response:
                    if not chunk['choices'][0]['finish_reason']:
                        collected_resp_content += chunk['choices'][0]['delta']['content']
                        resp_display.write(collected_resp_content)
            except Exception as e:
                logging.error(f"Error generating response from OpenAI: {e}")
                st.error('AI没有响应，可能是因为我们正在经历访问高峰，请稍后刷新页面重试。如果问题仍然存在，请联系我的主人。')
                st.stop() 
            logging.info(f"AI's response: {collected_resp_content[:50]}".replace("\n", "") + "..." + f"{collected_resp_content[-50:]}".replace("\n", ""))

    # Count the number of tokens used
    tokenizer = tiktoken.encoding_for_model(GPT_MODEL)
    # Because of ChatGPT's message format, there are 4 extra tokens for each message
    # Ref: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    n_token_input = 0
    for message in st.session_state.messages:
        n_token_input += len(tokenizer.encode(message["content"])) + 4 
    n_token_input += 3 # every reply is primed with <|start|>assistant<|message|>
    n_token_output = len(tokenizer.encode(collected_resp_content))
    logging.info(f"Token usage: {n_token_input} -> {n_token_output}")

    # Add the generated msg to session state
    st.session_state.messages[-1] = {"role": "assistant", "content": collected_resp_content}

    
    # 确保有足够的结果
    if len(search_results) > 0 and len(search_results[0]) > 1:
        first_two_hits = search_results[0]  # 第一个查询的结果
        st.write(f"\n参考白皮书：第{first_two_hits[0].entity.get('page_num')}页和第{first_two_hits[1].entity.get('page_num')}页")
    else:
        st.write("搜索结果不足以提供相应的页面号。")
    
    # # 检测搜索结果, 用于测试
    # if search_results and len(search_results[0]) > 0:
    #     for hit in search_results[0]:  # 遍历第一个（或唯一的）查询结果中的命中
    #         logging.info(f"Document ID: {hit.id}, Score: {hit.score}, Content: {hit.entity.get('content')[:50]}")  # 日志记录前50个字符
    # else:
    #     logging.error("No results found or search failed.")

st.write("**注意：人工智能生成内容仅供参考。**")
logging.info("Streamlit script ended.")