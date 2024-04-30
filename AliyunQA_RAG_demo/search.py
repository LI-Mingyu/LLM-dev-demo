import numpy as np
import streamlit as st
import tiktoken
import openai 
import logging
from redis import Redis
from redis.commands.search.query import Query
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
INDEX_NAME = "AliyunQA"

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

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# The streamlit script starts here
logging.info("Starting Streamlit script ...")

# Prepare to connect to Redis
redis_host = os.getenv('REDIS_HOST', 'localhost')  # default to 'localhost' if not set
redis_port = os.getenv('REDIS_PORT', '6379')  # default to '6379' if not set
redis_db = os.getenv('REDIS_DB', '0')  # default to '0' if not set. RediSearch only operates on the default (0) db
 # Instantiates a Redis client. decode_responses=False to avoid decoding the returned embedding vectors
r = Redis(host=redis_host, port=redis_port, db=redis_db, decode_responses=False)


st.set_page_config(
    page_title="阿里云运维助手",
    page_icon="🤖",
)
st.subheader("阿里云运维助手")

user_prompt = st.text_input("请输入您的问题：","", key="input")

if st.button('提问', key='generationSubmit'):
    with st.spinner("人工智能正在思考..."):
        QUERY_GEN_PROMPT = f"""
You are an Aliyun operations assistant
You have access to a search API that returns troubleshooting articles.
Generate search query extracting key words from the user's question.

User question: {user_prompt}

Format: {{"searchQuery": "search query"}}"""
        
    query_str = json_gpt(QUERY_GEN_PROMPT)["searchQuery"]
    logging.info(f"Generated query: {query_str}")

    with st.spinner("人工智能正在搜索..."):
        query_embedding = openai.Embedding.create(input=query_str, model="text-embedding-ada-002")["data"][0]["embedding"]
        query_vec = np.array(query_embedding).astype(np.float32).tobytes()
        # Prepare the query
        query_base = (Query("*=>[KNN 2 @md_embedding $vec as score]").sort_by("score").return_fields("score", "url", "md").dialect(2))
        query_param = {"vec": query_vec}
        query_results = r.ft(INDEX_NAME).search(query_base, query_param).docs
        result_md = query_results[0].md + "\n\n" + query_results[1].md
    logging.info(f"Search results: ({query_results[0].score}){query_results[0].md[:20]}".replace("\n", "") +
                "..." + f"({query_results[1].score}){query_results[1].md[:20]}".replace("\n", "") + "...")

    system_prompt = "你是一个阿里云运维助手。请根据搜索结果回答用户提问，注意，请务必首先依赖搜索结果，而不是你自己已有的知识。如果搜索结果中包含了具体操作步骤，也请据此给用户具体操作指引。"
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": "用户提问：" + query_str},
                {"role": "user", "content": "搜索结果：" + result_md}]
    
    response = openai.ChatCompletion.create(
                        model=GPT_MODEL,
                        messages=messages,
                        temperature=0.5,
                        stream=True
                    )
    collected_resp_content = ""
    resp_display = st.empty()
    for chunk in response:
        if not chunk['choices'][0]['finish_reason']:
            collected_resp_content += chunk['choices'][0]['delta']['content']
            resp_display.write(collected_resp_content)
    logging.info(f"AI's response: {collected_resp_content[:50]}".replace("\n", "") + "..." + f"{collected_resp_content[-50:]}".replace("\n", ""))

st.write("**注意：人工智能生成内容仅供参考。**")
logging.info("Streamlit script ended.")