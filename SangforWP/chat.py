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


# The streamlit script starts here
logging.info("Starting Streamlit script ...")

# Prepare to connect to Redis
redis_host = os.getenv('REDIS_HOST', 'localhost')  # default to 'localhost' if not set
redis_port = os.getenv('REDIS_PORT', '6379')  # default to '6379' if not set
redis_db = os.getenv('REDIS_DB', '0')  # default to '0' if not set. RediSearch only operates on the default (0) db
 # Instantiates a Redis client. decode_responses=False to avoid decoding the returned embedding vectors
r = Redis(host=redis_host, port=redis_port, db=redis_db, decode_responses=False)


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
            query_str = json_gpt(QUERY_GEN_PROMPT)["searchQuery"]
            logging.info(f"Generated query: {query_str}")
            try:
                query_embedding = openai.Embedding.create(input=query_str, model="text-embedding-ada-002")["data"][0]["embedding"]
                query_vec = np.array(query_embedding).astype(np.float32).tobytes()
                # Prepare the query
                query_base = (Query("*=>[KNN 2 @page_embedding $vec as score]").sort_by("score").return_fields("score", "page_num", "content").dialect(2))
                query_param = {"vec": query_vec}
                query_results = r.ft(INDEX_NAME).search(query_base, query_param).docs
                result_str = query_results[0].content + "\n\n------another page------\n\n" + query_results[1].content
            except Exception as e:
                logging.error(f"Error querying Reids with embedding: {e}")
                st.error("无法搜索答案，这很可能是系统故障导致，请联系我的主人。")
                st.stop()
            logging.info(f"Search results: ({query_results[0].score}){query_results[0].content[:20]}".replace("\n", "") +
                        "..." + f"({query_results[1].score}){query_results[1].content[:20]}".replace("\n", "") + "...")
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
    st.write(f"\n参考白皮书：第{query_results[0].page_num}页和第{query_results[1].page_num}页")

st.write("**注意：人工智能生成内容仅供参考。**")
logging.info("Streamlit script ended.")