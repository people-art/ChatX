# main.py
import os
import tempfile

import streamlit as st
from files import file_uploader, url_uploader
from question import chat_with_doc
from aiavatar import AiAvatar
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import SupabaseVectorStore
from supabase import Client, create_client
from explorer import view_document
from stats import get_usage_today


logo = "./assets/logo.png" # 将这个路径换成你的logo图片的路径

# 添加logo
st.image(logo, use_column_width=True)

st.set_page_config(
    page_title="ChatX",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="./assets/favicon.ico"  # Replace this with the path to your favicon
)

supabase_url = st.secrets.supabase_url
supabase_key = st.secrets.supabase_service_key
openai_api_key = st.secrets.openai_api_key
anthropic_api_key = st.secrets.anthropic_api_key
supabase: Client = create_client(supabase_url, supabase_key)
self_hosted = st.secrets.self_hosted

embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
vector_store = SupabaseVectorStore(
    supabase, embeddings, table_name="documents")
models = ["gpt-3.5-turbo", "gpt-4"]
if anthropic_api_key:
    models += ["claude-v1", "claude-v1.3",
               "claude-instant-v1-100k", "claude-instant-v1.1-100k"]

# Set the theme
st.set_page_config(
    page_title="ChatX",
    layout="wide",
    initial_sidebar_state="expanded",
)

language_options = ["English", "Chinese"]
default_language = "English"
st.session_state['language'] = st.selectbox("Select Language", language_options, index=language_options.index(default_language))

english_text = {
    "title": "😊 ChatX - Retrieve information and Knowledge by AI 😊",
    "add_knowledge": "Add Knowledge",
    "chat": "Chat with your Ai-Avatar",
    "forget": "Forget",
    "explore": "Explore",
    
    # Add more keys as needed
}

chinese_text = {
    "title": "😊 ChatX - 改变获取信息和知识的方式 😊",
    "add_knowledge": "添加知识",
    "chat": "聊天",
    "forget": "删除",
    "explore": "探索",
    # Add more keys as needed
}

if st.session_state['language'] == "English":
    text = english_text
else:
    text = chinese_text

st.title(text["title"])

if self_hosted == "false":
    st.markdown('**📢 Note: In the public demo, access to functionality is restricted. You can only use the GPT-3.5-turbo model and upload files up to 1Mb. To use more models and upload larger files, consider self-hosting ChatX.**')

st.markdown("---\n\n")

st.session_state["overused"] = False
if self_hosted == "false":
    usage = get_usage_today(supabase)
    if usage > st.secrets.usage_limit:
        st.markdown(
            f"<span style='color:red'>You have used {usage} tokens today, which is more than your daily limit of {st.secrets.usage_limit} tokens. Please come back later or consider self-hosting.</span>", unsafe_allow_html=True)
        st.session_state["overused"] = True
    else:
        st.markdown(f"<span style='color:blue'>Usage today: {usage} tokens out of {st.secrets.usage_limit}</span>", unsafe_allow_html=True)
    st.write("---")

# Initialize session state variables
if 'model' not in st.session_state:
    st.session_state['model'] = "gpt-3.5-turbo"
if 'temperature' not in st.session_state:
    st.session_state['temperature'] = 0.0
if 'chunk_size' not in st.session_state:
    st.session_state['chunk_size'] = 500
if 'chunk_overlap' not in st.session_state:
    st.session_state['chunk_overlap'] = 0
if 'max_tokens' not in st.session_state:
    st.session_state['max_tokens'] = 256

# Create a radio button for user to choose between adding knowledge or asking a question
user_choice = st.radio(
    "Choose an action", (text['add_knowledge'], text['chat'], text['forget'], text["explore"]))

st.markdown("---\n\n")

if user_choice == text['add_knowledge']:
    # Display chunk size and overlap selection only when adding knowledge
    st.sidebar.title("Configuration")
    st.sidebar.markdown(
        "Choose your chunk size and overlap for adding knowledge.")
    st.session_state['chunk_size'] = st.sidebar.slider(
        "Select Chunk Size", 100, 1000, st.session_state['chunk_size'], 50)
    st.session_state['chunk_overlap'] = st.sidebar.slider(
        "Select Chunk Overlap", 0, 100, st.session_state['chunk_overlap'], 10)

    # Create two columns for the file uploader and URL uploader
    col1, col2 = st.columns(2)

    with col1:
        file_uploader(supabase, vector_store)
    with col2:
        url_uploader(supabase, vector_store)
elif user_choice == text['chat']:
    # Display model and temperature selection only when asking questions
    st.sidebar.title("Configuration")
    st.sidebar.markdown(
        "Choose your model and temperature for asking questions.")
    if self_hosted != "false":
        st.session_state['model'] = st.sidebar.selectbox(
        "Select Model", models, index=(models).index(st.session_state['model']))
    else:
        st.sidebar.write("**Model**: gpt-3.5-turbo")
        st.sidebar.write("**Self Host to unlock more models such as claude-v1 and GPT4**")
        st.session_state['model'] = "gpt-3.5-turbo"
    st.session_state['temperature'] = st.sidebar.slider(
        "Select Temperature", 0.0, 1.0, st.session_state['temperature'], 0.1)
    if st.secrets.self_hosted != "false":
        st.session_state['max_tokens'] = st.sidebar.slider(
            "Select Max Tokens", 256, 2048, st.session_state['max_tokens'], 2048)
    else:
        st.session_state['max_tokens'] = 256

    chat_with_doc(st.session_state['model'], vector_store, stats_db=supabase)
elif user_choice == text['forget']:
    st.sidebar.title("Configuration")

    AiAvatar(supabase)
elif user_choice == text['explore']:
    st.sidebar.title("Configuration")
    view_document(supabase)

st.markdown("---\n\n")




# 定义中英文footer文本
footer_text = {
    'en': "Copyright © 2023 Ai-Avatar Labs. All rights reserved.",
    'cn': "版权所有 © 2023 艾凡达实验室。保留所有权利。"
}

# 定义footer的HTML模板
footer_template = """<style>
.footer {{
  position: fixed;
  left: 0;
  bottom: 0;
  width: 100%;
  background-color: white;
  color: grey;
  text-align: center;
}}
</style>
<div class="footer">
<p>{}</p>
</div>
"""

# 根据用户选择的语言来设置footer
footer = footer_template.format(footer_text[st.session_state['language']])
st.markdown(footer, unsafe_allow_html=True)