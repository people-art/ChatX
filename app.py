import streamlit as st
from chatpdf import chatpdf_main
from chatgithub import chatgithub_main
import asyncio

# 创建侧边栏菜单
mode = st.sidebar.radio(
    "Choose a working mode",
    ("chatpdf", "chatgithub", "chatdb", "chatvideo"),
    index=0,
)

def app_main():
    # 初始化 session_state
    if 'mode' not in st.session_state:
        st.session_state['mode'] = mode

    # 当用户更改模式时，更新 session_state
    if mode != st.session_state['mode']:
        st.session_state['mode'] = mode

    if st.session_state['mode'] == "chatpdf":
        asyncio.run(chatpdf_main())
    elif st.session_state['mode'] == "chatgithub":
        # 调用 chatgithub_main 函数（需要在 chatgithub.py 中创建）
        chatgithub_main()
    elif st.session_state['mode'] == "chatdb":
        # 调用 chatdb_main 函数（需要在 chatdb.py 中创建）
        pass
    elif st.session_state['mode'] == "chatvideo":
        # 调用 chatvideo_main 函数（需要在 chatvideo.py 中创建）
        pass

app_main()
