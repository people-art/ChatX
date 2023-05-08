import os
import requests
import fnmatch
import base64
import streamlit as st

import asyncio

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain

GITHUB_TOKEN = "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"

def parse_github_url(url):
    parts = url.strip("/").split("/")
    owner = parts[-2]
    repo = parts[-1]
    return owner, repo

def get_files_from_github_repo(owner, repo):
    url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/main?recursive=1"
    headers = {
        #"Authorization": f"token {token}",
        "Accept": "application/vnd.github+json"
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        content = response.json()
        return content["tree"]
    else:
        st.error(f"Error fetching repo contents: {response.status_code}")
        return []

def fetch_md_contents(files):
    md_contents = []
    for file in files:
        if file["type"] == "blob" and fnmatch.fnmatch(file["path"], "*.md"):
            response = requests.get(file["url"])
            if response.status_code == 200:
                content = response.json()["content"]
                decoded_content = base64.b64decode(content).decode('utf-8')
                print("Fetching Content from ", file['path'])
                md_contents.append(Document(page_content=decoded_content, metadata={"source": file['path']}))
            else:
                print(f"Error downloading file {file['path']}: {response.status_code}")
    return md_contents

def get_source_chunks(files):
    print("In get_source_chunks ...")
    source_chunks = []
    splitter = CharacterTextSplitter(separator=" ", chunk_size=1024, chunk_overlap=0)
    md_contents = fetch_md_contents(files)
    if not md_contents:
        print("No markdown contents found in the repository.")
        return source_chunks
    for source in md_contents:
        for chunk in splitter.split_text(source.page_content):
            source_chunks.append(Document(page_content=chunk, metadata=source.metadata))
    return source_chunks


def message(text, is_user=True, key=None, avatar_style="thumbs"):
    col1, col2 = st.columns([2, 8])

    with col1:
        if is_user:
            st.write("")
        else:
            avatar = get_avatar(style=avatar_style)
            st.image(avatar, width=50)

    with col2:
        if is_user:
            bg_color = "blue"
            color = "white"
        else:
            bg_color = "white"
            color = "black"

        st.markdown(
            f'<div style="background-color: {bg_color}; color: {color}; border-radius: 5px; padding: 5px;">{text}</div>',
            unsafe_allow_html=True,
        )

async def conversational_chat(qa, query, history):
    truncated_history = []
    remaining_tokens = 4097 - len(query) - 1  # Subtract 1 to account for the delimiter token

    for message in reversed(st):
        user_message, model_message = message
        message_tokens = len(user_message) + len(model_message) + 2  # Add 2 to account for the delimiter tokens

        if remaining_tokens - message_tokens >= 0:
            truncated_history.insert(0, message)
            remaining_tokens -= message_tokens
        else:
            break

    result = qa({"question": query, "chat_history": truncated_history})
    st.append((query, result["answer"]))
    return result["answer"]



async def chatgithub_main(github_url=None):

    logo_image = "./static/logo.png"
    st.image(logo_image,width=50)
    st.markdown(
            ''' 
            ## :black **ChatX** -- *Chat with anything you want* 
            This is `chat` with *github* demo. <br>
            You could `chat` with *video*, *images*, *database* and so on...<br>
            **ChatX** could `chat` with any *digital objects* and *agent*.
            ''',unsafe_allow_html=True)
    #st.markdown(" > Powered by -  🦜 LangChain + OpenAI + Streamlit + Whisper")


    # 添加访问令牌输入字段

    url = st.text_input("Enter the GitHub repository URL:")

    if url:
        GITHUB_OWNER, GITHUB_REPO = parse_github_url(url)
        all_files = get_files_from_github_repo(GITHUB_OWNER, GITHUB_REPO)

        CHROMA_DB_PATH = f'./chroma/{os.path.basename(GITHUB_REPO)}'

        chroma_db = None

        if not os.path.exists(CHROMA_DB_PATH):
            print(f'Creating Chroma DB at {CHROMA_DB_PATH}...')
            source_chunks = get_source_chunks(all_files)
            if not source_chunks:
                st.error("No documents found in the repository. Please try another repository.")
                return
    
            chroma_db = Chroma.from_documents(source_chunks, OpenAIEmbeddings(), persist_directory=CHROMA_DB_PATH)
            chroma_db.persist()
        else:
            print(f'Loading Chroma DB from {CHROMA_DB_PATH} ... ')
            chroma_db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=OpenAIEmbeddings())

        qa_chain = load_qa_chain(OpenAI(temperature=1), chain_type="stuff")
        qa = RetrievalQA(combine_documents_chain=qa_chain, retriever=chroma_db.as_retriever())

        st.session_state['ready'] = True

        if st.session_state['ready']:

            if 'generated' not in st.session_state:
                st.session_state['generated'] = ["Welcome! You can now ask any questions regarding " + GITHUB_REPO]

            if 'past' not in st.session_state:
                st.session_state['past'] = ["Hey!"]

            # container for chat history
            response_container = st.container()

            # container for text box
            container = st.container()

            with container:
                with st.form(key='my_form', clear_on_submit=True):
                    user_input = st.text_input("Query:", placeholder="e.g: Summarize the paper in a few sentences", key='input')
                    submit_button = st.form_submit_button(label='Send')

                if submit_button and user_input:
                    output = await conversational_chat(qa, user_input, st.session_state['history'])
                    output = str(output)


                    st.session_state['past'].append(user_input)
                    st.session_state['generated'].append(output)

            if st.session_state['generated']:
                with response_container:
                    for i in range(len(st.session_state['generated'])):
                        message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                        message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")

if __name__ == "__main__":
    asyncio.run(chatgithub_main())