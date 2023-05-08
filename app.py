import streamlit as st


#Creating the chatbot interface
st.set_page_config(
        page_title='ChatX',
        page_icon='./assets/favicon.ico',
        layout='wide',
        menu_items={
            'Get help': 'https://www.ai-avatar.org/chatx',
            'Report a Bug': 'mailto:jerry.zhang@ai-avatar.org',
            'About': 'https://www.ai-avatar.org'
        }
    )


from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
import pickle
from pathlib import Path
from dotenv import load_dotenv
import os
from streamlit_chat import message
import io
import asyncio
from chatpdf import getDocEmbeds, conversational_chat


# add Starter

def add_logo_and_title():

    logo_image = "./static/logo.png"
    st.image(logo_image,width=50)

    st.markdown(
            ''' 
            ## :black **ChatX** 
            Chat with pdf, word, images, audio, video files and so on ... \
            Chat with anything you want...
            ''')
    #st.markdown(" > Powered by -  🦜 LangChain + OpenAI + Streamlit + Whisper")



# Add this function to create a footer
def add_footer():
    footer = """
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        width: 100%;
        background-color: white;
        text-align: center;
    }
    </style>
    <div class="footer">
        <p>© 2023 艾凡达实验室</p>
    </div>
    """
    st.markdown(footer, unsafe_allow_html=True)


async def main():
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    chain = load_qa_chain(llm, chain_type="stuff")

    add_logo_and_title()

    if 'history' not in st.session_state:
        st.session_state['history'] = []

    

    if 'ready' not in st.session_state:
        st.session_state['ready'] = False

    # Add sidebar for PDF upload and API key input
    st.sidebar.title("Settings")
    openai_api_key = st.sidebar.text_input("Enter OpenAI API key:")
    uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")

    if openai_api_key:
        api_key = openai_api_key

    if uploaded_file is not None:

        with st.spinner("Processing..."):
            uploaded_file.seek(0)
            file = uploaded_file.read()
            vectors = await getDocEmbeds(io.BytesIO(file), uploaded_file.name)
            qa = ConversationalRetrievalChain.from_llm(llm, retriever=vectors.as_retriever(), return_source_documents=True)




        st.session_state['ready'] = True

    st.divider()

    if st.session_state['ready']:

        if 'generated' not in st.session_state:
            st.session_state['generated'] = ["Welcome! You can now ask any questions regarding " + uploaded_file.name]

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

                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)

        if st.session_state['generated']:
            with response_container:
                for i in range(len(st.session_state['generated'])):
                    message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                    message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")

    add_footer()


if __name__ == "__main__":
    asyncio.run(main())

