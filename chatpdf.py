# chat with PDF files using langchain
import os
import pickle
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from pathlib import Path
from dotenv import load_dotenv
import io
import asyncio
import boto3
import streamlit as st

from streamlit_chat import message




load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')  
aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')


async def storeDocEmbeds(file, filename):
    reader = PdfReader(file)
    corpus = ''.join([p.extract_text() for p in reader.pages if p.extract_text()])
    
    splitter =  RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200,)
    chunks = splitter.split_text(corpus)
    
    embeddings = OpenAIEmbeddings(openai_api_key = api_key)
    vectors = FAISS.from_texts(chunks, embeddings)
    
    with open(filename + ".pkl", "wb") as f:
        pickle.dump(vectors, f)
    
    # Upload the file to S3
    bucket_name = "chatx"
    object_name = filename + ".pkl"
    file_path = object_name
    upload_to_s3(bucket_name, file_path, object_name)

def download_from_s3(bucket_name, object_name, file_path=None):
    s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
    
    if file_path is None:
        file_path = Path(object_name).name

    try:
        s3.download_file(bucket_name, object_name, file_path)
        print(f"File {object_name} downloaded from {bucket_name} to {file_path}")
        return file_path
    except Exception as e:
        print(f"Error downloading file from S3: {e}")
        return None


async def getDocEmbeds(file, filename):
    local_path = filename + ".pkl"
    s3_object_name = filename + ".pkl"
    
    if not os.path.isfile(local_path):
        downloaded_path = download_from_s3('chatx', s3_object_name)
        
        if downloaded_path is None:
            await storeDocEmbeds(file, filename)

    with open(local_path, "rb") as f:
        global vectores
        vectors = pickle.load(f)
        
    return vectors


async def conversational_chat(qa, query, st):
    # Truncate the history to fit within the model's maximum context length
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

def upload_to_s3(bucket_name, file_path, object_name=None):
    s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
    
    try:
        s3.upload_file(file_path, bucket_name, object_name)
        print(f"{object_name} uploaded to {bucket_name} bucket successfully.")
    except Exception as e:
        print(f"Error uploading {object_name} to {bucket_name} bucket: {e}")


# add Starter

def add_logo_and_title():

    logo_image = "./static/logo.png"
    st.image(logo_image,width=50)

    st.markdown(
            ''' 
            ## :black **ChatX** -- *Chat with anything you want* 
            This is `chat` with *pdf* demo. <br>
            You could `chat` with *video*, *images*, *database* and so on...<br>
            **ChatX** could `chat` with any *digital objects* and *agent*.
            ''',unsafe_allow_html=True)
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




async def chatpdf_main():
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
    asyncio.run(chatpdf_main())
