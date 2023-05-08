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