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


load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')  

async def storeDocEmbeds(file, filename):
    reader = PdfReader(file)
    corpus = ''.join([p.extract_text() for p in reader.pages if p.extract_text()])
    
    splitter =  RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200,)
    chunks = splitter.split_text(corpus)
    
    embeddings = OpenAIEmbeddings(openai_api_key = api_key)
    vectors = FAISS.from_texts(chunks, embeddings)
    
    with open(filename + ".pkl", "wb") as f:
        pickle.dump(vectors, f)

async def getDocEmbeds(file, filename):
    if not os.path.isfile(filename + ".pkl"):
        await storeDocEmbeds(file, filename)
    
    with open(filename + ".pkl", "rb") as f:
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
