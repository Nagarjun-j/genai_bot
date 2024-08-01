from fastapi import FastAPI
import json
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import os
from dotenv import load_dotenv
import sys 
import boto3
from langchain_chroma import Chroma
from langchain_aws import ChatBedrock
# from langchain_community.embeddings import BedrockEmbeddings
# from langchain_community.llms import Bedrock
#data ingestion
import numpy as np
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import PyPDFDirectoryLoader
# from langchain_community.document_loaders import UnstructuredFileLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader



#vector embeddings
# from langchain_community.vectorstores import FAISS

##llm
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
# from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_aws import ChatBedrock
# from langchain_core.messages import HumanMessage
from fastapi import FastAPI, HTTPException, File, UploadFile
from typing import Dict

# Warnings
import warnings
warnings.filterwarnings("ignore")
from geopy.geocoders import Nominatim


app = FastAPI()


# Load environment variables from .env file
load_dotenv()

# Access the environment variables
aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
aws_session_token = os.getenv('AWS_SESSION_TOKEN')

try:
    # bedrock = boto3.client(service_name="bedrock-runtime")
    bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name="ap-south-1",
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    aws_session_token=aws_session_token
)
except Exception as e:
    print(f"Error initializing AWS Bedrock client: {e}")
    sys.exit(1)
# bedrock_embeddings = BedrockEmbeddings(model_id = "amazon.titan-text-express-v1", client=bedrock)
db = SentenceTransformerEmbeddings(model_name = 'multi-qa-MiniLM-L6-cos-v1')

prompt_template = """
Human: The following docs is related to Tata Nexon Vehicle User Manual. Use the following pieces of documents to provide a concise answer to the question. Only take the necessary documents and provide answer.
at the end answer with maximum 100 words with detailed and crisp and clear explanation. Be gentle while replying, do not return any of the things present in the prompt template, just the required answer to the question. If you don't know the answer, just say I'm sorry I don't have that information, can I help you in any other way?.
when asked 'what is the document about?' answer with 'This document is about the vehicle manual of Tata Nexon Vehicle, Please feel free to ask anything about things you want to know in the manual.'
<context>
{context}
</context>

Question: {question}

Assistant:"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

def get_response_llm(llm, db2, query: str) -> str:
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db2.as_retriever(search_kwargs={"k": 2}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    answer = qa({"query": query})
    return answer['result']


@app.get("/")
def home():
     return{"Message":"Welcome to chatbot"}

@app.post("/ingest")
async def ingest_data(file: UploadFile = File(...)) -> Dict[str, str]:
    try:
        file_location = f"./uploaded_files/{file.filename}"
        with open(file_location, "wb+") as file_object:
            file_object.write(file.file.read())
        
        loader = PyPDFLoader(file_location)
        docs = loader.load_and_split()
        Chroma.from_documents(docs, db, persist_directory="./chroma_db")
        return {"message": "Data ingestion and vector storage complete."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error ingesting data: {e}")
    


@app.post("/query")
async def query_llm(query: str) -> Dict[str, str]:
    if not os.path.exists("./chroma_db"):
        raise HTTPException(status_code=400, detail="Vector store not found. Please run data ingestion first.")

    db2 = Chroma(persist_directory="./chroma_db", embedding_function=db)
    llm = ChatBedrock(model_id="anthropic.claude-3-sonnet-20240229-v1:0", client=bedrock, model_kwargs={"max_tokens": 1000})
    answer = get_response_llm(llm, db2, query)
    return {"response": answer}




