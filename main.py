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
# import numpy as np
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
from typing import Any, Dict
from botocore.exceptions import ClientError
import pandas as pd
from langchain_community.document_loaders.csv_loader import CSVLoader
import requests
import re




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




prompt_template_1 = """
Human: The following docs is related to the data received in real time from the vehicle.Use the measure name to match the question asked by user. Use the following pieces of documents to provide a concise answer to the question. Only take the necessary documents and provide answer.The answer shuould be either yes/no, or the value of the measure value.Be gentle while replying, do not return any of the things present in the prompt template, just the required answer to the question. Assume SI units accordingly if necessary. If you don't know the answer, just say I'm sorry I don't have that information, can I help you in any other way?.
<context>
{context}
</context>

Question: {question}

Assistant:"""

PROMPT_1 = PromptTemplate(template=prompt_template_1, input_variables=["context", "question"])

with open('records_remote_commands.json', 'r') as openfile:
    json_object_commands = json.load(openfile)
    
fuel = None
row1_lt = None
row1_rt = None
row2_lt = None
row2_rt = None

for i in json_object_commands['Data']:

    x = str(i)
    if "FuelSystem.AbsoluteLevel" in x:
        fuel = i['measureValue']
    if "Row1.Wheel.Left.Tire.Pressure" in x:
        row1_lt = i['measureValue']
    if "Row1.Wheel.Right.Tire.Pressure" in x:
        row1_rt = i['measureValue']
    if "Row2.Wheel.Right.Tire.Pressure" in x:
        row2_rt = i['measureValue']
    if "Row2.Wheel.Left.Tire.Pressure" in x:
        row2_lt = i['measureValue']

# latitude = "12.993734369830559"
# longitude = "77.70530673649098"
# dtc_value = "B2840"
# fuel = "35"

# file_path = (
#     r"C:\Users\34491\aws_bedr_genai\DTC\TCU_ DTC_Troubleshooting_Guide_TML_v1.12.pdf"
# )

# loader = PyPDFLoader(file_path)
# pages = loader.load_and_split()
# pages = pages[16:]
# query= dtc_value
# send_bedrock = " "
# for i in pages:
#     if query in i.page_content:
#         x = i.page_content
#         send_bedrock += x

# prompt_1 = send_bedrock+"\nUsing the above information of DTC troubleshooting. Give me the probable trouble area and the healing conditions along with DTC info, do not give summary"

prompt_2 = f"fuel level of the car is {fuel}, Front left tire pressure is {row1_lt}, Front right tire pressure is {row1_rt}, Rear left tire pressure is {row2_lt}, Rear right tire pressure is {row2_rt} \nfrom this above sentence, give me a status report of a vehicle. strictly use only fuel level, tire pressures and DTC code if present if not present say NONE. Just sentence response in a presentable format like a report."


# geolocator = Nominatim(user_agent="GetLoc")

# location = geolocator.reverse(latitude+","+longitude)
 
# address = location.address

    
bedrock_runtime = boto3.client(service_name="bedrock-runtime")

# url = "https://q789819xlj.execute-api.us-west-2.amazonaws.com/dev/cvp/v1/vehicles/data/realTimeData"
# #payload for remote commands
# payload_1 = {
#     "vin": "5NSVDK999JF937434",
#     "interval": {
#         "second": 30
#     },
#     "limit": 1000,
#     "pageNumber":1,
#     "campaignName" : "TML_SANDBOX_SENSORS_TS"
# }
# #payload for alerts
# payload_2 = {
#     "vin": "5NSVDK999JF937434",
#     "interval": {
#         "second": 30
#     },
#     "limit": 1000,
#     "pageNumber":1,
#     "campaignName" : "TML_ALERTS_TS"
# }

# headers = {
#     "x-api-key": "VsWlUhL16R4U0w4i7xJKS8cWSN2ET3Sea05TEC7f",
#     "authorization": "allow",
#     "Content-Type": "application/json"
# }
# response_1 = requests.post(url, headers=headers, json=payload_1)
# response_2 = requests.post(url, headers=headers, json=payload_2)



# data_1 = response_1.json()

# data_2 = response_2.json()
# with open('commands.json', 'w') as f1, open('alerts.json', 'w') as f2:
#     json.dump(data_1, f1)
#     json.dump(data_2, f2)
# with open('commands.json', 'r') as openfile:
#     json_object_1 = json.load(openfile)

# with open("alerts.json", 'r') as openfile:
#     json_object_2 = json.load(openfile)

# json_object = json_object_1['Data']+json_object_2['Data']
# df = pd.json_normalize(json_object)
# df.to_csv(r"C:\Users\34491\aws_bedr_genai\json_file.csv")
# df_csv = pd.read_csv("json_file.csv")
# loader = CSVLoader(file_path=r"C:\Users\34491\aws_bedr_genai\json_file.csv")
# docs = loader.load()
# Chroma.from_documents(docs, db, persist_directory="./commands_alerts")
# db2 = Chroma(persist_directory="./commands_alerts", embedding_function=db)
# llm = ChatBedrock(model_id="anthropic.claude-3-sonnet-20240229-v1:0", client=bedrock, model_kwargs={"max_tokens": 1000})

# realtime_qa = RetrievalQA.from_chain_type(
#         llm=llm,
#         chain_type="stuff",
#         retriever=db2.as_retriever(search_kwargs={"k": 10}),
#         return_source_documents=True,
#         chain_type_kwargs={"prompt": PROMPT}
#     )
# with open('records_remote_commands.json', 'w') as f1:
#     json.dump(json_object, f1)
# with open('records_remote_commands.json', 'r') as openfile:
#     json_object_commands = json.load(openfile)
    
# fuel = None
# row1_lt = None
# row1_rt = None
# row2_lt = None
# row2_rt = None

# for i in json_object_commands['Data']:

#     x = str(i)
#     if "FuelSystem.AbsoluteLevel" in x:
#         fuel = i['measureValue']
#     if "Row1.Wheel.Left.Tire.Pressure" in x:
#         row1_lt = i['measureValue']
#     if "Row1.Wheel.Right.Tire.Pressure" in x:
#         row1_rt = i['measureValue']
#     if "Row2.Wheel.Right.Tire.Pressure" in x:
#         row2_rt = i['measureValue']
#     if "Row2.Wheel.Left.Tire.Pressure" in x:
#         row2_lt = i['measureValue']

        

def chatmodel(prompt):
    kwargs = {
  "modelId": "anthropic.claude-3-sonnet-20240229-v1:0",
  "contentType": "application/json",
  "accept": "application/json",
  "body": json.dumps({
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 1000,
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": prompt
          }
        ]
      }
    ]
  })
}

    response = bedrock_runtime.invoke_model(**kwargs)
    body = json.loads(response['body'].read())
    return body['content'][0]['text']


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

@app.get("/ask_car_realtime_sample_data")
async def ask_car_llm(query: str) -> Dict[str, str]:
    # promtpt_3 = f"This is the data received fromt the car with which you have to answer from by accessing the values.\n{json_object}\n if info is present that is asked in the question return precise and concise answers taken directly from the values of the data and add units as well, do not say show any part of data in the response, as it is confidential.Follow the above strictly. If not there, just say 'The question you asked for does not seem to be there, can i help you in any other way?'.\n If the question is asking about the location info just say 'car's location is'{address} in a presentable format. \nThe question is {query}"
    # result = chatmodel(promtpt_3)
    # return {"response": result}
    # Chroma.from_documents(docs_csv, db, persist_directory="./commands_alerts")
    with open('records_alerts.json', 'r') as openfile:
        json_object_alerts = json.load(openfile)
        for i in json_object_alerts['Data']:
            x = str(i)
            if "Vehicle.Service.IsServiceDueAlert" in x:
                if i['measureValue'] == "1":
                    answer_service = "Please Note your Service is Due."
    db2 = Chroma(persist_directory="./commands_alerts", embedding_function=db)
    llm = ChatBedrock(model_id="anthropic.claude-3-sonnet-20240229-v1:0", client=bedrock, model_kwargs={"max_tokens": 1000})

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db2.as_retriever(search_kwargs={"k": 10}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT_1}
    )
    llm = ChatBedrock(model_id="anthropic.claude-3-sonnet-20240229-v1:0", client=bedrock, model_kwargs={"max_tokens": 1000})
    # query = "is my service due?"

    answer = qa({"query": query})
    return{"Response ":f"{answer['result']} \n\n\n\n\n\n\n {answer_service}"}

@app.get("/ask_car_realtime")
async def when_testing_endpoint(query: str) -> Dict[str, str]:
    # promtpt_3 = f"This is the data received fromt the car with which you have to answer from by accessing the values.\n{json_object}\n if info is present that is asked in the question return precise and concise answers taken directly from the values of the data and add units as well, do not say show any part of data in the response, as it is confidential.Follow the above strictly. If not there, just say 'The question you asked for does not seem to be there, can i help you in any other way?'.\n If the question is asking about the location info just say 'car's location is'{address} in a presentable format. \nThe question is {query}"
    # result = chatmodel(promtpt_3)
    # return {"response": result}
    if query != "exit":
        answer = realtime_qa({"query": query})
        return{"Response ":answer['result']}
    else:
        open('file1.json', 'w').close()
        open('file2.json', 'w').close()
        return{"Response":"Exiting..."}


@app.get("/ask_car/user_report")
async def user_report_generator():
    user_report = chatmodel(prompt_2)

    return{"Here is user report":user_report}

@app.get("/remote_commands")
async def lock_unlock_commands(query: str) -> Dict[str, Any]:
    payload_unlock = {
    "vin": "5NSVDK999JF937434",
    "receiver": "Vehicle.Cabin.Door.IsLockedCommand",
    "value": {
        "integerValue": 0
    }
}
    payload_lock = {
    "vin": "5NSVDK999JF937434",
    "receiver": "Vehicle.Cabin.Door.IsLockedCommand",
    "value": {
        "integerValue": 1
    }
}
    headers = {
    "x-api-key": "VsWlUhL16R4U0w4i7xJKS8cWSN2ET3Sea05TEC7f",
    "authorization": "allow",
    "Content-Type": "application/json"
}
    if re.search(r'\block\b', query, re.IGNORECASE):
        url = "https://q789819xlj.execute-api.us-west-2.amazonaws.com/dev/v1/vehicles/commands"
        response = requests.post(url, headers=headers, json=payload_lock)
        if response.status_code == 200:
            data_1 = response.json()
            return{"Vehicle Locked":data_1}
        else:
            return{"Failed": response.status_code}
    elif re.search(r'\bunlock\b', query, re.IGNORECASE):
        url = "https://q789819xlj.execute-api.us-west-2.amazonaws.com/dev/v1/vehicles/commands"
        response = requests.post(url, headers=headers, json=payload_unlock)
        if response.status_code == 200:
            data_1 = response.json()
            return{"Vehicle Unlocked":data_1}
        else:
            return{"Failed": response.status_code}
    else:
        return {"Error": "Invalid command. Use 'lock' or 'unlock' in your query."}

