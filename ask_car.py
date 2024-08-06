import json
import os
import sys 
import boto3
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from langchain_chroma import Chroma
from langchain_aws import ChatBedrock
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.llms import Bedrock
#data ingestion
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from unstructured.cleaners.core import clean_extra_whitespace


#vector embeddings
from langchain_community.vectorstores import FAISS

##llm
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage

# Warnings
import warnings
warnings.filterwarnings("ignore")

from langchain_community.document_loaders import PyPDFLoader

aws_access_key_id = 'ASIAU6GDXWDTQEJUY6ZO'
aws_secret_access_key = '45I2XgZHLj/ej/A/z5p1CaR1EhFM5w3qcHYdZVgw'
aws_session_token = 'IQoJb3JpZ2luX2VjEKH//////////wEaCmFwLXNvdXRoLTEiRzBFAiEAie1VOzKGBs3urCac5uHXwdO8onubyFEdZOfse4TMXCsCIEUflrgNteNTTEgKG2G4OlpyM1xjXrNEoG0/VF+ZloUKKpEDCIr//////////wEQABoMMzM5NzEyOTEzNjM5IgxOXySGyPWDsOG7xAAq5QKIaIBlXDy1rrIgKpJj4Ue2ZfM5mU5A/Z0AZes6dqHhcamBUKJMh/GlYpTXYAe6XGjSWAjy9ZyqvGrPlXYvcTEfyZFQeunXfMoDIeVjGQGlCm14PRmjJ2nepM/mFzosLingMeEGs5EnhixIo3RYyGXhoOUd9stxYO8XNvIiK17YaSC8URR+UbXTdVc90USKHG+ieUohNEPk+wn7hCBZEC7JkBHmguj+gevRgFyraJAw6u+lp0gQ96XsVqU4FI5y7rsGZ9QqYQmrmxIZChtLY3M6UQJ43Xt7vPBH4fk+RSqqqib4u6cLfy1Cetv+xeLZyp1RLT4HW+pgv8xuaiP3lTZN8H8K9SONoEfATE8zRSZd3lx8J5zRufYznpEYvjZ5dDFjelnRKG+w75K9Lw9L15F0R/QgHucwEOe1urQWVYP7npHXsEy/8gpn7t/BJJdssSdV6Rw4EfQspeyR4LJPlj1QRQMfEgQwgb+ytQY6pgHbmkzFSycNDdP1cG7tQrb5JMq0ipncDEVmLYUs6mg/CIMmOG2iRPVsDo84SWFKxOrlCSUIReLakom/5mf4SRVvCnrdhahYYtXYgalvNMVdgHV+5Ci4wXikd7ekkj7n12B+VCMFp3PSc+oCQoQQBBM+K3kYFKwOm0ehmS0F2H5lQE+Z9gIH3t7VuM1Pf3bmPUjm2szqbtb4dhia/2hyP916ZENV/2/N'

bedrock_runtime = boto3.client(service_name="bedrock-runtime")
 
bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name="ap-south-1",
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    aws_session_token=aws_session_token
)


with open('vehicle_user_report_data.json', 'r') as openfile:
    json_object = json.load(openfile)
    
for i in json_object['Data']:
    x = str(i)
    if "Latitude" in x:
        latitude = i['measureValue']
    elif "Longitude" in x:
        longitude = i['measureValue']
    elif "DTC" in x:
        dtc_value = i['measureValue']
    elif "FuelSystem" in x:
        fuel = i["measureValue"]

geolocator = Nominatim(user_agent="GetLoc")

location = geolocator.reverse(latitude+","+longitude)
 
address = location.address


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

def chat():
    x = input("Enter the query: ")
    promtpt_3 = f"This is the data received fromt the car with which you have to answer from by accessing the values.\n{json_object}\n if info is present that is asked in the question return precise and concise answers taken directly from the values of the data and add units as well, do not say show any part of data in the response, as it is confidential.Follow the above strictly. If not there, just say 'The question you asked for does not seem to be there, can i help you in any other way?'.\n If the question is asking about the location info just say 'car's location is'{address} in a presentable format. \nThe question is {x}"
    result = chatmodel(promtpt_3)
    return result

while True:
    chat_def = chat()
    print(chat_def)
    # print(promtpt_3)

    