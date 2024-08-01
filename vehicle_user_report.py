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


file_path = (
    r"C:\Users\34491\aws_bedr_genai\DTC\TCU_ DTC_Troubleshooting_Guide_TML_v1.12.pdf"
)
loader = PyPDFLoader(file_path)
pages = loader.load_and_split()
pages = pages[16:]
query= dtc_value
send_bedrock = " "
for i in pages:
    if query in i.page_content:
        x = i.page_content
        send_bedrock += x






aws_access_key_id = 'ASIAU6GDXWDTZPINN7NV'
aws_secret_access_key = 'GOCjnDCzrpYNRVztjvEZBOZSWVw8p3lL/rWhvwn9'
aws_session_token = 'IQoJb3JpZ2luX2VjEG8aCmFwLXNvdXRoLTEiRzBFAiEApW54MQ52VTHTL5nhWJiHTyIsh7uiVr2MbzQdi7pz/7gCIBoJHDwy0SwJ/wv7AvxGC5Hu1SIIVmErBEWKTnO3PP+mKogDCFgQABoMMzM5NzEyOTEzNjM5IgzRnH1LFbNQWlkOR3oq5QJ8caPrXoaegCgqfXXV61FXfiZ38oM275DP84XqcWWLPW/pWQYgL4Zx8WhV2p0xvYdnUWWqPLGHfBm+U8WHZGuHyJZn9BCHt+35HHbnUN0da9HaxJhz+WPdPTMpc5lzLBXZq2z7f7M4pMXvowXvRL6g1oLVn/0vgCeUHnjUutYms30CiuE7ou4onyrs7sAUCfuKQg/SsScpq3cN44kUYzdwi20lLIMNGnUpvIfKhgeF4UZm1cg+51ZC0lEESKM7MzqjlKWfkel5tH3b9MR+WdD3Y0WSakOc6fwesP2xsnpaNWoSdPJDp0gNhG23qstWVtBNbA3ziONLq3gLaqw2uSkfXdq5vKe61YDc3qir7rnsgNuYyGtQ7AjBojAHQNd3xL137uOp+VhaYmHjbU9zMZrFjUxje97XRO0oxlfkI2BEFNIXpIRFEvoKtlQJalogz6LkF264rKYY2ys6HhSc+YH8PhuxkW8w6LentQY6pgFSo//GEWc8S5fFjirPVkE2wdBZqAur6USnT5ywlgqvT8HvrvWT52Kv/jen93SxPkI8MhXdWRN16QCIgHAJyfspUYGEZaCm88/e+MxWaEYPwMgeUpI261OZMcSZJEu7rMkc3XQ1pX0T2wMMcdRUEiAl1y2PTVJJNqUK6aujL/E0D86a4EG5Lcx/eTK2eID7Vv9ew1CzGJ6C3cP3GtNswTE0xJOTtWk6'

bedrock_runtime = boto3.client(service_name="bedrock-runtime")
 
bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name="ap-south-1",
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    aws_session_token=aws_session_token
)

prompt_1 = send_bedrock+"\nUsing the above information of DTC troubleshooting. Give me the probable trouble area and the healing conditions along with DTC info, do not give summary"

prompt_2 = f"{json_object} \nfrom this above file, give me a status report of a vehicle. strictly use only fuel level, tire pressures and DTC code. do not give me details in the json file. Just sentence response in a presentable format."

def find_nearby_places(lat, lon, place_type, radius):
    geolocator = Nominatim(user_agent="nearby_search")
    # location = geolocator.reverse((lat, lon))
    # print(f"\nYour current location: {location}\n")
    
    query = f"{place_type} near {lat}, {lon}"
    results = []
    try:
        places = geolocator.geocode(query, exactly_one=False, limit=None)
        if places:
            for place in places:
                place_coords = (place.latitude, place.longitude)
                place_distance = geodesic((lat, lon), place_coords).kilometers
                if place_distance <= radius:
                    # print(f"{place.latitude}, {place.longitude} ({place_distance:.2f} km)")
                    # return {"Response": f"{place.latitude}, {place.longitude}, ({place_distance:.2f} km)"}
                    results.append({
                        "Address": place.address,
                        "Latitude": place.latitude,
                        "Longitude": place.longitude,
                        "Distance": round(place_distance, 2),
                    })
            if results:
                return{"Response": results}
            else:
                return{"Message": "No petrol stations found nearby 10km radius"}
        
        else:
            return{"Message": "No nearby places found for the given type."}
    except:
        return{"Error": "Unable to fetch nearby places."}


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


while True:
    x = input("Do you want Vehicle Health Report [y/n]: ")
    if x == "y":
        print(f"Here is the quick report: \n{chatmodel(prompt_2)}")
        y = input("Do you want to troubleshoot the DTC [y/n]: ")
        if y == 'y':
            print(f"Here is the DTC troubleshooting info: \n{chatmodel(prompt_1)}")
            break
        
        else:
            print("Exit")
            break
        
        
    else:
        print("Exit")
        break
if int(fuel) <= 30:
    print("Your fuel level is low, here are the list of nearby petrol stations under a 10km radius")
    response = find_nearby_places(lat=latitude, lon=longitude, place_type = "Petrol station", radius = 10)
    print(response)


# print(boto3.__version__)

# db = SentenceTransformerEmbeddings(model_name = 'multi-qa-MiniLM-L6-cos-v1')
# # print(len(pages))
# # print(pages[0])
# # db2 = Chroma.from_documents(pages, db)
# db2 = Chroma(persist_directory="./DTC_db", embedding_function=db)
# query= "Tell me what this B283D code is?"

# # docs_final = db2.similarity_search("how can i activate the child lock?", k=3)
# # print(docs_final.page_content)
# docs_final = db2.as_retriever(k=4)
# print(docs_final.invoke(query))

# with open('vehicle_user_report_data.json', 'r') as openfile:
#     json_object = json.load(openfile)


