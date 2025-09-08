from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, SystemMessage
from pydantic import BaseModel
from flask import  jsonify,request
from typing import List
from meta_retriever import retriever
import dotenv

dotenv.load_dotenv()


class MetaItem(BaseModel):
    _id: int
    uuid: str
    name: str
    metaType: str


class MetaDataList(BaseModel):
    data: List[MetaItem]


class InputParser(BaseModel):
    collection: str
    search_string: str

collections = ["alert", "application",
                "batch","caserule", 
               "classificationrule", "dag", "dashboard", "dataarchive", "dataasset",
               "datadomain", "dataglossary", "datapod", "dataproduct", "dataset", "datasource",
               "domain", "dq", "dqgroup",  "entity", "er",
               ,"feature", 
               "function", "graphpod", "ingest", "ingestgroup", "map", "mapgroup", 
               "model",, "organization",
                "predict", "profile", "profilegroup", "recon",
               "recongroup","report", "rule", "rulegroup",  "train",
              "vizpod"]




def search_meta_index():

    print("[INFO] Received POST request at /framework/SearchDataCatalog")

    # Parse request JSON
    data = request.get_json()
    print(f"[DEBUG] Request JSON: {data}")

    search_string = data.get("search_string")
    app_uuid = data.get("app_uuid")


    if not search_string or not app_uuid:
        print("[ERROR] Missing search_string or app_uuid")
        return jsonify({"error": "Missing search_string or app_uuid"}), 400

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2
    )

    prompt = PromptTemplate.from_template(f"""
   You are given a list of available collections in a MongoDB database.

   Your task is to:
   1. Understand the user input.
   2. Identify and extract the following:
      - **collection**: the correct collection name the user is referring to (choose from the available collections only) .
      - **search_string**: identify what user want to search. It should be short
   Note: If you are not sure about the collection then say "Unknown". 
   Available MongoDB collections:
   {collections}

   user_input : {{input}}

   """)

    structured_output = prompt | llm.with_structured_output(InputParser)
    output = structured_output.invoke({'input': search_string})
    meta = output.model_dump()

    collection = meta['collection']
    search_string = meta['search_string']

    if collection not in collections:
        collection = "Unknown"


    data= retriever(app_uuid=app_uuid,search_string=search_string,collection=collection)

    return jsonify(data)
