from flask import Flask, request, jsonify
from pymongo import MongoClient
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os


def err_template(err,msg=None,code=None):
    response = {
        "status": "error",
        "message": "Request failed" if not msg else msg,
        "data": [],
        "meta": {},
        "error": {
            "code": code,
            "message": err,
            "details": None
        }
    }
    return response


def success_template(text):
    response = {
        "status": "success",
        "message": "Request processed successfully",
        "data": {
            "text": text
        },
        "meta": {
            "nodeCount": None,
            "edgeCount": None,
            "total": None,
            "type": "TEXT",
            "query": None
        },
        "error": {
            "code": None,
            "message": None,
            "details": None
        }
    }
    return response



os.environ["GOOGLE_API_KEY"] = "AIzaSyBkfcgoZZ8AhsdjDtjiQiFIuZcHxHUrSoI"
print("[INFO] Set Google API key")

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    task_type="semantic_similarity"
)
print("[INFO] Initialized Google Generative AI Embeddings")

def retriever(app_uuid,collection,search_string):
    try:



        vectorstore_path = f"/app/framework/index/{app_uuid}"
        print(f"[INFO] Vectorstore path: {vectorstore_path}")




        # Load FAISS index
        print("[INFO] Loading FAISS index...")
        faiss_index = FAISS.load_local(
            vectorstore_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
        print("[INFO] FAISS index loaded successfully")

        if collection == 'Unknown':
            print(f'No Filter Selected ')
            results = faiss_index.similarity_search_with_relevance_scores(search_string, k=10)
            print(f"[INFO] Retrieved {len(results)} results from FAISS")
        else:
            print(f' {collection} Filter Selected ')
            results = faiss_index.similarity_search_with_relevance_scores(search_string, k=10, filter={"collection": {"$in": collection}})
            print(f"[INFO] Retrieved {len(results)} results from FAISS")
        # Filter results with score > 0.7 and deduplicate (uuid, version)
        unique_keys = set()
        structured = []
        for doc, score in results:
            print(f"[DEBUG] Score: {score}, Metadata: {doc.metadata}")
            if score > 0.6:
                meta = doc.metadata
                key = (meta.get("uuid"), meta.get("version"))
                if key not in unique_keys:
                    unique_keys.add(key)
                    structured.append({
                        "uuid": meta.get("uuid"),
                        "version": meta.get("version"),
                        "metaType": meta.get("collection")
                    })

        print(f"[INFO] Filtered {len(structured)} unique results with score > 0.6")

        # Connect to MongoDB
        print("[INFO] Connecting to MongoDB...")
        client = MongoClient(
            "localhost",
            27017,
            username="admin",
            password="20Admin19",
            authSource="admin",
            authMechanism="SCRAM-SHA-256"
        )
        print("[INFO] Connected to MongoDB")
        db = client["framework"]

        # Fetch unique full documents with active="Y"
        final_data = []
        seen_ids = set()
        for item in structured:
            print(f"[DEBUG] Querying MongoDB with: {item}")
            query = {
                "uuid": item["uuid"],
                "version": item["version"],
                "active": "Y"
            }

            if collection != "Unknown":
                mongo_collection = db[collection]
            else:
                mongo_collection = db[item["metaType"]]

            docs = list(mongo_collection.find(query))
            print(f"[INFO] Retrieved {len(docs)} documents from MongoDB")


            for doc in docs:
                doc_id_str = str(doc["_id"])
                if doc_id_str not in seen_ids:
                    doc["_id"] = doc_id_str  # Convert ObjectId to string for JSON serialization
                    final_data.append(doc)
                    seen_ids.add(doc_id_str)

        print(f"[INFO] Returning {len(final_data)} unique documents from {collection} as final result from ")
        return success_template(final_data)

    except Exception as e:
        print(f"[ERROR] Exception occurred: {str(e)}")
        response= err_template(str(e),"somthing went wrong",500)
        return  response
