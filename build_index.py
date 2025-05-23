from dotenv import load_dotenv
import os
import json
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

load_dotenv()
assert os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY not found in environment variables!"

# Make sure this filename matches your actual data file
with open("inferyx_docs.json", "r") as f:
    data = json.load(f)

print(f"Type of data: {type(data)}")
print(f"Sample data (first item): {data[0]}")  # Optional debug print

docs_data = data if isinstance(data, list) else data.get("documents", [])

docs = [Document(page_content=doc["content"], metadata={"title": doc.get("title", "Untitled"), "url": doc.get("url", "")}) for doc in docs_data]

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(docs)
print(f"📄 Total chunks: {len(chunks)}")

embedding_model = OpenAIEmbeddings()
db = FAISS.from_documents(chunks, embedding_model)
db.save_local("inferyx_faiss_index")

print("✅ FAISS index built and saved as 'inferyx_faiss_index'")
