import os
import json
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings

# Load environment variables from .env file
load_dotenv()

# Get the OpenAI API key from environment variables
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ ERROR: Please set your OPENAI_API_KEY environment variable in the .env file")

# Set your API key for OpenAI
os.environ["OPENAI_API_KEY"] = api_key

# Load your document links from the JSON file
with open("inferyx_doc_links.json", "r") as f:
    links = json.load(f)

print(f"🔗 Loaded {len(links)} document links from inferyx_doc_links.json")

def build_index():
    # Load all documents' text from the links (you should have your scraper here to fetch doc content)
    # For demo, let's pretend you have loaded all documents as strings in a list
    # Replace this with your actual fetching logic
    all_docs_texts = []
    for url in links:
        # Here you should have your scraping or loading logic for the document content by url
        # For now, we just append url as dummy text (replace this)
        all_docs_texts.append(f"Document content fetched from: {url}")

    print(f"Loaded {len(all_docs_texts)} documents.")

    # Split documents into chunks to embed
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = []
    for doc_text in all_docs_texts:
        chunks = text_splitter.split_text(doc_text)
        split_docs.extend(chunks)

    print(f"Split documents into {len(split_docs)} chunks.")

    # Create embeddings instance
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)

    # Build FAISS index
    print("Building FAISS index...")
    db = FAISS.from_texts(split_docs, embeddings)

    # Save index locally
    db.save_local("inferyx_faiss_index")
    print("✅ FAISS index built and saved to 'inferyx_faiss_index' folder.")

if __name__ == "__main__":
    build_index()
