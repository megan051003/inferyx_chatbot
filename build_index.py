import json
import requests
from bs4 import BeautifulSoup
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import os
import argparse
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file if present

def load_links():
    with open("confluence_links.json", "r") as f:
        return json.load(f)

def scrape_content(url):
    try:
        res = requests.get(url)
        soup = BeautifulSoup(res.text, "html.parser")
        content = soup.get_text()
        title = soup.title.string if soup.title else "Untitled"
        return Document(page_content=content, metadata={"title": title, "url": url})
    except Exception as e:
        print(f"Failed to scrape {url}: {e}")
        return None

def build_index(api_key):
    links = load_links()
    docs = []

    for link in links:
        doc = scrape_content(link)
        if doc:
            docs.append(doc)

    print(f"Scraped {len(docs)} documents.")

    if not docs:
        print("No documents to index. Exiting.")
        return

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    db = FAISS.from_documents(split_docs, embeddings)
    db.save_local("inferyx_index")
    print("Index saved locally as 'inferyx_index'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FAISS index from Confluence links.")
    parser.add_argument("--api_key", help="OpenAI API key")

    args = parser.parse_args()
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "OpenAI API key not provided. Set OPENAI_API_KEY env variable or pass --api_key."
        )

    build_index(api_key)
