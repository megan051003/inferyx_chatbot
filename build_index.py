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
import time

load_dotenv()  # Load environment variables from .env if present

def load_links(filename="inferyx_doc_links.json"):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Link file {filename} not found. Run the scraper first.")
    with open(filename, "r") as f:
        links = json.load(f)
    print(f"Loaded {len(links)} document links.")
    return links

def scrape_content(url, max_retries=3, timeout=10):
    for attempt in range(1, max_retries + 1):
        try:
            print(f"Scraping {url} (Attempt {attempt})...")
            res = requests.get(url, timeout=timeout)
            res.raise_for_status()
            soup = BeautifulSoup(res.text, "html.parser")
            content = soup.get_text(separator="\n")
            title = soup.title.string.strip() if soup.title else "Untitled"
            return Document(page_content=content, metadata={"title": title, "url": url})
        except Exception as e:
            print(f"Failed to scrape {url}: {e}")
            if attempt < max_retries:
                print("Retrying...")
                time.sleep(2)
            else:
                print("Giving up on this URL.")
                return None

def build_index(api_key, links):
    docs = []
    for url in links:
        doc = scrape_content(url)
        if doc:
            docs.append(doc)
    print(f"Successfully scraped {len(docs)} documents out of {len(links)}.")

    if not docs:
        print("No documents to index. Exiting.")
        return

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(docs)
    print(f"Split documents into {len(split_docs)} chunks.")

    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    db = FAISS.from_documents(split_docs, embeddings)
    db.save_local("inferyx_index")
    print("Index saved locally as 'inferyx_index'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FAISS index from Confluence links.")
    parser.add_argument("--api_key", help="OpenAI API key (optional, can be set in .env)")

    args = parser.parse_args()
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "OpenAI API key not provided. Set OPENAI_API_KEY in your environment or pass --api_key."
        )

    try:
        links = load_links()
    except Exception as e:
        print(f"Error loading links: {e}")
        exit(1)

    build_index(api_key, links)
