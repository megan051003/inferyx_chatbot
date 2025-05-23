import json
import requests
from bs4 import BeautifulSoup
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

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

def build_index():
    links = load_links()
    docs = []

    for link in links:
        doc = scrape_content(link)
        if doc:
            docs.append(doc)

    print(f"Scraped {len(docs)} documents.")

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(docs)

    db = FAISS.from_documents(split_docs, OpenAIEmbeddings())
    db.save_local("inferyx_index")

if __name__ == "__main__":
    build_index()
