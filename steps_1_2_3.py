import os
import json
import asyncio
from time import sleep
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from playwright.async_api import async_playwright

docs_url = "https://inferyx.atlassian.net"
ALL_CONTENT_URL = f"{docs_url}/wiki/spaces/IID/pages"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
assert GROQ_API_KEY, "❌ GROQ_API_KEY not found in .env file!"

# -----------------------------
# STEP 1: SCRAPE LINKS
# -----------------------------
async def extract_all_doc_links():
    all_links = set()
    previous_count = -1

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)  # headless for labs
        page = await browser.new_page()
        print("🔗 Visiting:", ALL_CONTENT_URL)
        await page.goto(ALL_CONTENT_URL)
        await page.wait_for_timeout(7000)

        for i in range(100):
            await page.mouse.wheel(0, 2000)
            await page.wait_for_timeout(5000)
            try:
                load_more = page.locator("button:has-text('Load more')")
                if await load_more.is_visible():
                    await load_more.click()
                    await page.wait_for_timeout(6000)
            except:
                pass

            anchors = await page.locator('a[href^="/wiki/spaces/IID/pages/"]').all()
            for a in anchors:
                href = await a.get_attribute("href")
                if href:
                    all_links.add(BASE_URL + href)

            if len(all_links) == previous_count:
                break
            previous_count = len(all_links)

        await browser.close()

    with open("inferyx_doc_links.json", "w") as f:
        json.dump(list(all_links), f, indent=2)
    print(f"✅ Collected {len(all_links)} links.")

# -----------------------------
# STEP 2: FETCH DOCUMENTS
# -----------------------------
def fetch_docs():
    with open("inferyx_doc_links.json") as f:
        urls = json.load(f)

    docs = []
    for url in urls:
        try:
            res = requests.get(url)
            res.raise_for_status()
            soup = BeautifulSoup(res.text, "html.parser")
            title = soup.title.string if soup.title else "Untitled"
            content_div = soup.find("div", {"id": "main-content"}) or soup.body
            content = content_div.get_text(separator="\n", strip=True) if content_div else ""
            docs.append({"title": title, "url": url, "content": content})
            sleep(1)
        except Exception as e:
            print(f"⚠️ Failed to fetch {url}: {e}")

    with open("inferyx_docs.json", "w") as f:
        json.dump(docs, f, indent=2)
    print(f"✅ Saved {len(docs)} documents.")

# -----------------------------
# STEP 3: BUILD VECTOR INDEX
# -----------------------------
def build_index():
    with open("inferyx_docs.json") as f:
        data = json.load(f)

    docs = [
        Document(
            page_content=doc["content"],
            metadata={"title": doc.get("title", "Untitled"), "url": doc.get("url", "")}
        )
        for doc in data
    ]

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    db = FAISS.from_documents(chunks, embedding_model)
    db.save_local("inferyx_faiss_index")
    print("✅ FAISS index built.")

# -----------------------------
# MAIN EXECUTION (Run Everything)
# -----------------------------
if __name__ == "__main__":
    print("🚀 Starting full pipeline...\n")
    asyncio.run(extract_all_doc_links())
    fetch_docs()
    build_index()
