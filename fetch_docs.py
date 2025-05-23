import json
import requests
from bs4 import BeautifulSoup
from time import sleep

# Load the URLs from your links file
with open("inferyx_doc_links.json") as f:
    urls = json.load(f)

docs = []

for url in urls:
    print(f"Fetching {url} ...")
    try:
        res = requests.get(url)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")

        # Extract title - this depends on page structure
        title = soup.title.string if soup.title else "Untitled"

        # Extract main content - adjust selector based on page structure
        # For example, Inferyx docs may have content inside <div id="main-content"> or similar
        content_div = soup.find("div", {"id": "main-content"}) or soup.body
        content = content_div.get_text(separator="\n", strip=True) if content_div else ""

        docs.append({
            "title": title,
            "url": url,
            "content": content
        })

        sleep(1)  # be polite with a delay

    except Exception as e:
        print(f"Error fetching {url}: {e}")

# Save all fetched docs to JSON
with open("inferyx_docs.json", "w") as f:
    json.dump(docs, f, indent=2)

print(f"Saved {len(docs)} documents to inferyx_docs.json")
