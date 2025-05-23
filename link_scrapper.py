import asyncio
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
import json

BASE_URL = "https://inferyx.atlassian.net/wiki/spaces/INF/pages/13238391/Inferyx+Documentation"

async def scrape_links():
    links = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(BASE_URL, timeout=60000)

        content = await page.content()
        soup = BeautifulSoup(content, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.startswith("https://inferyx.atlassian.net/wiki/spaces/INF/pages/"):
                links.append(href)

        await browser.close()

    with open("confluence_links.json", "w") as f:
        json.dump(list(set(links)), f, indent=2)

if __name__ == "__main__":
    asyncio.run(scrape_links())
