import asyncio
from playwright.async_api import async_playwright
import json

BASE_URL = "https://inferyx.atlassian.net"
ALL_CONTENT_URL = f"{BASE_URL}/wiki/spaces/IID/pages"

async def extract_all_doc_links():
    all_links = set()
    previous_count = -1

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)  # Set False so you see what's happening
        page = await browser.new_page()
        print("▶️ Going to", ALL_CONTENT_URL)
        await page.goto(ALL_CONTENT_URL)

        # Initial wait for page content to fully load
        await page.wait_for_timeout(7000)

        print("🔁 Starting to scroll and collect document links...")

        for round_num in range(100):
            # Scroll down a bit slower
            print(f"⏳ Scrolling round {round_num + 1}...")
            await page.mouse.wheel(0, 2000)  # smaller scroll step than before
            await page.wait_for_timeout(5000)  # longer wait for loading

            # Try clicking "Load more" button if it exists (optional)
            try:
                load_more = page.locator("button:has-text('Load more')")
                if await load_more.is_visible():
                    print(f"👆 Clicking Load More button on round {round_num + 1}")
                    await load_more.click()
                    await page.wait_for_timeout(6000)
            except Exception as e:
                # If no load more button or error, just continue
                pass

            # Collect anchors again
            anchors = await page.locator('a[href^="/wiki/spaces/IID/pages/"]').all()
            new_links = 0
            for a in anchors:
                href = await a.get_attribute("href")
                full_url = BASE_URL + href
                if full_url not in all_links:
                    all_links.add(full_url)
                    new_links += 1

            print(f"🔄 Round {round_num + 1}: Found {len(all_links)} links (+{new_links})")

            if len(all_links) == previous_count:
                print("✅ No new links after this round — stopping.")
                break
            previous_count = len(all_links)

        await browser.close()

    with open("inferyx_doc_links.json", "w") as f:
        json.dump(list(all_links), f, indent=2)
    print(f"✅ Done. Saved {len(all_links)} links to inferyx_doc_links.json")

asyncio.run(extract_all_doc_links())
