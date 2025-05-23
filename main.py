echo "🔍 Scraping Inferyx Confluence links..."
python3 scraper/scrape_links.py

echo "📚 Building vector index..."
python3 indexer/build_index.py

echo "💬 Launching Streamlit chatbot..."
streamlit run app/chatbot.py
