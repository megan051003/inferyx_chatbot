Run the chatbot as follows:

1. Download all the files
2. Run link_scrapper.py (This will scrape all the links from the url: https://inferyx.atlassian.net/wiki/spaces/IID/pages)
3. Run fetch_docs.py (This will go into each link created above and fetch all the metadata and organize it into a single file)
4. Run build_index.py (This will take all the data from the file above and store it into a local database)
5. Run chatbot.py (streamlit run chatbot.py) This will open the streamlit interface where you can ask chatbot questions

IMPORTANT: Make sure to create a .env file and put the OpenAI key into it as the py files above need it to run the OpenAI libraries!
Sample is given in repo 
