import os
import sys
import re
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import logging

# ✅ Suppress noisy logs from third-party libraries
logging.getLogger().setLevel(logging.WARNING)
logging.getLogger("_client").setLevel(logging.ERROR)  # Just in case "_client" is the source
logging.getLogger("httpx").setLevel(logging.WARNING)  # Suppress HTTP logs

# ✅ Inferyx modules
sys.path.insert(0, '/app/framework/script/module/src')
from inferyx.components.data_preparation import AppConfig, Datapod

# ✅ Load environment variables
load_dotenv("/app/framework/test/env.txt")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
INFERYX_HOST = os.getenv("INFERYX_HOST", "dev.inferyx.com")
INFERYX_APP_TOKEN = os.getenv("INFERYX_APP_TOKEN", "OwWji5rBNaSNJoJhItCYjQ4wTdPLUmluqOlqXs2k")
INFERYX_ADMIN_TOKEN = os.getenv("INFERYX_ADMIN_TOKEN", "iresTHOb208NrFOuLbdrgNNYuUNHYOrCyeQRrISL")
FILE_PATH = os.getenv("INFERYX_FILE_PATH", "/app/framework/upload/dummy.csv")

# ✅ Check for file
if not Path(FILE_PATH).exists():
    print(f"[❌] CSV file not found at: {FILE_PATH}")
    sys.exit(1)

# ✅ Load CSV to get schema
df = pd.read_csv(FILE_PATH)
column_names = df.columns.tolist()

# ✅ Initialize LLM
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model="llama3-8b-8192"
)

chat_history = []

system_prompt = f"""
You are a helpful assistant that helps the user create or delete a datapod for a CSV file.

1. First, determine whether the user wants to **create** or **delete** a datapod.
2. For **create**, ask for:
   - the datapod name (e.g. sales_data)
   - a suggested primary key from the column names below, or ask user to pick one:

   Columns: {column_names}

3. For **delete**, just ask for the datapod name.

Respond conversationally but at the same time don't overtalk and ask what's missing like below

What do you want your datapod name to be?
What primary key do you want [insert datapod name  or if missing just datapod] to be. If you are unsure, let me know , I can suggest.

5. just suggest 1 primary key and if user doesn't liek then suggest another. Suggest the best one without explaining ; just say "I suggest [PK_name] as the primary key, what do you think? 

once all info is given, FOLLOW EXACT FORMAT OUTPUT, NOTHING MORE OR LESS

format/template :

create datapod_name primary_key_name
"""

app_config = AppConfig(
    host=INFERYX_HOST,
    appToken=INFERYX_APP_TOKEN,
    adminToken=INFERYX_ADMIN_TOKEN
)

user_values = {
    "action": None,
    "name": None,
    "pk": None,
}

def extract_fields_conversational(user_input: str) -> Optional[str]:
    chat_history.append(HumanMessage(content=user_input))

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        *[("human" if isinstance(m, HumanMessage) else "ai", m.content) for m in chat_history]
    ])

    chain = prompt | llm
    response = chain.invoke({})

    chat_history.append(AIMessage(content=response.content))
    print("Assistant:", response.content)


# ✅ Start interaction
if __name__ == "__main__":
    print("💬 Talk to the CLI to create/delete a datapod based on your CSV.\n")
    while True:
        user_input = input("You: ")
        result = extract_fields_conversational(user_input)
        if result == "done":
            break
