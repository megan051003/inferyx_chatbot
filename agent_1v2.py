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

# Inferyx modules
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
4. Once all required inputs are gathered, clearly summarize the action.

Respond conversationally but at the same time don't overtalk and ask what's missing like below

What do you want your datapod name to be?
What primary key do you want [insert datapod name  or if missing just datapod] to be. If you are unsure, let me know , I can suggest.

5. just suggest 1 primary key and if user doesn't liek then suggest another. Suggest the best one without explaining ; just say "I suggest [PK_name] as the primary key, what do you think? 


in the end say the function use wants to do: create or delete with details neededd for each as a final statement which can be passed to .create_datapod() or .delete()

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

    # Try to extract fields from response using regex
    if "creating datapod" in response.content.lower():
        name_match = re.search(r"datapod ['\"]?(\w+)['\"]?", response.content)
        pk_match = re.search(r"primary key[:\s]*['\"]?(\w+)['\"]?", response.content)
        if name_match:
            user_values["action"] = "create"
            user_values["name"] = name_match.group(1)
        if pk_match:
            user_values["pk"] = pk_match.group(1)

    elif "deleting datapod" in response.content.lower():
        name_match = re.search(r"datapod ['\"]?(\w+)['\"]?", response.content)
        if name_match:
            user_values["action"] = "delete"
            user_values["name"] = name_match.group(1)

    # If all fields are collected, act
    if user_values["action"] == "create" and user_values["name"] and user_values["pk"]:
        print(f"\n✅ Proceeding to create datapod '{user_values['name']}' with PK '{user_values['pk']}'\n")
        datapod = Datapod(
            app_config=app_config,
            name=user_values["name"],
            datasource="mysql_framework_aml",
            file_path=FILE_PATH,
            primary_key=user_values["pk"],
            desc="Created via CLI chatbot",
            keyType="PHYSICAL"
        )
        response = datapod.create()
        print(f"📦 Datapod Created: {response}")
        return "done"

    elif user_values["action"] == "delete" and user_values["name"]:
        print(f"\n🗑️ Proceeding to delete datapod '{user_values['name']}'\n")
        datapod = Datapod(
            app_config=app_config,
            name=user_values["name"],
            datasource="mysql_framework_aml",
            file_path=FILE_PATH,
            primary_key="id",  # dummy
            desc="",
            keyType="PHYSICAL"
        )
        response = datapod.delete()
        print(f"🧹 Datapod Deleted: {response}")
        return "done"

    return None


# ✅ Start interaction
if __name__ == "__main__":
    print("💬 Talk to the CLI to create/delete a datapod based on your CSV.\n")
    while True:
        user_input = input("You: ")
        result = extract_fields_conversational(user_input)
        if result == "done":
            break
