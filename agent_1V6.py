import os
import sys
import re
import pandas as pd
import logging
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# ✅ Suppress noisy logs
logging.getLogger().setLevel(logging.WARNING)
logging.getLogger("_client").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("inferyx.components.data_preparation").setLevel(logging.CRITICAL)
logging.getLogger("data_preparation").setLevel(logging.CRITICAL)

# ✅ Inferyx modules
sys.path.insert(0, '/app/framework/script/module/src')
from inferyx.components.data_preparation import AppConfig, Datapod

# ✅ Load environment variables
load_dotenv("/app/framework/test/env.txt")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
INFERYX_HOST = os.getenv("INFERYX_HOST", "dev.inferyx.com")
INFERYX_APP_TOKEN = os.getenv("INFERYX_APP_TOKEN", "OwWji5rBNaSNJoJhItCYjQ4wTdPLUmluqOlqXs2k")
INFERYX_ADMIN_TOKEN = os.getenv("INFERYX_ADMIN_TOKEN", "iresTHOb208NrFOuLbdrgNNYuUNHYOrCyeQRrISL")
FILE_PATH = "/app/framework/upload/dummy.csv"


# ✅ Check for file
if not Path(FILE_PATH).exists():
    print(f"[❌] CSV file not found at: {FILE_PATH}")
    sys.exit(1)

# ✅ Load CSV to get schema
df = pd.read_csv(FILE_PATH)
column_names = df.columns.tolist()
print(column_names)

# ✅ Initialize LLM
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model="llama3-8b-8192"
)

chat_history = []


system_prompt = f"""
You are a helpful assistant that helps the user create or delete a datapod for a CSV file.

1. First, determine whether the user wants to **create** or **delete** a datapod.

2. For **create**, user should give:
   - datapod name (e.g. sales_data)
   - a PK (primary key) from the column names below

Columns: {column_names}

Important: ONLY ASK for name or PK if user DIDN'T give. If they already gave name or PK , don’t ask again. 

Important: Also, don't allow user to choose a PK not in columns , keep telling them to pick from columns b/c PK has to be in columns

Also, suggest primary keys from column u think fit best if user needs help 

3. For **delete**, user should only give the datapod name. Ask if missing. The name of the datapod the user wants to delete doesn't have to be known to you, they didn't have to create it with u to delete it. ASSUME IT EXISTS ALWAYS

3. For **read**, user should only give the datapod name. Ask if missing. The name of the datapod the user wants to read doesn't have to be known to you, they didn't have to create it with u to read it. ASSUME IT EXISTS ALWAYS


Respond in a conversational tone, short & clear. 

When ready with all parameters, output final command: DO NOT SAY ANYTHING AFTER THIS COMMAND 
create datapod_name PK_name
delete datapod_name
read datapod_name

⚠️ Final command must be on the last line, no code blocks, no quotes, no prefix.
"""

app_config = AppConfig(
    host=INFERYX_HOST,
    appToken=INFERYX_APP_TOKEN,
    adminToken=INFERYX_ADMIN_TOKEN
)

user_values = {"action": None, "name": None, "pk": None}

def extract_fields_conversational(user_input: str) -> Optional[str]:
    chat_history.append(HumanMessage(content=user_input))

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        *[("human" if isinstance(m, HumanMessage) else "ai", m.content) for m in chat_history]
    ])

    chain = prompt | llm
    response = chain.invoke({})
    assistant_reply = response.content.strip()
    chat_history.append(AIMessage(content=assistant_reply))

    print("Assistant:", assistant_reply)

    last_line = assistant_reply.strip().split("\n")[-1]

    create_match = re.match(r"^create\s+(\w+)\s+(\w+)$", last_line, re.IGNORECASE)
    if create_match:
        user_values["action"] = "create"
        user_values["name"] = create_match.group(1)
        user_values["pk"] = create_match.group(2)

    delete_match = re.match(r"^delete\s+(\w+)$", last_line, re.IGNORECASE)
    if delete_match:
        user_values["action"] = "delete"
        user_values["name"] = delete_match.group(1)

    read_match = re.match(r"^read\s+(\w+)$", last_line, re.IGNORECASE)
    if read_match:
        user_values["action"] = "read"
        user_values["name"] = read_match.group(1)

    # ✅ Create Datapod
    if user_values["action"] == "create" and user_values["name"] and user_values["pk"]:
        print(f"\n✅ Proceeding to create datapod '{user_values['name']}' with PK '{user_values['pk']}'\n")
        try:
            datapod = Datapod(
                app_config=app_config,
                name=user_values["name"],
                datasource="mysql_framework_aml",
                file_path=FILE_PATH,
                primary_key=user_values["pk"],
                desc="Created via CLI chatbot",
                keyType="PHYSICAL"
            )
            datapod.create_datapod()
        except Exception as e:
            error_msg = str(e).lower()
            if "already exists" in error_msg:
                print(f"⚠️ Datapod with name '{user_values['name']}' already exists, choose a new one.")
            elif "primary key" in error_msg:
                print(f"❌ Invalid PK '{user_values['pk']}'. Please choose a valid column.")
            else:
                print("❌ Something went wrong while creating the datapod.")
        reset_user_values()
        return

    # ✅ Delete Datapod
    elif user_values["action"] == "delete" and user_values["name"]:
        print(f"\n🗑️ Proceeding to delete datapod '{user_values['name']}'\n")
        try:
            datapod = Datapod(
                app_config=app_config,
                name=user_values["name"],
                datasource="mysql_framework_aml",
                file_path=FILE_PATH,
                primary_key="id",  # doesn't matter for delete
                desc="",
                keyType="PHYSICAL"
            )
            datapod.delete()
        except Exception as e:
            print(f"❌ Could not delete datapod: {str(e)}")
        reset_user_values()
        return

    # ✅ Read Datapod
    elif user_values["action"] == "read" and user_values["name"]:
        print(f"\n📖 Proceeding to read datapod '{user_values['name']}'\n")
        try:
            datapod = Datapod(
                app_config=app_config,
                name=user_values["name"],
                datasource="mysql_framework_aml",
                file_path=FILE_PATH,
                primary_key="id",  # doesn't matter for read
                desc="",
                keyType="PHYSICAL"
            )
            datapod.read()
        except Exception as e:
            print(f"❌ Could not read datapod: {str(e)}")
        reset_user_values()
        return

def reset_user_values():
    user_values["action"] = None
    user_values["name"] = None
    user_values["pk"] = None

# ✅ Start CLI chat
if __name__ == "__main__":
    print("💬 Talk to the CLI to create/read/delete a datapod based on your CSV.\n")
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() in {"exit", "quit"}:
            print("👋 Exiting. Bye!")
            break
        extract_fields_conversational(user_input)
