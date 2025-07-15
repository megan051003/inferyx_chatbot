import os
import sys
import re
import pandas as pd
from pathlib import Path
from inferyx.components.data_preparation import AppConfig, Datapod

# --- 1. Inferyx Platform Config ---
INFERYX_HOST = os.getenv("INFERYX_HOST", "dev.inferyx.com")
INFERYX_APP_TOKEN = os.getenv("INFERYX_APP_TOKEN", "OwWji5rBNaSNJoJhItCYjQ4wTdPLUmluqOlqXs2k")
INFERYX_ADMIN_TOKEN = os.getenv("INFERYX_ADMIN_TOKEN", "iresTHOb208NrFOuLbdrgNNYuUNHYOrCyeQRrISL")
FILE_PATH = os.getenv("INFERYX_FILE_PATH", "/app/framework/upload/healthcare_data_1000_records.csv")

if not Path(FILE_PATH).exists():
    print(f"[❌] CSV file not found at: {FILE_PATH}")
    sys.exit(1)

app_config = AppConfig(
    host=INFERYX_HOST,
    appToken=INFERYX_APP_TOKEN,
    adminToken=INFERYX_ADMIN_TOKEN
)

# --- 2. Command Parsers ---
def parse_create_command(user_input):
    match = re.match(r"create me datapod ([\w\d_]+) with primary key ([\w\d_]+)", user_input.lower())
    if match:
        return match.group(1), match.group(2)
    return None, None

def parse_delete_command(user_input):
    match = re.match(r"delete datapod ([\w\d_]+)", user_input.lower())
    if match:
        return match.group(1)
    return None

def parse_read_command(user_input):
    match = re.match(r"read datapod ([\w\d_]+)", user_input.lower())
    if match:
        return match.group(1)
    return None

# --- 3. Chatbot Loop ---
def run_chatbot():
    print("🧠 Inferyx CLI Chatbot (Structured, Rule-Based)")
    print("Supported commands:")
    print("  → create me datapod <name> with primary key <pk>")
    print("  → delete datapod <name>")
    print("  → read datapod <name>")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("🗣️  You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break

        # --- Create ---
        name, pk = parse_create_command(user_input)
        if name and pk:
            print(f"🛠️ Attempting to create datapod '{name}' with primary key '{pk}'...")
            try:
                df = pd.read_csv(FILE_PATH, nrows=1)
                print(f"📋 CSV Columns: {list(df.columns)}")
                if pk not in df.columns:
                    print(f"⚠️ Primary key '{pk}' not found in CSV columns.\n")
                    continue
            except Exception as e:
                print(f"[❌] Failed to read CSV: {e}\n")
                continue

            try:
                print("\n📦 Creating Datapod object...")
                datapod = Datapod(
                    app_config=app_config,
                    name=name,
                    datasource="mysql_framework_aml",
                    file_path=FILE_PATH,
                    primary_key=pk,
                    desc="Created via CLI chatbot",
                    keyType="PHYSICAL"
                )
                print("📦 Datapod object created. Now calling `create_datapod()`...")
                datapod.create_datapod()
                print("✅ Datapod created and data uploaded successfully.\n")
            except Exception as e:
                if "name already exists" in str(e).lower():
                    print(f"⚠️ Datapod '{name}' already exists.\n")
                else:
                    print(f"[❌] Error creating datapod '{name}': {e}\n")

        # --- Delete ---
        elif (name := parse_delete_command(user_input)) is not None:
            print(f"🗑️ Attempting to delete datapod '{name}'...")
            try:
                datapod = Datapod(
                    app_config=app_config,
                    name=name,
                    datasource="mysql_framework_aml",
                    file_path=FILE_PATH,
                    primary_key="id",
                    desc="Created via CLI chatbot",
                    keyType="PHYSICAL"
                )
                datapod.delete()
                print(f"✅ Datapod '{name}' deleted.\n")
            except Exception as e:
                if f"'{name}'" in str(e):
                    print(f"⚠️ Datapod '{name}' does not exist.\n")
                else:
                    print(f"[❌] Error deleting datapod '{name}': {e}\n")

        # --- Read ---
        elif (name := parse_read_command(user_input)) is not None:
            print(f"📖 Attempting to read datapod '{name}'...")
            try:
                datapod = Datapod(
                    app_config=app_config,
                    name=name,
                    datasource="mysql_framework_aml",
                    file_path=FILE_PATH,
                    primary_key="id",
                    desc="Created via CLI chatbot",
                    keyType="PHYSICAL"
                )
                datapod.read(limit=10)
                print(f"✅ Datapod '{name}' read successfully.\n")
            except Exception as e:
                if f"'{name}'" in str(e):
                    print(f"⚠️ Datapod '{name}' does not exist.\n")
                else:
                    print(f"[❌] Error reading datapod '{name}': {e}\n")

        else:
            print("⚠️ Invalid input format. Try a supported command.\n")

# --- 4. Start Chatbot ---
if __name__ == "__main__":
    run_chatbot()
