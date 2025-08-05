#FINAL FINAL 4 pm August 5th 
import os
import csv
import json
import re
import sys
import pandas as pd  # pandas is imported but not used in this script, consider removing if not needed
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from typing import Optional

# --- FIX for ImportError: Add the correct path to the Inferyx module ---
# This line tells Python to look in your custom module directory first.
sys.path.insert(0, '/Users/mpalrecha/Downloads/module/src')
# ----------------------------------------------------------------------

# Import the real SDK components
from inferyx.components.data_preparation import AppConfig, Datapod

# Load environment variables
load_dotenv("/Users/mpalrecha/Downloads/documents_1/env.txt")

# --- Inferyx SDK Configuration ---
INFERYX_HOST = os.getenv("INFERYX_HOST", "dev.inferyx.com")
INFERYX_APP_TOKEN = os.getenv("INFERYX_APP_TOKEN")
INFERYX_ADMIN_TOKEN = os.getenv("INFERYX_ADMIN_TOKEN")

if not all([INFERYX_HOST, INFERYX_APP_TOKEN, INFERYX_ADMIN_TOKEN]):
    print("Warning: Inferyx SDK credentials not found. Datapod creation will fail.")

app_config = AppConfig(
    host=INFERYX_HOST,
    appToken=INFERYX_APP_TOKEN,
    adminToken=INFERYX_ADMIN_TOKEN
)

# 🧠 In-memory schema and data
global_schema = []
generated_data = []
inferred_data = []
suggested_pk = None
temp_file_path = None


# --- New Tool for Datapod Creation ---
@tool
def create_datapod_from_csv(file_path: str, primary_key: str, datapod_name: str, with_data: bool) -> str:
    """
    Tool: Creates a datapod from a local CSV file using the real SDK.
    Behavior:
    - Takes a file path, primary key field, datapod name, and a boolean flag `with_data`.
    - If `with_data=True`: Calls `datapod.create_datapod()` then `datapod.write()`.
    - If `with_data=False`: Calls only `datapod.create_datapod()`.
    - Clears all in-memory state after creation.
    """
    global global_schema, generated_data, inferred_data, suggested_pk, temp_file_path

    if not os.path.exists(file_path):
        return f"❌ Error: The file '{file_path}' does not exist."

    try:
        # Instantiate the Datapod object
        datapod = Datapod(
            app_config=app_config,
            name=datapod_name,
            datasource="mysql_framework_aml",  # This datasource name is hardcoded in agent_1V6.py
            file_path=file_path,
            primary_key=primary_key,
            desc="Created by agent",
            keyType="PHYSICAL"
        )

        # Create the datapod schema
        datapod.create()

        # If `with_data` is true, write the data
        if with_data:
            datapod.write(mode="OVERWRITE", data_source=file_path)
            sdk_response = f"Datapod '{datapod_name}' created and data written successfully."
        else:
            sdk_response = f"Datapod '{datapod_name}' schema created successfully."

        # Clear in-memory state
        global_schema.clear()
        generated_data.clear()
        inferred_data.clear()
        suggested_pk = None
        temp_file_path = None

    except Exception as e:
        return f"❌ Failed to create datapod: {e}"

    return sdk_response


# --- Existing Tools (Modified for new workflow) ---
@tool
def create_or_edit_schema(user_input: str) -> str:
    """
    Tool: Create or modify a schema using natural language instructions.
    Behavior:
    - If no schema exists: creates one with 5 fields (unless more explicitly requested).
    - If schema exists: updates it by adding or modifying fields (never deletes unless explicitly asked).
    - Stores updated schema in memory.
    - Returns a human-readable list of fields. The agent decides the next step.
    """
    global global_schema, generated_data, inferred_data, suggested_pk

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0,
    )

    if not global_schema:
        prompt = f"""
You are a schema assistant. Create a JSON schema with the 5 most important fields for the topic: "{user_input}".
Only go beyond 5 if the user clearly asks.

Each field must include:
- name
- description
- type (string, integer, float, boolean, date)

Return ONLY valid JSON like this:
[
  {{"name": "field_name", "description": "what it means", "type": "string"}}
]
No explanation. No markdown.
"""
    else:
        current_schema_json = json.dumps(global_schema, indent=2)
        prompt = f"""
You are editing this schema:
{current_schema_json}

Apply the user instruction: "{user_input}".

⚠️ DO NOT remove existing fields unless the instruction *explicitly* asks for it.
Just add or modify fields as required.

Return ONLY valid JSON as a list of fields. No explanation. No markdown.
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    raw_output = response.content.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_output, flags=re.DOTALL)

    try:
        parsed = json.loads(cleaned)
        global_schema = parsed
        generated_data = []
        inferred_data = []

        for field in global_schema:
            if 'id' in field['name'].lower() or 'uuid' in field['name'].lower():
                suggested_pk = field['name']
                break

    except Exception as e:
        return f"❌ Failed to parse schema: {e}\n\nRaw output:\n{raw_output}"

    human_view = "\n".join([
        f"{col['name']} -> {col['description']} ({col['type']})"
        for col in global_schema
    ])

    return f"✅ Schema updated.\n{human_view}"


@tool
def generate_data_from_schema(row_count_str: str = "3") -> str:
    """
    Tool: Generate synthetic data using the current schema.
    Behavior:
    - Uses LLM to generate realistic rows.
    - Accepts an optional row count (default: 3).
    - Appends to memory for later export.
    """
    global global_schema, generated_data, inferred_data, suggested_pk

    if not global_schema:
        return "⚠️ No schema found. Please create one first."

    try:
        row_count = max(1, int(row_count_str.strip()))
    except:
        row_count = 3

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.5,
    )

    schema_desc = json.dumps(global_schema, indent=2)
    prompt = f"""
Generate {row_count} rows of realistic example data for this schema:

{schema_desc}

Return only valid JSON like this:
[
  {{"field1": "value1", "field2": "value2", ...}},
  ...
]
No explanation. No markdown.
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", response.content.strip(), flags=re.DOTALL)

    try:
        rows = json.loads(cleaned)
        if isinstance(rows, list):
            generated_data.extend(rows)
            inferred_data = []
        else:
            return "❌ Data is not in expected list format."
    except Exception as e:
        return f"❌ Error parsing data: {e}\n\nRaw:\n{cleaned}"

    preview = json.dumps(rows[:3], indent=2)

    return f"✅ Generated {len(rows)} rows of data. Here's a preview:\n{preview}\n\nPlease provide a filename to save this data as a CSV. For example, 'my_data.csv'."


@tool
def save_schema_and_data(file_name: str) -> str:
    """
    Tool: Save schema and data into a clean CSV file using schema-defined headers.
    - Takes a filename as input.
    - Saves the file to a predefined path: /Users/mpalrecha/Downloads/documents_1.
    - Uses 'global_schema' for headers.
    - Accepts either 'inferred_data' or 'generated_data'.
    - Outputs a real CSV with headers and values, no metadata.
    """
    global global_schema, inferred_data, generated_data, temp_file_path

    if not global_schema:
        return "❌ No schema found. Please define or infer a schema first."

    data_to_use = generated_data or inferred_data
    if not data_to_use:
        return "❌ No data to save. Please generate or infer data first."

    base_dir = "/Users/mpalrecha/Downloads/documents_1"
    file_path = os.path.join(base_dir, file_name)
    temp_file_path = file_path

    os.makedirs(base_dir, exist_ok=True)

    headers = [field["name"] for field in global_schema]

    sample_row = list(data_to_use)[0]
    if all(k.startswith("Column") for k in sample_row.keys()):
        formatted_data = [
            [row.get(f"Column {i + 1}", "") for i in range(len(headers))]
            for row in data_to_use
        ]
    else:
        formatted_data = [
            [row.get(field, "") for field in headers]
            for row in data_to_use
        ]

    try:
        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(formatted_data)
    except Exception as e:
        return f"❌ Failed to write to file: {e}"

    return f"✅ Schema and data successfully saved to '{file_path}'. What primary key field should I use for the datapod and what should the datapod be named?"


# --- NEW: Autodetect Tool ---
@tool
def autodetect_csv_structure(file_path: str) -> str:
    """
    Tool: Automatically detects the structure of a CSV file and calls the correct tool.
    - Takes a file path.
    - Determines if the file has headers and/or data.
    - Calls `assign_schema_from_csv` if headers are detected.
    - Calls `infer_schema_from_csv` if no headers are detected.
    - Returns the result of the called tool.
    """
    if not os.path.exists(file_path):
        return f"❌ Error: The file '{file_path}' does not exist."

    try:
        with open(file_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)

        if not rows:
            return f"❌ The file '{file_path}' is empty."

        # Heuristic to check for headers
        # Read the first few rows for a robust check
        sample_rows = rows[:5] if len(rows) >= 5 else rows

        first_row = sample_rows[0]
        other_rows = sample_rows[1:]

        has_headers = False

        if not other_rows:
            # If there's only one row, it's hard to tell, but we can assume no headers if it's all text.
            has_headers = any(re.search(r'[a-zA-Z]', cell) for cell in first_row)
        else:
            # Check for type consistency across columns
            num_cols = len(first_row)
            for i in range(num_cols):
                first_cell_is_text = bool(re.search(r'[a-zA-Z]', first_row[i]))
                other_cells_are_numeric = all(
                    re.match(r'^\d+(\.\d+)?$', row[i]) for row in other_rows if len(row) > i
                )

                # If a column's first cell is text but the rest are numbers, it's a strong header indicator.
                if first_cell_is_text and other_cells_are_numeric:
                    has_headers = True
                    break

        if has_headers:
            print(f"✅ Auto-detected headers in '{file_path}'. Calling `assign_schema_from_csv`.")
            return assign_schema_from_csv.invoke(file_path)
        else:
            print(f"✅ Auto-detected no headers in '{file_path}'. Calling `infer_schema_from_csv`.")
            return infer_schema_from_csv.invoke(file_path)

    except Exception as e:
        return f"❌ Failed to read or analyze file: {e}"


@tool
def infer_schema_from_csv(file_path: str) -> str:
    """
    Tool: Infer a schema from a CSV file with no headers.
    - Reads all rows as raw data.
    - Generates generic column names (Column 1, Column 2, etc.).
    - Infers schema with name, description, and type for each column.
    - Stores schema and data in memory.
    """
    global global_schema, generated_data, inferred_data, suggested_pk

    try:
        with open(file_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)

        if not rows:
            return f"❌ The file '{file_path}' is empty or unreadable."

        inferred_data = [dict(zip([f"Column {i + 1}" for i in range(len(row))], row)) for row in rows]
        sample_data = inferred_data[:3]
        column_names = list(sample_data[0].keys())
        sample_data_string = json.dumps(sample_data, indent=2)
    except FileNotFoundError:
        return f"❌ File not found at '{file_path}'."
    except Exception as e:
        return f"❌ Error reading file: {e}"

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0,
    )

    prompt = f"""
I have a CSV file with NO headers.

Here are the generic column names:
{column_names}

And here is a sample of the data:
{sample_data_string}

Please infer a JSON schema for this table. For each column, return:
- name: A descriptive name (replace "Column X" with a better guess if possible)
- description: What kind of data it likely contains
- type: One of string, integer, float, boolean, or date

Return ONLY valid JSON like this:
[
  {{ "name": "field_name", "description": "what it means", "type": "string" }},
  ...
]
No explanation. No markdown.
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    raw_output = response.content.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_output, flags=re.DOTALL)

    try:
        parsed = json.loads(cleaned)
        global_schema = parsed

        for field in global_schema:
            if 'id' in field['name'].lower() or 'uuid' in field['name'].lower() or 'key' in field['name'].lower():
                suggested_pk = field['name']
                break

    except Exception as e:
        return f"❌ Failed to parse schema: {e}\n\nRaw output:\n{raw_output}"

    human_view = "\n".join([
        f"{col['name']} -> {col['description']} ({col['type']})"
        for col in global_schema
    ])

    return f"✅ Schema inferred from '{file_path}'. Here is the suggested schema:\n{human_view}\n\nPlease provide a filename to save this data as a CSV. For example, 'my_data.csv'."



@tool
def assign_schema_from_csv(file_path: str) -> str:
    """
    Tool: Reads the header row from a CSV and creates a schema from it.
    - Assumes the first row contains headers.
    - Creates a basic schema with string types and generic descriptions.
    - Stores the schema and data in memory.
    - This tool is designed to be followed directly by datapod creation.
    """
    global global_schema, generated_data, inferred_data, suggested_pk

    try:
        with open(file_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            headers = next(reader)
            rows = list(reader)
            if rows:
                inferred_data = [dict(zip(headers, row)) for row in rows]
            else:
                inferred_data = []

    except FileNotFoundError:
        return f"❌ File not found at '{file_path}'."
    except StopIteration:
        return f"❌ The file '{file_path}' is empty or has no headers."
    except Exception as e:
        return f"❌ Error reading file: {e}"

    global_schema = [{"name": h.strip(), "description": f"The value for {h.strip()}", "type": "string"} for h in
                     headers]
    generated_data = []

    for h in headers:
        if 'id' in h.lower() or 'uuid' in h.lower() or 'key' in h.lower():
            suggested_pk = h
            break

    human_view = "\n".join([
        f"{col['name']} -> {col['description']} ({col['type']})"
        for col in global_schema
    ])

    if inferred_data:
        return f"✅ Schema loaded from '{file_path}'. Here is the schema:\n{human_view}\n\nData was also found in the file. What primary key field should I use for the datapod and what should the datapod be named?"
    else:
        return f"✅ Schema loaded from '{file_path}'. Here is the schema:\n{human_view}\n\nNo data was found. I can now generate some data for this schema. How many rows would you like to generate?"


# 🤖 Agent setup
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0,
)

tools = [
    create_or_edit_schema,
    generate_data_from_schema,
    save_schema_and_data,
    infer_schema_from_csv,
    assign_schema_from_csv,
    autodetect_csv_structure,
    create_datapod_from_csv
]

agent = create_react_agent(
    model=llm,
    tools=tools
)


def extract_agent_output(result):
    messages = result.get("messages", [])
    for msg in messages:
        if msg.__class__.__name__ == "ToolMessage":
            return msg.content
    for msg in reversed(messages):
        if msg.__class__.__name__ == "AIMessage" and msg.content.strip():
            return msg.content
    return "No response."


def chat():
    print("Welcome! I can help you create a new datapod. (type 'exit' to quit):")

    system_prompt = (
        "You are a helpful assistant for creating datapods. "
        "Your job is to guide the user through a multi-step process. "

        "**PRIMARY Objective:** Ask the user if they want to create a datapod from scratch or from an existing file. "

        "**Workflow 1: Create from Scratch (No CSV):**\n"
        "1. **Start:** If the user says 'from scratch', call `create_or_edit_schema` with the topic (e.g., 'customer data'). "
        "2. **Data Generation:** After the schema is created, ask the user if they want to generate example data. If they say 'yes', call `generate_data_from_schema`. "
        "3. **Saving to Temp CSV:** After the schema is finalized (either by creation or by data generation), you **must** ask the user to provide a new filename to save the data. This will call `save_schema_and_data`, which creates a new CSV with the correct headers. "
        "4. **Datapod Creation:** Once you have the primary key and datapod name, you **must** ask the user if they want to create the datapod with **just the schema** or with the **schema and data**. Once you have their answer, call `create_datapod_from_csv` with the temporary file path, the user-provided primary key, the datapod name, and a boolean `with_data` flag based on their choice. "

        "**Workflow 2: Create from Existing CSV:**\n"
        "1. **Start:** If the user says 'I have a file' and provides a file path, immediately call `autodetect_csv_structure` with the path. "
        "2. **Schema and Data:** The `autodetect_csv_structure` tool will handle everything automatically. "
        "   - **If the CSV had headers and data:** The tool will return the schema and data. You should immediately ask for the primary key and datapod name. "
        "   - **If the CSV had no headers or was empty:** The tool will return with a schema, but a clean file has not yet been created. You **must** ask the user to provide a new filename to save the data with proper headers. This will call `save_schema_and_data`. "
        "3. **Datapod Creation:** Once you have the primary key and datapod name, you **must** ask the user if they want to create the datapod with **just the schema** or with the **schema and data**. Do not proceed to call `create_datapod_from_csv` until you have this explicit answer. "

        "**General Rules:**\n"
        "- The `create_datapod_from_csv` tool now takes a new boolean parameter `with_data`. Use `with_data=True` if the user wants to include data, and `with_data=False` for just the schema."
        "- When the user provides a primary key, check if it exists in the current `global_schema`. If not, ask the user to provide a valid one. "
        "- When a primary key is suggested by a tool (and stored in the `suggested_pk` global variable), use it in your response to the user. "
        "- Ensure you get the primary key, datapod name, and the `with_data` preference before calling `create_datapod_from_csv`."
        "- Your first response to a user providing a file path should be to call the `autodetect_csv_structure` tool. Do not ask them if the file has headers."
    )

    chat_history = [SystemMessage(content=system_prompt)]

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        if not user_input.strip():
            print("Agent: Please provide a valid input.")
            continue

        chat_history.append(HumanMessage(content=user_input))

        try:
            result = agent.invoke({"messages": chat_history})
            agent_response = extract_agent_output(result)
            print("Agent:\n", agent_response)
            chat_history.append(AIMessage(content=agent_response))
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    chat()
