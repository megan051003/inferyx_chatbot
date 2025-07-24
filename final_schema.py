import streamlit as st
import os
import pandas as pd
from dotenv import load_dotenv
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from io import StringIO

# === Load API Key ===
load_dotenv("/Users/mpalrecha/Downloads/documents_1/env.txt")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY not found in environment.")
    st.stop()

# === Setup LLM ===
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model="llama3-8b-8192"
)

# === Prompt Setup ===
system_prompt = """
You are a data generation expert that helps users generate sample data in CSV format.

Handle 3 cases:
1. If user gives a schema, generate 100 rows of dummy data.
2. If user gives sample rows, infer schema.
3. If user gives an intent, suggest schema.

Allow user to edit schema (add/remove/rename/change types) before generating.
Only generate CSV when asked to.

⚠️ When generating the final CSV, put it at the end of your reply, and make sure it starts with the header (e.g., name, age...) and contains no extra explanation after it.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# === Session Initialization ===
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

conversation = ConversationChain(
    llm=llm,
    memory=st.session_state.memory,
    prompt=prompt,
    verbose=False,
)

# === Streamlit Layout ===
st.set_page_config(page_title="Data Generator Chatbot", page_icon="📊")
st.title("📊 Data Generator Chatbot")

# Show full chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# === Helper to Extract CSV from LLM Output ===
def extract_csv_block(text):
    lines = text.strip().split("\n")
    csv_lines = []
    found_header = False

    for line in lines:
        if "," in line and not found_header:
            found_header = True
            csv_lines.append(line)
        elif found_header:
            if "," in line:
                csv_lines.append(line)
            else:
                break
    return "\n".join(csv_lines) if len(csv_lines) >= 2 else None

# === Chat Input ===
user_input = st.chat_input("Send a message...")
if user_input:
    # Show user input
    st.chat_message("user").markdown(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Get LLM response
    with st.spinner("Thinking..."):
        response = conversation.predict(input=user_input)

    st.chat_message("assistant").markdown(response)
    st.session_state.chat_history.append({"role": "assistant", "content": response})

    # Extract CSV if found
    csv_text = extract_csv_block(response)
    if csv_text:
        try:
            df = pd.read_csv(StringIO(csv_text))
            if not df.empty:
                st.success("📁 Auto-detected and saved CSV!")
                st.dataframe(df.head(10))

                save_dir = "./saved_csvs"
                os.makedirs(save_dir, exist_ok=True)
                file_path = os.path.join(save_dir, "latest_verified.csv")
                df.to_csv(file_path, index=False)

                msg = f"✅ CSV automatically saved to `{file_path}`."
                st.chat_message("assistant").markdown(msg)
                st.session_state.chat_history.append({"role": "assistant", "content": msg})
            else:
                st.warning("⚠️ CSV found but appears empty.")
        except Exception as e:
            st.error(f"❌ Error reading CSV: {e}")

# === File Upload Section (Displayed like a chatbot message) ===
with st.container():
    st.divider()
    st.chat_message("assistant").markdown("📤 **Optional: Upload a CSV to infer schema or continue editing it.**")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")

    if uploaded_file:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            st.success("✅ Uploaded CSV preview:")
            st.dataframe(uploaded_df.head(10))

            csv_text = uploaded_df.to_csv(index=False)
            context = f"The user uploaded this CSV:\n```\n{csv_text}\n```"

            # Add messages to memory
            st.session_state.memory.chat_memory.add_user_message("I uploaded a CSV file.")
            st.session_state.memory.chat_memory.add_ai_message(context)

            bot_msg = "📥 Got your file. You can now ask me to analyze it or generate more rows."
            st.chat_message("assistant").markdown(bot_msg)
            st.session_state.chat_history.append({"role": "assistant", "content": bot_msg})
        except Exception as e:
            st.error(f"❌ Failed to read uploaded CSV: {e}")
