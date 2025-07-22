import streamlit as st
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

# === Load API Key ==
load_dotenv("/app/framework/test/env.txt")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY not found in environment.")
    st.stop()

# === Setup LLM ===
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model="llama3-8b-8192"
)

# === System Prompt ===
system_prompt = """
You are a data generation expert that helps users generate sample data in CSV format.

Handle 3 cases:
1. If user gives a schema, generate 100 rows of dummy data.
2. If user gives sample rows, infer schema.
3. If user gives an intent, suggest schema.

Allow user to edit schema (add/remove/rename/change types) before generating.
Only generate CSV when asked to.

⚠️ When generating the final CSV, put it at the end of your reply, and make sure it starts with the header (e.g., name, age...) and contains no extra explanation after it.

Example format:

Here is your generated data:

name,age
Alice,30
Bob,25
...
"""


# === Prompt Template ===
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# === Initialize Memory and Chat History ===
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# === LangChain Chain ===
conversation = ConversationChain(
    llm=llm,
    memory=st.session_state.memory,
    prompt=prompt,
    verbose=False,
)

# === Streamlit UI ===
st.set_page_config(page_title="Data Generator Chatbot", page_icon="📊")
st.title("📊 Data Generator Chatbot")

# Display conversation history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input box (chat-style)
user_input = st.chat_input("Send a message...")
if user_input:
    # Show user input
    st.chat_message("user").markdown(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Generate response
    with st.spinner("Thinking..."):
        response = conversation.predict(input=user_input)

    # Show response
    st.chat_message("assistant").markdown(response)
    st.session_state.chat_history.append({"role": "assistant", "content": response})
