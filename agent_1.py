import os
from typing import Dict, Optional
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv("/app/framework/test/env.txt")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ✅ Use correct Groq model
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model="llama3-8b-8192"
)

system_prompt = """
You are a helpful assistant that extracts all required fields to either create or delete a datapod. The user may provide incomplete or ambiguous input.

Step 1: Based on the user's intent (create or delete), identify what fields are needed.
Step 2: If any required fields are missing, ask the user for those details clearly.
Step 3: Do NOT proceed until all fields are complete. Wait for user's next message.
Step 4: Once complete, summarize the extracted values clearly.

Expected fields:
- action: create | delete
- datapod_name
- primary_key (only for create)

Examples:
User: "Create a datapod called user_data"
Assistant: "What should be the primary key for the datapod 'user_data'?"

User: "Use id"
Assistant: "✅ Got it. Creating datapod 'user_data' with primary key: id"

User: "Delete datapod orders"
Assistant: "✅ Got it. Deleting datapod 'orders'"
"""

chat_history = []

def extract_fields_conversational(user_input: str) -> str:
    chat_history.append(HumanMessage(content=user_input))

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        *[("human" if isinstance(msg, HumanMessage) else "ai", msg.content) for msg in chat_history]
    ])

    chain = prompt | llm
    response = chain.invoke({})

    chat_history.append(AIMessage(content=response.content))
    return response.content


# Example usage
if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        reply = extract_fields_conversational(user_input)
        print("Assistant:", reply)
