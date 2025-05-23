# app/chatbot.py
import os
os.environ["OPENAI_API_KEY"] = "your-api-key"  # Replace with env var or secure method

import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, OpenAI

db = FAISS.load_local("inferyx_index", OpenAIEmbeddings(), allow_dangerous_deserialization=True)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=OpenAI(temperature=0),
    retriever=db.as_retriever(),
    return_source_documents=True
)

st.set_page_config(page_title="Inferyx Documentation Chatbot", page_icon="💬")
st.title("💬 Inferyx Documentation Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about Inferyx docs..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    chat_history = []
    i = 0
    msgs = st.session_state.messages
    while i < len(msgs):
        if msgs[i]["role"] == "user":
            user_msg = msgs[i]["content"]
            assistant_msg = ""
            if i + 1 < len(msgs) and msgs[i + 1]["role"] == "assistant":
                assistant_msg = msgs[i + 1]["content"]
                i += 1
            chat_history.append((user_msg, assistant_msg))
        i += 1

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = qa_chain({
                "question": prompt,
                "chat_history": chat_history[:-1]
            })

            answer = result["answer"]
            sources = []
            seen = set()
            for doc in result.get("source_documents", []):
                title = doc.metadata.get("title", "Untitled")
                url = doc.metadata.get("url", "")
                key = (title, url)
                if key not in seen:
                    seen.add(key)
                    sources.append(f"- [{title}]({url})" if url else f"- {title}")

            source_text = "\n\n📚 **Sources:**\n" + "\n".join(sources) if sources else ""
            full_response = answer + source_text
            st.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
