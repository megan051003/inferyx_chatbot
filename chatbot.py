import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, OpenAI

# Load FAISS vector store
db = FAISS.load_local("inferyx_faiss_index", OpenAIEmbeddings(), allow_dangerous_deserialization=True)

# Create the QA chain with conversational memory
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=OpenAI(temperature=0),
    retriever=db.as_retriever(),
    return_source_documents=True
)

st.set_page_config(page_title="Inferyx Documentation Chatbot", page_icon="💬")
st.title("💬 Inferyx Documentation Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Get user input
if prompt := st.chat_input("Ask about Inferyx docs..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build chat history as list of (user_msg, assistant_msg) tuples
    chat_history = []
    msgs = st.session_state.messages
    i = 0
    while i < len(msgs):
        if msgs[i]["role"] == "user":
            user_msg = msgs[i]["content"]
            assistant_msg = ""
            if i + 1 < len(msgs) and msgs[i + 1]["role"] == "assistant":
                assistant_msg = msgs[i + 1]["content"]
                i += 1
            chat_history.append((user_msg, assistant_msg))
        i += 1

    # Pass the conversation history except current prompt
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = qa_chain({
                "question": prompt,
                "chat_history": chat_history[:-1]  # all Q-A pairs before current input
            })

            answer = result["answer"]

            # Format sources
            sources = []
            seen = set()
            for doc in result.get("source_documents", []):
                title = doc.metadata.get("title", "Untitled")
                url = doc.metadata.get("url", "")
                key = (title, url)
                if key not in seen:
                    seen.add(key)
                    if url:
                        sources.append(f"- [{title}]({url})")
                    else:
                        sources.append(f"- {title}")
            source_text = "\n\n📚 **Sources:**\n" + "\n".join(sources) if sources else ""

            full_response = answer + source_text
            st.markdown(full_response)

            st.session_state.messages.append({"role": "assistant", "content": full_response})
