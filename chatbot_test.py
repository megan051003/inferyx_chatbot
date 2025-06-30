#Steps/changes: download Ollama + ollama pull mistral on terminal

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
import os

# Paths and models
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_PATH = os.path.expanduser("/Users/name/Downloads/inferyx_test/inferyx_faiss_index")

# Load vector store and embedding model
embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
db = FAISS.load_local(INDEX_PATH, embedding_model, allow_dangerous_deserialization=True)

# Load local LLM model via Ollama
llm = OllamaLLM(model="mistral")

def ask_question(question: str):
    # Search docs for relevant context
    docs = db.similarity_search(question, k=5)
    context = "\n\n".join([doc.page_content for doc in docs])

    # Prepare prompt with context and question
    prompt = (
        "You are a helpful assistant that helps users understand Inferyx documentation.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        "Answer clearly and concisely:"
    )

    # Get response from LLM
    response = llm.invoke(prompt)
    return response.strip()

if __name__ == "__main__":
    print("🤖 Ask questions about Inferyx docs. Type 'exit' or 'quit' to stop.\n")
    while True:
        q = input("Enter your question: ").strip()
        if q.lower() in {"exit", "quit"}:
            print("👋 Goodbye!")
            break
        if not q:
            print("Please enter a question or type 'exit' to quit.")
            continue

        answer = ask_question(q)
        print("\nAnswer:")
        print(answer)
        print("\n" + "-"*50 + "\n")

