# server.py
from flask import Flask, request, jsonify
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import json
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

# Load env and constants
load_dotenv("env.txt")
#load_dotenv("/app/framework/test/.env")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
assert GROQ_API_KEY, "GROQ_API_KEY missing!"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_PATH = "/app/framework/test/inferyx_faiss_index"

# Initialize models once
embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
db = FAISS.load_local(INDEX_PATH, embedding_model, allow_dangerous_deserialization=True)
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model="meta-llama/llama-4-scout-17b-16e-instruct")

@app.route('/launch_chatbot', methods=['POST'])
def launch_chatbot():
    try:
        data = request.get_json()
        question = data.get("question", "").strip()

        if not question:
            return jsonify({"error": "Question is required"}), 400

        docs = db.similarity_search(question, k=5)
        context = "\n\n".join([doc.page_content for doc in docs])
        sources = list(set(doc.metadata.get("url", "") for doc in docs if doc.metadata.get("url", "")))
        prompt = (
            f"You are a helpful assistant that helps users understand Inferyx documentation.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{question}\n\n"
            f"Answer clearly and concisely:"
        )

        response = llm.invoke(prompt)
        return jsonify({
            "status": "success",
            "question": question,
            "answer": response.content.strip(),
            "sources": sources
        }), 200

    except Exception as e:
        logging.exception("Error in chatbot")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5005)
