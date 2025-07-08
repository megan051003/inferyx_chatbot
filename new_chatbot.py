import os
import json
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from flask import Flask, jsonify, request
from flask_cors import CORS
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv("/app/framework/test/env.txt")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GROQ_API_KEY:
    logging.error("GROQ_API_KEY environment variable is required.")
    raise ValueError("GROQ_API_KEY environment variable is required.")
if not GOOGLE_API_KEY:
    logging.error("GOOGLE_API_KEY environment variable is required.")
    raise ValueError("GOOGLE_API_KEY environment variable is required.")

INDEX_PATH = "/app/framework/documents/inferyx_faiss_index"

# Initialize embeddings
logging.info("Initializing Google Generative AI Embeddings...")
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    task_type="semantic_similarity",
    google_api_key=GOOGLE_API_KEY
)
logging.info("Embeddings initialized.")

# Load FAISS index
logging.info(f"Loading FAISS vector index from {INDEX_PATH}...")
try:
    db = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    logging.info("FAISS index loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load FAISS index: {e}")
    raise RuntimeError(f"Failed to load FAISS index: {e}")

# Initialize Groq LLM
logging.info("Initializing Groq LLM...")
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model="llama3-8b-8192")
logging.info("Groq LLM initialized.")

# Initialize Flask app
app = Flask(__name__)
CORS(app)

def err_template(err_msg: str) -> dict:
    return {
        "status": "error",
        "message": "Request failed",
        "data": [],
        "meta": {},
        "error": {
            "code": None,
            "message": err_msg,
            "details": None
        }
    }

def success_template(text: str, user_input: str, sources: list) -> dict:
    unique_sources = sorted(list(set(src for src in sources if src)))
    if unique_sources:
        formatted_sources_list = [
            f"- Source {i} :- [{src}]({src})"
            for i, src in enumerate(unique_sources, 1)
        ]
        sources_text = "\n".join(formatted_sources_list)
    else:
        sources_text = "No relevant sources found in the documentation."

    full_response_markdown = (
        f"## Your Answer:\n\n"
        f"{text.strip()}\n\n"
        f"---\n\n"
        f"### Sources:\n"
        f"{sources_text}"
    )

    return {
        "status": "success",
        "message": "Request processed successfully",
        "data": {
            "text": full_response_markdown,
            "sources": unique_sources
        },
        "meta": {
            "nodeCount": None,
            "edgeCount": None,
            "total": len(unique_sources),
            "type": "TEXT",
            "query": user_input
        },
        "error": {
            "code": None,
            "message": None,
            "details": None
        }
    }

def chatbot_logic(user_input: str) -> str:
    if not user_input or not user_input.strip():
        return json.dumps(err_template("User input cannot be empty."))

    cleaned_input = user_input.strip().lower()

    # Handle greetings directly
    greeting_inputs = {"hi", "hello", "hey", "greetings", "good morning", "good afternoon"}
    if any(cleaned_input.startswith(greet) for greet in greeting_inputs):
        logging.info("Greeting detected — skipping FAISS + LLM.")
        return json.dumps(success_template(
            text="👋 Hello! Feel free to ask me anything about the Inferyx documentation.",
            user_input=user_input.strip(),
            sources=[]
        ))

    try:
        # Similarity search with score threshold
        scored_docs = db.similarity_search_with_score(user_input.strip(), k=5)
        threshold = 0.5
        filtered_docs = [doc for doc, score in scored_docs if score >= threshold]

        for i, (doc, score) in enumerate(scored_docs):
            logging.info(f"[Doc {i+1}] score={score:.3f} | preview={doc.page_content[:80]}")

        if not filtered_docs:
            logging.info("No relevant documents matched — returning fallback message.")
            return json.dumps(success_template(
                text="I'm sorry, I couldn't find anything in the Inferyx documentation to answer that.",
                user_input=user_input.strip(),
                sources=[]
            ))

        context = "\n\n".join([doc.page_content for doc in filtered_docs])
        sources = [doc.metadata.get("url", "") for doc in filtered_docs]

        # Updated prompt with self-reporting
        prompt = (
            "You are a helpful assistant that answers questions about Inferyx documentation.\n\n"
            "Answer clearly and concisely based only on the provided context.\n"
            "If the context does not contain enough information to answer the question, "
            "state that you cannot answer and do not make up information.\n\n"
            f"Context:\n{context}\n\n"
            f"User Input:\n{user_input}\n\n"
            "Answer clearly.\n\n"
            "At the end of your answer, write one of the following:\n"
            "- If the answer was based on the context: [ANSWERED_FROM_CONTEXT]\n"
            "- If the answer was NOT based on the context: [NOT_ANSWERED_FROM_CONTEXT]"
        )

        response = llm.invoke(prompt)
        llm_response_content = response.content.strip()

        # Use self-report tag to determine whether to show sources
        if "[NOT_ANSWERED_FROM_CONTEXT]" in llm_response_content:
            logging.info("LLM indicated it could not answer from context — hiding sources.")
            sources = []
        elif "[ANSWERED_FROM_CONTEXT]" in llm_response_content:
            logging.info("LLM confirmed it answered from context — keeping sources.")
        else:
            logging.warning("LLM did not include answer flag — treating as not answered.")
            sources = []

        # Remove the tags before displaying final answer
        llm_response_content = llm_response_content.replace(
            "[ANSWERED_FROM_CONTEXT]", "").replace("[NOT_ANSWERED_FROM_CONTEXT]", "").strip()

        return json.dumps(success_template(
            text=llm_response_content,
            user_input=user_input.strip(),
            sources=sources
        ))

    except Exception as e:
        logging.error(f"An error occurred during chatbot processing: {str(e)}")
        return json.dumps(err_template(f"An error occurred during chatbot processing: {str(e)}"))

@app.route('/ask_ai', methods=['POST'])
def ask_ai():
    try:
        data = request.get_json()
        if not data or 'user_input' not in data:
            return jsonify(err_template("User input is required in the request body.")), 400

        user_input = data['user_input']
        response_json_str = chatbot_logic(user_input)
        response_data = json.loads(response_json_str)

        status_code = 200 if response_data.get("status") == "success" else 500
        return jsonify(response_data), status_code

    except Exception as e:
        logging.error(f"Server error in /ask_ai route: {str(e)}")
        return jsonify(err_template(f"Server error: {str(e)}")), 500

if __name__ == '__main__':
    logging.info("Starting Flask server on http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
