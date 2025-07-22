import os
import asyncio
import json
import mysql.connector
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_community.graphs import Neo4jGraph

# -----------------------
# 🔐 Load environment variables
# -----------------------
load_dotenv("/app/framework/test/env.txt")  # Ensure this path is correct
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# -----------------------
# 🤖 Set up LLM
# -----------------------
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name="llama3-8b-8192",
    temperature=0
)

# -----------------------
# ✍️ Custom Prompt for Schema-to-Graph
# -----------------------
custom_prompt = ChatPromptTemplate.from_template("""
You are given the schema of a MySQL database. Your job is to analyze the schema and extract a knowledge graph.

- Each table represents an entity.
- Columns are attributes of that entity.
- If a column looks like a foreign key (e.g., user_id in 'orders' table referring to 'users'), treat it as a relationship between two entities.

Return the output in structured JSON with:
- `nodes`: an array of nodes. Each node should have:
  - `id`: unique node id
  - `type`: entity type
  - `properties`: a list of objects with `{{"key"}}` and `{{"value"}}`, e.g., `[{{"key": "user_id", "value": "string"}}]`

- `relationships`: an array of edges, each with:
  - `source_node_id`, `source_node_type`, `target_node_id`, `target_node_type`, and `type`.

Schema:
---
{input}
""")




# -----------------------
# 🔄 Graph Transformer with LLM
# -----------------------
transformer = LLMGraphTransformer(
    llm=llm,
    prompt=custom_prompt,
    node_properties=True
)


# -----------------------
# 🌐 Neo4j Graph Setup
# -----------------------
graph = Neo4jGraph(
    url="neo4j://65.1.212.204:7687",
    username="neo4j",
    password="20Inferyx!9"
)

# -----------------------
# 🧠 Extract database name from input sentence
# -----------------------
def extract_db_name(text):
    prompt = f"""Extract the name of the MySQL database from this sentence:
{text}
Just return the database name only, no extra text or quotes."""
    response = llm.invoke(prompt)
    return response.content.strip(" `\n")

# -----------------------
# 🗄️ Get schema from MySQL
# -----------------------
def get_database_schema_as_text(database_name):
    schema_text = f"Schema for database `{database_name}`:\n"
    conn = mysql.connector.connect(
        host="localhost",  # ⚠️ Change if your DB is remote
        user="inferyx",
        password="inferyx",
        database=database_name
    )
    cursor = conn.cursor()
    cursor.execute("SHOW TABLES;")
    tables = [row[0] for row in cursor.fetchall()]

    for table in tables:
        cursor.execute(f"SHOW COLUMNS FROM {table};")
        columns = [col[0] for col in cursor.fetchall()]
        schema_text += f"\nTable `{table}`:\n  " + ", ".join(columns)

    cursor.close()
    conn.close()
    return schema_text

# -----------------------
# 🚀 Async main workflow
# -----------------------
async def main():
    user_input = input("📝 Enter a sentence with the DB name: ")
    db_name = extract_db_name(user_input)
    print(f"✅ Extracted DB name: {db_name}")

    schema_text = get_database_schema_as_text(db_name)
    print("\n✅ Got schema. Converting to graph...")

    doc = Document(page_content=schema_text)
    graph_docs = await transformer.aconvert_to_graph_documents([doc])

    print("\n🧾 Graph JSON Output:")
    print(json.dumps([doc.dict() for doc in graph_docs], indent=2))

    print(f"\n📦 Storing {len(graph_docs)} document(s) to Neo4j...")
    graph.add_graph_documents(
        graph_docs,
        baseEntityLabel=True)
    print("✅ Done. View in Neo4j Browser!")

# -----------------------
# ▶️ Run the app
# -----------------------
if __name__ == "__main__":
    asyncio.run(main())
