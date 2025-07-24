import os
import asyncio
import json
from pathlib import Path
from typing import List
from flask import Flask, request, jsonify
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel

app = Flask(__name__)

# 🔐 Set API key securely
os.environ['GOOGLE_API_KEY'] = 'AIzaSyDSimKG7qJwpNYN13faRVgf5yBVs2AaK9I'

# 🔧 Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2
)

# 🧠 Prompt template
schema_prompt = ChatPromptTemplate.from_template("""
You are a data modeling expert.
Given the CSV schema below, extract the **graph schema**, not data instances.

Output must contain:
1. A list of nodes (with type and properties).
2. A list of relationships (with type, source type, and target type).

Only include high-level entity nodes (like Customer, Order, Product) — no field-level nodes.

Return only structured data.

Previously extracted graph schema:
{context}

New schema to process:
{schema}
""")

# 📦 Output schema structure
class GraphNode(BaseModel):
    node_type: str
    properties: List[str]
    node_type_id: int

class GraphRelationship(BaseModel):
    edge_type: str
    source: str
    target: str

class GraphSchema(BaseModel):
    nodes: List[GraphNode]
    relationships: List[GraphRelationship]

# 🔁 Utility: batching
def batchify(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]

# 📂 Read headers from files
def get_schema_text_from_dir(csv_dir: str) -> List[str]:
    schemas = []
    for file in Path(csv_dir).glob("*"):
        try:
            if file.suffix == ".csv":
                df = pd.read_csv(file, nrows=1)
            elif file.suffix in [".xlsx", ".xls"]:
                df = pd.read_excel(file, nrows=1)
            else:
                continue
            schemas.append(f"Table: {file.stem}\nColumns: {', '.join(df.columns)}")
        except Exception as e:
            print(f"⚠️ Skipping {file.name}: {e}")
    return schemas

# 🔍 Core extraction logic
async def extract_kg_from_csv(csv_dir: str):
    schema_chunks = list(batchify(get_schema_text_from_dir(csv_dir), 10))
    structured_llm = llm.with_structured_output(GraphSchema)

    all_nodes, all_edges = [], []
    context = ""

    for idx, chunk in enumerate(schema_chunks):
        schema_text = "\n\n\n".join(chunk)

        prompt = schema_prompt.format_messages(
            context=context,
            schema=schema_text
        )

        try:
            result = await structured_llm.ainvoke(prompt)
            print(f"✅ Processed batch {idx + 1}")
        except Exception as e:
            print(f"❌ LLM error in batch {idx + 1}: {e}")
            continue

        all_nodes.extend(result.nodes)
        all_edges.extend(result.relationships)

        context = f"""
        Nodes:
        {[f"{n.node_type}: {n.properties}" for n in all_nodes]}

        Relationships:
        {[f"{r.edge_type}: {r.source} -> {r.target}" for r in all_edges]}
        """

    return all_nodes, all_edges

# 🚀 Flask route
@app.route('/infer_graph_from_csv', methods=['POST'])
def infer_graph_from_csv():
    try:
        data = request.get_json()
        csv_dir = data.get('csvDir', '/app/framework/upload/user_files')

        print(f"📁 Reading files from: {csv_dir}")
        nodes, edges = asyncio.run(extract_kg_from_csv(csv_dir))

        # Print to terminal as raw JSON
        kg_json = {
            "nodes": [n.dict() for n in nodes],
            "relationships": [r.dict() for r in edges]
        }
        print(json.dumps(kg_json, indent=2))

        return jsonify({
            "status": "success",
            "message": "Knowledge graph extracted successfully",
            "data": kg_json,
            "meta": {
                "nodeCount": len(nodes),
                "edgeCount": len(edges)
            }
        }), 200

    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({
            "status": "error",
            "message": str(e),
            "data": None
        }), 500

if __name__ == '__main__':
    app.run(debug=True)
