import os
import asyncio
import pandas as pd
from pathlib import Path
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.pydantic_v1 import BaseModel
from typing import List
from langchain_community.graphs import Neo4jGraph

# 🔐 API + Neo4j credentials
os.environ['GOOGLE_API_KEY'] = 'AIzaSyDSimKG7qJwpNYN13faRVgf5yBVs2AaK9I'

graph = Neo4jGraph(
    url="neo4j://65.1.212.204:7687",
    username="neo4j",
    password="20Inferyx!9"
)

# 🔧 Initialize the LLM
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

# 🧱 Output schema structure
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

# 🔁 Batching helper
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

# 🧠 LLM + Neo4j Push
async def run_kg_extraction(csv_dir="/app/framework/upload/user_files"):
    print(f"📁 Reading files from {csv_dir}...")
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
            print(f"✅ Batch {idx+1} processed")
        except Exception as e:
            print(f"❌ LLM error in batch {idx+1}: {e}")
            continue

        all_nodes.extend(result.nodes)
        all_edges.extend(result.relationships)

        context = f"""
        Nodes:
        {[f"{n.node_type}: {n.properties}" for n in all_nodes]}

        Relationships:
        {[f"{r.edge_type}: {r.source} -> {r.target}" for r in all_edges]}
        """

    # 🌐 Push nodes to Neo4j
    for node in all_nodes:
        try:
            # Fill properties with dummy values for now
            props = ", ".join([f"{prop}: '{prop}'" for prop in node.properties])
            cypher = f"""
            MERGE (n:{node.node_type} {{id: {node.node_type_id}}})
            SET n += {{
                {props},
                name: '{node.node_type}'
            }}
            """
            graph.query(cypher)
        except Exception as e:
            print(f"❌ Error creating node {node.node_type}: {e}")

    # 🔗 Push relationships to Neo4j
    for rel in all_edges:
        try:
            cypher = f"""
            MATCH (a:{rel.source}), (b:{rel.target})
            MERGE (a)-[:{rel.edge_type}]->(b)
            """
            graph.query(cypher)
        except Exception as e:
            print(f"❌ Error creating relationship {rel.edge_type}: {e}")

    return all_nodes, all_edges

# 🚀 Run the whole thing
if __name__ == "__main__":
    nodes, edges = asyncio.run(run_kg_extraction("/app/framework/upload/user_files"))

    print("\n=== 📌 Final Graph Schema ===")
    print(f"Nodes ({len(nodes)}):")
    for n in nodes:
        print(f" - {n.node_type} (ID {n.node_type_id}) → Props: {n.properties}")

    print(f"\nEdges ({len(edges)}):")
    for e in edges:
        print(f" - {e.edge_type}: {e.source} → {e.target}")
