import requests
import json
import pickle
import sys
from pathlib import Path
from typing import Set, Dict, List
from langchain_community.vectorstores.faiss import FAISS
from langchain.schema import Document
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import random
import re

# --- Functions from inferyx_sdk.py ---
class AppConfig:
    """
    Configuration class to hold API-related settings.
    """
    def __init__(self, host: str, appToken: str, adminToken: str):
        self.host = host
        self.appToken = appToken
        self.adminToken = adminToken

    def get_platform_url(self):
        """
        Constructs and returns the base platform URL.
        """
        return f"https://{self.host}/framework"

def get_all(data_type: str, app_config: AppConfig):
    """
    Retrieves all latest data for a specified type from the Inferyx API.

    Args:
        data_type (str): The type of data collection to retrieve.
        app_config (AppConfig): The application configuration object.

    Returns:
        dict: A dictionary containing the JSON response from the API, or None if the request fails.
    """
    BASE_URL = f"https://{app_config.host}/framework/common/getAllLatest"
    
    if not isinstance(data_type, str) or not data_type:
        print("Error: 'data_type' must be a non-empty string.")
        return None
        
    params = {
        "action": "view",
        "type": data_type
    }
    
    headers = {
        "token": app_config.appToken,
    }
    
    print(f"Fetching all latest data for type: {data_type}...")
    
    try:
        response = requests.get(BASE_URL, params=params, headers=headers)
        response.raise_for_status()
        
        print("Request successful.")
        data = response.json()
        
        return data
        
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while making the request: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Failed to decode JSON from the response: {e}")
        return None

def get_one(uuid: str, version: int, type: str, app_config: AppConfig):
    """
    Retrieves a single item by UUID, version, and type from the Inferyx API.
    
    Args:
        uuid (str): The UUID of the item to retrieve.
        version (int): The version number of the item.
        type (str): The type of data collection the item belongs to.
        app_config (AppConfig): The application configuration object.

    Returns:
        dict: A dictionary containing the JSON response from the API, or None if the request fails.
    """
    url = f'{app_config.get_platform_url()}/common/getOneByUuidAndVersion?action=view&uuid={uuid}&version={version}&type={type}'
    headers = {
        "token": app_config.appToken,
        "Content-Type": "application/json",
    }

    print(f"Fetching data for UUID: {uuid}, Version: {version}, Type: {type}...")
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        print("Request successful.")
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while making the request: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Failed to decode JSON from the response: {e}")
        return None
# --- End of functions from inferyx_sdk.py ---

# List of all available collections
collections = ["alert", "application",
                "batch","caserule", 
               "classificationrule", "dag", "dashboard", "dataarchive", "dataasset",
               "datadomain", "dataglossary", "datapod", "dataproduct", "dataset", "datasource",
               "domain", "dq", "dqgroup",  "entity", "er",
               "feature", 
               "function", "graphpod", "ingest", "ingestgroup", "map", "mapgroup", 
               "model", "organization",
                "predict", "profile", "profilegroup", "recon",
               "recongroup","report", "rule", "rulegroup",  "train",
              "vizpod"]

def flatten_json(y, parent_key="", sep="."):
    """Flattens a nested JSON object into a single-level dictionary."""
    items = []
    if isinstance(y, dict):
        for k, v in y.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            items.extend(flatten_json(v, new_key).items())
    elif isinstance(y, list):
        for i, v in enumerate(y):
            items.extend(flatten_json(v, f"{parent_key}[{i}]").items())
    else:
        items.append((parent_key, y))
    return dict(items)

def generate_prefix_from_meta(base_meta: Dict, k: str, v) -> str:
    """Generates a descriptive prefix for a document based on its metadata."""
    collection = base_meta.get("collection", "document")
    name = base_meta.get("name", "Unnamed")
    display_name = base_meta.get("displayName", name)
    uuid = base_meta.get("uuid", "unknown UUID")
    version = base_meta.get("version", "unknown version")
    doc_id = base_meta.get("doc_id", "unknown ID")

    sentence = (
        f"'{k}:{v}' is the metadata for the '{collection}' named '{name}', "
        f"with display name '{display_name}', UUID '{uuid}', version '{version}', "
        f"and internal document ID '{doc_id}'."
    )
    return sentence

def describe_flattened_json(base_meta: Dict, k: str, v) -> str:
    """Generates a descriptive sentence for a flattened JSON field."""
    prefix = generate_prefix_from_meta(base_meta, k, v)
    f = k.split('.')

    level_3_plus_templates = [
        lambda fields, value: f"This {base_meta['collection']} has '{fields[0]}', and its '{fields[-1]}' is '{value}'.",
        lambda fields, value: f"In '{fields[0]}', the field '{fields[-1]}' has value '{value}'.",
        lambda fields, value: f"The nested field '{fields[0]}.{fields[-1]}' in the {base_meta['collection']} is '{value}'.",
        lambda fields, value: f"This {base_meta['collection']} has {fields[0]} whose '{fields[-1]}' is '{value}'."
    ]

    level_2_templates = [
        lambda fields, value: f"This {base_meta['collection']} contains '{fields[0]}' with '{fields[-1]}' set to '{value}'.",
        lambda fields, value: f"The field '{fields[-1]}' inside '{fields[0]}' is assigned the value '{value}'.",
        lambda fields, value: f"In the section '{fields[0]}', the '{fields[-1]}' is '{value}'."
    ]

    level_1_templates = [
        lambda key, value: f"This {base_meta['collection']} has a field '{key}' with value '{value}'.",
        lambda key, value: f"The field '{key}' is defined as '{value}'.",
        lambda key, value: f"Field '{key}' has been assigned the value '{value}'."
    ]

    if len(f) >= 3:
        s = prefix + "\n" + random.choice(level_3_plus_templates)(f, v)
    elif len(f) == 2:
        s = prefix + "\n" + random.choice(level_2_templates)(f, v)
    else:
        s = prefix + "\n" + random.choice(level_1_templates)(k, v)
    return s

def create_output_dir(app_uuid: str) -> Path:
    """Creates the output directory for the FAISS index."""
    base_path = Path(f"/app/framework/index/{app_uuid}")
    base_path.mkdir(parents=True, exist_ok=True)
    return base_path

def load_indexed_ids(indexed_ids_path: Path, vectorstore_path: Path, embeddings) -> (Set[str], FAISS):
    """Loads previously indexed IDs and the existing FAISS index."""
    if indexed_ids_path.exists() and vectorstore_path.exists():
        try:
            with indexed_ids_path.open("rb") as f:
                indexed_ids = pickle.load(f)
            print(f"[INFO] Loaded {len(indexed_ids)} previously indexed IDs from '{indexed_ids_path}'.")

            faiss_index = FAISS.load_local(
                str(vectorstore_path),
                embeddings,
                allow_dangerous_deserialization=True
            )
            print(f"[INFO] Loaded existing FAISS index from '{vectorstore_path}'.")
            return indexed_ids, faiss_index
        except Exception as e:
            print(f"[ERROR] Failed to load existing index or IDs: {e}. Starting fresh.")
            return set(), None
    
    return set(), None

def prepare_documents(documents: List[Dict], collection_name: str) -> (List[Document], Set[str]):
    """Prepares documents for indexing by flattening and adding metadata."""
    doc_chunks = []
    new_ids = set()

    print(f"[INFO] Preparing {len(documents)} documents for indexing from '{collection_name}'...")
    
    for i, doc in enumerate(documents):
        doc_id = doc.get("_id", doc.get("uuid")) # Use UUID if _id is not available
        if doc_id:
            new_ids.add(doc_id)
        
        # Exclude common fields that are not useful for semantic search
        doc_copy = doc.copy()
        for key in ["_id", "active", "appInfo", "isClone", "createdAt", "updatedAt"]:
            doc_copy.pop(key, None)
        
        flat_doc = flatten_json(doc_copy)

        base_meta = {
            "doc_id": doc_id,
            "uuid": doc.get("uuid"),
            "version": doc.get("version"),
            "name": doc.get("name"),
            "displayName": doc.get("displayName"),
            "collection": collection_name
        }

        for key, val in flat_doc.items():
            if isinstance(val, (str, int, float)) and str(val).strip():
                # Avoid indexing excessively long strings
                if len(str(val)) > 1000:
                    continue
                
                # Create a meaningful description for the vector store
                text = describe_flattened_json(base_meta, key, str(val))
                doc_chunks.append(Document(
                    page_content=text,
                    metadata={**base_meta, "key_path": key}
                ))

        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1} of {len(documents)} documents in '{collection_name}'...")
            sys.stdout.flush()

    print(f"[INFO] Prepared {len(doc_chunks)} chunks for indexing from '{collection_name}'.")
    return doc_chunks, new_ids

def build_faiss_index(all_docs: List[Document], embeddings) -> FAISS:
    """Builds a FAISS index from a list of documents in batches."""
    if not all_docs:
        return None
    try:
        print(f"[INFO] Building FAISS index from {len(all_docs)} documents... This may take a while.")
        sys.stdout.flush()
        
        # Initialize an empty FAISS index
        faiss_index = None
        
        # Process in batches to show progress
        batch_size = 1000
        for i in range(0, len(all_docs), batch_size):
            batch = all_docs[i:i + batch_size]
            if not faiss_index:
                # Create the initial index from the first batch
                faiss_index = FAISS.from_documents(batch, embeddings)
                print(f"  Indexed 0-{len(batch)} documents.")
            else:
                # Merge subsequent batches into the existing index
                temp_index = FAISS.from_documents(batch, embeddings)
                faiss_index.merge_from(temp_index)
                print(f"  Indexed {i}-{i + len(batch)} documents.")
            sys.stdout.flush()

        print("[INFO] FAISS index successfully built.")
        sys.stdout.flush()
        return faiss_index
    except Exception as e:
        raise RuntimeError(f"[ERROR] FAISS index creation failed: {e}")

def persist_outputs(base_path: Path, index: FAISS, indexed_ids: Set[str]) -> None:
    """Saves the FAISS index and the list of indexed IDs to disk."""
    if not index:
        print("[INFO] No index to save.")
        return
    try:
        index.save_local(str(base_path))
        print(f"[SUCCESS] FAISS index saved to '{base_path}'.")

        with (base_path / "indexed_ids.pkl").open("wb") as f:
            pickle.dump(indexed_ids, f)
        print("[SUCCESS] Indexed IDs updated.")
    except Exception as e:
        raise RuntimeError(f"[ERROR] Failed to save outputs: {e}")

def run_data_catalog_indexing():
    """Main function to perform the data catalog indexing."""
    
    # --- Integration with inferyx_sdk ---
    app_config = AppConfig(
        host="dev.inferyx.com",
        appToken="bM8P1i6Xf7O1NpyCOoWmJrrcELE3JkcGzBbguCzB",
        adminToken="iresTHOb208NrFOuLbdrgNNYuUNHYOrCyeQRrISL"
    )

    try:
        print("[INFO] Initializing Google Generative AI Embeddings...")
        os.environ["GOOGLE_API_KEY"] = "AIzaSyBkfcgoZZ8AhsdjDtjiQiFIuZcHxHUrSoI"
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            task_type="semantic_similarity"
        )
    except Exception as e:
        raise RuntimeError(f"[ERROR] Failed to initialize embeddings: {e}")

    # Get all applications to process
    applications = get_all(data_type="application", app_config=app_config)
    if not applications:
        print("\n[INFO] No applications found. Exiting.")
        return

    for app in applications:
        app_uuid = app.get("uuid")
        if not app_uuid:
            print(f"[WARN] Skipping application with missing UUID: {app.get('name', 'Unnamed')}")
            continue

        print(f"\n--- Processing application: {app.get('name', 'Unnamed')} (UUID: {app_uuid}) ---")
        base_path = create_output_dir(app_uuid)
        indexed_ids_path = base_path / "indexed_ids.pkl"
        vectorstore_path = base_path
        
        all_indexed_ids, combined_index = load_indexed_ids(indexed_ids_path, vectorstore_path, embeddings)
        all_new_documents_to_index = []

        for coll_name in collections:
            print(f"\n--- Processing collection: '{coll_name}' ---")
            try:
                # Use the SDK to get all documents for the current collection
                new_docs = get_all(data_type=coll_name, app_config=app_config)
                
                if not new_docs:
                    print(f"[INFO] No new documents to index in '{coll_name}'.")
                    continue

                doc_chunks, new_ids = prepare_documents(new_docs, coll_name)
                all_new_documents_to_index.extend(doc_chunks)
                all_indexed_ids.update(new_ids)
            except Exception as e:
                print(f"[ERROR] An error occurred while processing '{coll_name}': {e}")
                continue

        if not all_new_documents_to_index:
            print("\n[INFO] No new documents found across all collections. Index is up to date.")
            continue
        
        # Create a new index from the new documents
        new_index = build_faiss_index(all_new_documents_to_index, embeddings)
        
        if new_index:
            if combined_index:
                print("[INFO] Merging new index with existing index...")
                combined_index.merge_from(new_index)
                print("[SUCCESS] Vectorstore updated with new documents.")
            else:
                combined_index = new_index
                print("[SUCCESS] New vectorstore created.")

            persist_outputs(base_path, combined_index, all_indexed_ids)
        
    print("\nAll indexing processes finished.")


if __name__ == "__main__":
    print("Starting data catalog indexing process...")
    try:
        run_data_catalog_indexing()
    except RuntimeError as e:
        print(f"[FATAL] Indexing failed: {e}")
