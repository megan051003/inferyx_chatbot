'''
*******************************************************************************
 * Copyright (C) Inferyx Inc, 2018 to 2022  All rights reserved. 
 *
 * This unpublished material is proprietary to Inferyx Inc.
 * The methods and techniques described herein are considered  trade 
 * secrets and/or confidential. Reproduction or distribution, in whole or 
 * in part, is forbidden.
 *
 * Written by Yogesh Palrecha <ypalrecha@inferyx.com>
*****************************************************************************
'''

import os, sys, json, pandas as pd
import shutil 
import glob

print("Inside python script ")

if len(sys.argv) > 1 and sys.argv[1] != 'predictApiHostname':

    sourceDbName, activation, lossFunction, layerNames, dbName, targetPath, operation, password, url, port, special_space_replacer, \
    targetDbName, targetPort, userName, targetHostName, inputConfigFilePath, optimizationAlgo, sourceHostName, sourceDsType, \
    modelFilePath, featureWeightPath, sourceUserName, targetUserName, targetPassword, tableName, targetDsType, sourcePassword, \
    hostName, sourcePort, sourceFilePath, paramName, query, outputResultPath, targetTableName, \
    targetDriver, includeFeatures, updater = (' ') * 37

    zero = lambda n: [0 for _ in range(n)]
    nEpochs, batch_size, seed, iterations, learningRate, weightInit, num_leaves, min_data, \
    numInput, numOutputs, numHidden, numLayers, numHiddenLayers, output_dim, n_estimators, colsample_bylevel, \
    min_child_weight, random_state, reg_lambda, reg_alpha, scale_pos_weight, max_delta_step, width, height, poss_outcome, \
    interval, frameGroup, possOutcome, numIter, errorScore, verbose, numJobs, numFolds, numSplits, batchSize, \
    max_features, min_df, colsample_bynode, num_classes, max_leaves, n_jobs, verbosity, subsample_freq, subsample_for_bin, min_child_samples, gpu_id, \
    min_samples_split, min_samples_leaf, intercept_scaling, max_iter, n_clusters, n_init = zero(52)

    none = lambda n: [None for _ in range(n)]
    otherParams, paramList, sourceDsDetails, targetDsDetails, rowIdentifier, sourceAttrDetails, featureAttrDetails, sourceQuery, \
    inputSourceFileName, init, activation, optimizer, loss, average, encodingDetails, imputationDetails, modelSchema, sourceSchema, \
    inputColList, trainSetDetails, saveTrainingSet, trainSetDsType, trainSetTableName, trainSetSavePath, trainSetHostName, trainSetDbName, \
    trainSetPort, trainSetUserName, trainSetPassword, trainSetDriver, trainSetUrl, testSetDetails, testSetDsType, testSetTableName, \
    testSetSavePath, testSetHostName, testSetDbName, testSetPort, testSetUserName, testSetPassword, testSetDriver, testSetUrl, \
    boosting_type, metric, samplingTech, samplingType, objective, calcFeatureWeight, trainOverwrite, testOverwrite, resultOverwrite, \
    inputColLis, booster, mode, algoType, algoName, trainClass, resourceParams, filters, kernel_size, padding, pool_size, dropout, kernelInitializer, strides, poolSize, \
    kernelSize, metrics, optimiser, loss, kernelInitializer, seed, base_score, colsample_bylevel, colsample_bytree, gamma, max_depth, min_child_weight, n_estimators, reg_alpha, reg_lambda, lin_alpha, scale_pos_weight, cvType, max_depth, max_delta_step, learning_rate, execVersion, hyperParam, restartMode, checkPoint, checkPointDir, scoring, subsample, saveTrainVersion, scalingDetails, algoType, saveVersion, \
    saveTestVersion, grow_policy, missing, nthread, tree_method, silent, importance_type, interaction_constraints, monotone_constraints, validate_parameters, num_parallel_tree, class_weight, criterion, solver, splitter, presort, penalty, multi_class, max_leaf_nodes, n_iter_no_change, min_impurity_split, l1_ratio, max_samples, bootstrap, oob_score, warm_start, fit_intercept, normalize, dual, in_it, precompute_distances, copy_x, algorithm, edges_path, nodes_path = none(
        133)

    flo = lambda n: [0.0 for _ in range(n)]
    trainPercent, testPercent, sub_feature, base_score, colsample_bytree, subsample, max_df, min_split_gain, min_weight_fraction_leaf, min_impurity_decrease, C, tol, ccp_alpha, alpha, validation_fraction \
        = flo(15)
    output_result = dict()
    probThreshold = 0.5
    logLevel = 0
    checkpoint_dir = "Y"

    os.environ["PYSPARK_PYTHON"] = sys.argv[1]
    os.environ["PYSPARK_DRIVER_PYTHON"] = "python3"
    SUBMIT_ARGS = "--packages mysql:mysql-connector-java:8.0.24  pyspark-shell"
    os.environ["PYSPARK_SUBMIT_ARGS"] = SUBMIT_ARGS
    pyspark_submit_args = ' --driver-memory ' + '1g' + ' pyspark-shell'
    os.environ["PYSPARK_SUBMIT_ARGS"] = pyspark_submit_args

    i = 0
    for eachArg in sys.argv:
        print(eachArg)
        if eachArg == "inputConfigFilePath":
            inputConfigFilePath = sys.argv[i + 1]
        i = i + 1

    print("Input Configuration File Path: ", inputConfigFilePath)

    with open(inputConfigFilePath, 'r') as json_file:
        input_config = json.load(json_file)

        for value in input_config:
            if value == "otherParams":
                otherParams = input_config[value]

        if otherParams is not None:
            for value in otherParams:
                if value == "docs_url": 
                    docs_url = otherParams[value]

                if value == "docs_path": 
                    docs_path = otherParams[value]




import os
import json
import asyncio
from time import sleep
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from playwright.async_api import async_playwright
import logging

# -----------------------------
# CONFIG
# -----------------------------
docs_url = docs_url
ALL_CONTENT_URL = f"{docs_url}/wiki/spaces/IID/pages"
SAVE_DIR = docs_path
LINKS_PATH = os.path.join(SAVE_DIR, "inferyx_doc_links.json")
INDEX_PATH = os.path.join(SAVE_DIR, "inferyx_faiss_index")
PROCESSED_PATH = os.path.join(SAVE_DIR, "indexed_urls.json")

# Setup
load_dotenv("/app/framework/test/env.txt")
os.makedirs(SAVE_DIR, exist_ok=True)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
assert GROQ_API_KEY, "❌ GROQ_API_KEY not found in .env file!"
assert GOOGLE_API_KEY, "❌ GOOGLE_API_KEY not found in .env file!"

# Logging (console only)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()  # Console logging only
    ]
)

# Initialize embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    task_type="semantic_similarity",
    google_api_key=GOOGLE_API_KEY
)

# -----------------------------
# UTILS
# -----------------------------
def load_processed_links():
    if os.path.exists(PROCESSED_PATH):
        with open(PROCESSED_PATH, "r") as f:
            return set(json.load(f))
    return set()

def save_processed_links(all_links):
    with open(PROCESSED_PATH, "w") as f:
        json.dump(list(all_links), f, indent=2)

def clean_text(text):
    """Clean and normalize text content"""
    if not text:
        return ""
    text = text.replace('\x00', '').strip()
    # Remove excessive whitespace and newlines
    text = ' '.join(text.split())
    return text

# -----------------------------
# STEP 1: SCRAPE LINKS
# -----------------------------
async def extract_all_doc_links():
    all_links = set()
    previous_count = -1

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        # Set timeout and retry logic
        page.set_default_timeout(30000)
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                logging.info(f"Attempt {attempt + 1}: Visiting {ALL_CONTENT_URL}")
                await page.goto(ALL_CONTENT_URL, wait_until="networkidle")
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                logging.warning(f"Failed to load page (attempt {attempt + 1}): {str(e)}")
                await asyncio.sleep(5)

        await page.wait_for_timeout(5000)

        for i in range(100):
            await page.mouse.wheel(0, 2000)
            await page.wait_for_timeout(5000)
            
            try:
                load_more = page.locator("button:has-text('Load more')")
                if await load_more.is_visible():
                    await load_more.click()
                    await page.wait_for_timeout(6000)
                    logging.info(f"'Load more' clicked on iteration {i}")
            except Exception as e:
                logging.warning(f"Error clicking 'Load more': {e}")

            anchors = await page.locator('a[href^="/wiki/spaces/IID/pages/"]').all()
            logging.info(f"Iteration {i}: Found {len(anchors)} anchors")
            
            for a in anchors:
                href = await a.get_attribute("href")
                if href:
                    full_link = docs_url + href
                    all_links.add(full_link)

            logging.info(f"Total collected links so far: {len(all_links)}")

            if len(all_links) == previous_count:
                logging.info(f"No new links found on iteration {i}, stopping scroll loop")
                break
            previous_count = len(all_links)

        await browser.close()

    if all_links:
        with open(LINKS_PATH, "w") as f:
            json.dump(list(all_links), f, indent=2)
        logging.info(f"Collected {len(all_links)} links and saved to {LINKS_PATH}")
    else:
        logging.warning(f"No links collected, NOT overwriting {LINKS_PATH}")

    return all_links

# -----------------------------
# STEP 2: FETCH DOCUMENTS
# -----------------------------
def fetch_docs(urls):
    docs = []
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    })

    for url in urls:
        try:
            res = session.get(url, timeout=10)
            res.raise_for_status()
            
            soup = BeautifulSoup(res.text, "html.parser")
            title = soup.title.string if soup.title else "Untitled"
            title = clean_text(title)[:500]
            
            content_div = soup.find("div", {"id": "main-content"}) or soup.body
            content = content_div.get_text(separator="\n", strip=True) if content_div else ""
            content = clean_text(content)
            
            docs.append({
                "title": title,
                "url": str(url),
                "content": content
            })
            logging.info(f"Fetched document: {url}")
            sleep(1)  # Be polite with requests
        except Exception as e:
            logging.error(f"Failed to fetch {url}: {e}")
    return docs

# -----------------------------
# STEP 3: VECTORIZE DOCUMENTS (Append to FAISS)
# -----------------------------
def build_index(new_docs):
    if not new_docs:
        logging.info("No new documents to index.")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Increased for better context
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    new_chunks = splitter.split_documents([
        Document(
            page_content=doc["content"],
            metadata={
                "title": doc["title"],
                "url": doc["url"],
                "source": doc["url"]
            }
        )
        for doc in new_docs
    ])

    try:
        if os.path.exists(INDEX_PATH):
            db = FAISS.load_local(
                INDEX_PATH,
                embeddings,
                allow_dangerous_deserialization=True  # Only if you trust the source
            )
            db.add_documents(new_chunks)
            logging.info(f"Added {len(new_chunks)} chunks to existing index")
        else:
            db = FAISS.from_documents(new_chunks, embeddings)
            logging.info(f"Created new index with {len(new_chunks)} chunks")

        db.save_local(INDEX_PATH)
        logging.info(f"✅ FAISS index saved to {INDEX_PATH}")
    except Exception as e:
        logging.error(f"Failed to build index: {str(e)}")
        raise

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    try:
        logging.info("🚀 Starting full pipeline...")

        # Step 1: Get all links from Confluence
        all_links = asyncio.run(extract_all_doc_links())

        # Step 2: Load previously indexed links
        processed_links = load_processed_links()

        # Step 3: Identify new links
        new_links = list(set(all_links) - processed_links)
        logging.info(f"🆕 Found {len(new_links)} new links to process.")

        if new_links:
            # Step 4: Fetch and vectorize only new documents
            new_docs = fetch_docs(new_links)
            build_index(new_docs)

            # Step 5: Save updated list of all links
            save_processed_links(all_links)
        else:
            logging.info("🛑 No new documents found. Skipping indexing.")

        logging.info("✅ All done.")
    except Exception as e:
        logging.error(f"❌ Pipeline failed: {str(e)}")
        raise
