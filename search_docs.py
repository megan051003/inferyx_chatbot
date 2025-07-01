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
                if value == "docs_path": 
                    docs_url = otherParams[value]



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
#load_dotenv("{docs_path}/.env")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
assert GROQ_API_KEY, "GROQ_API_KEY missing!"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_PATH = "{docs_path}/inferyx_faiss_index"

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
