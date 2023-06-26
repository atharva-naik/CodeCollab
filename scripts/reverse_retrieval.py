#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# reverse retrieval: match plan operators to individual code cells.

import json
import pandas as pd
from model.code_similarity_retrievers.dense import CodeBERTDenseSearcher

plan_ops = json.load(open("./data/juice-dataset/plan_ops.json"))
# ontology = load_ontology()

codes_KB = json.load(open("/home/arnaik/CodeCollab/JuICe_train_code_KB.json"))
codes = list(codes_KB.keys())[:10000]
codes_to_nbids = json.load(open("/home/arnaik/CodeCollab/JuICe_train_code_to_nbids.json"))
dense_searcher = CodeBERTDenseSearcher(
    ckpt_path="./experiments/CoNaLa_CSN_CodeBERT_CodeSearch2_CosSim/best_model.pt", 
    faiss_index_path="./dense_indices/CoNaLa_CSN_CodeBERT_CodeSearch2_CosSim/codebert_plan_ops_cos_sim.index"
)
# seed_queries = list(json.load(open("./data/juice-dataset/seed_queries.json")).keys())
result = dense_searcher.search(codes, obf_code=False, k=100)

code2plan_ops = {}
for i, code in enumerate(codes):
    idx = result[1][i][0]
    score = result[0][i][0].astype(float)
    code2plan_ops[code] = (plan_ops[idx], score)
with open("./experiments/CoNaLa_CSN_CodeBERT_CodeSearch2_CosSim/code2seed_plan_ops.json", "w") as f:
    json.dump(code2plan_ops, f, indent=4)