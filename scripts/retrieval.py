#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# retrieval: to assign codes to plan operators (as NL queries)

import json
import pandas as pd
from model.code_similarity_retrievers.dense import CodeBERTDenseSearcher

seed_queries = json.load(open("./data/juice-dataset/plan_ops.json"))

codes_KB = json.load(open("/home/arnaik/CodeCollab/JuICe_train_code_KB.json"))
codes = list(codes_KB.keys())
codes_to_nbids = json.load(open("/home/arnaik/CodeCollab/JuICe_train_code_to_nbids.json"))
dense_searcher = CodeBERTDenseSearcher(
    ckpt_path="./experiments/CoNaLa_CSN_CodeBERT_CodeSearch2_CosSim/best_model.pt", 
    faiss_index_path="./dense_indices/CoNaLa_CSN_CodeBERT_CodeSearch2_CosSim/codebert_cos_sim.index"
)
# seed_queries = list(json.load(open("./data/juice-dataset/seed_queries.json")).keys())
result = dense_searcher.search(seed_queries, text_query=True, k=100)

op2code = {}
for i,q in enumerate(seed_queries):
    op2code[q] = [(codes[idx], score.astype(float)) for score, idx in zip(result[0][i], result[1][i])]
with open("./experiments/CoNaLa_CSN_CodeBERT_CodeSearch2_CosSim/seed_query_op2code.json", "w") as f:
    json.dump(op2code, f, indent=4)