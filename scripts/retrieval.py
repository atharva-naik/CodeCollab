#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# retrieval: to assign codes to plan operators (as NL queries)

import json
import pandas as pd
from model.code_similarity_retrievers.dense import CodeBERTDenseSearcher

seed_queries = json.load(open("./data/juice-dataset/plan_ops.json"))
exp_name: str = "CoNaLa_CSN_GraphCodeBERT_CodeSearch_CosSim"
model_type: str = ["codebert", "graphcodebert", "unixcoder"][1]
codes_KB = json.load(open("/home/arnaik/CodeCollab/JuICe_train_code_KB.json"))
codes = list(codes_KB.keys())
codes_to_nbids = json.load(open("/home/arnaik/CodeCollab/JuICe_train_code_to_nbids.json"))
dense_searcher = CodeBERTDenseSearcher(
    ckpt_path=f"./experiments/{exp_name}/best_model.pt", model_type=model_type,
    faiss_index_path=f"./dense_indices/{exp_name}/cos_sim.index",
)
# seed_queries = list(json.load(open("./data/juice-dataset/seed_queries.json")).keys())
result = dense_searcher.search(seed_queries, text_query=True, k=100)

op2code = {}
for i,q in enumerate(seed_queries):
    op2code[q] = [(codes[idx], score.astype(float)) for score, idx in zip(result[0][i], result[1][i])]
with open(f"./experiments/{exp_name}/seed_query_op2code.json", "w") as f:
    json.dump(op2code, f, indent=4)