#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# retrieval_ensemble: to assign codes to plan operators (as NL queries) using an ensemble of models (currently of the same type and usually a mix of obfuscated code and non obfuscated code)

import json
import pandas as pd
from model.code_similarity_retrievers.dense import EnsembleDenseCodeSearcher

seed_queries = json.load(open("./data/juice-dataset/plan_ops.json"))
ind: int = 0
model_type: str = ["codebert", "graphcodebert", "unixcoder"][ind]
model_name: str = ["CodeBERT", "GraphCodeBERT", "UniXcoder"][ind]
codes_KB = json.load(open("/home/arnaik/CodeCollab/JuICe_train_code_KB.json"))
codes = list(codes_KB.keys())
codes_to_nbids = json.load(open("/home/arnaik/CodeCollab/JuICe_train_code_to_nbids.json"))
dense_searcher = EnsembleDenseCodeSearcher(model_type=model_type)
result = dense_searcher.search(seed_queries, k=100)

op2code = {}
for i,q in enumerate(seed_queries):
    op2code[q] = [(codes[idx], score.astype(float)) for score, idx in zip(result[0][i], result[1][i])]
with open(f"./experiments/{model_name}_ensemble/seed_query_op2code.json", "w") as f:
    json.dump(op2code, f, indent=4)