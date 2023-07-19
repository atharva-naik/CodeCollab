import os
import ast
import json
import parse
import torch
import pandas as pd
from typing import *
from collections import defaultdict
from sentence_transformers import util
from model.code_search import CodeBERTSimModel
from data.FCDS.code_chunking import extract_op_chunks
from model.code_similarity_retrievers.dense import CodeBERTDenseSearcher

def generalize_plan_op(plan_op: str):
    for canon_op, template in {
        "convert D1 to D2": "convert {} to {}",
        "function to do X": "function to do {}",
        "compute S": "compute {}", "create D": "create {}",
        "copy D": "copy {}", "populate D": "populate {}",
        "iterate over D": "iterate over {}",
        "data filtering": "data filtering conditison",
    }.items():
        op = parse.parse(template, plan_op)
        if op is not None: return canon_op

    return plan_op

def load_plan_op_data(
        path: str="./data/FCDS/FCDS Plan Operator Annotations.csv",
        canonicalize_plan_operator: bool=True, 
        filt_ops_list: List[str]=[],
        filt_mode: str="remove",
    ):
    assert filt_mode in ["keep", "remove"], "filt_mode should be `keep` or `remove`"
    annotations = pd.read_csv(path)
    plan_op_names = set() # unique plan operator names
    code_to_plan_op_data = []
    for rec in annotations.to_dict("records"):
        plan_op_string = rec["plan operator"]
        if isinstance(plan_op_string, str):
            if plan_op_string == "SKIP": continue
            plan_ops = []
            for plan_op in plan_op_string.split(";"):
                plan_op = plan_op.strip()
                if plan_op in filt_ops_list and filt_mode == "remove": continue
                # print(plan_op, filt_ops_list)
                elif plan_op not in filt_ops_list and filt_mode == "keep": continue
                if canonicalize_plan_operator:
                    plan_op = generalize_plan_op(plan_op)
                plan_ops.append(plan_op)
                plan_op_names.add(plan_op)
            if len(plan_ops) > 0:
                code_to_plan_op_data.append((rec["block"], plan_ops))
    plan_op_names = sorted(list(plan_op_names))

    return plan_op_names, code_to_plan_op_data

def load_primer_plan_ops(path: str="./data/FCDS/primer_only_plan_ops.json"):
    data = json.load(open(path))
    all_codes = []
    code_plan_op_mapping = {}
    for plan_op, rec in data.items():
        for code in rec['codes']:
            code_plan_op_mapping[code] = (plan_op, rec['path']) 
        all_codes += rec["codes"]

    return all_codes, code_plan_op_mapping

def zero_shot_ccsim_plan_op_map(code_to_plan_op_data):
    # intialize code similarity model and load checkpoint.
    codebert_code_sim_model = CodeBERTSimModel()
    ckpt_path = "./experiments/CoNaLa_CSN_CodeBERT_CodeSim_CosSim/best_model.pt"
    state_dict = torch.load(ckpt_path, map_location="cpu")["model_state_dict"]
    codebert_code_sim_model.load_state_dict(state_dict)
    if torch.cuda.is_available(): codebert_code_sim_model.cuda()
    # load primer plan operators.
    all_codes, code_plan_op_mapping = load_primer_plan_ops()
    plan_op_mat = codebert_code_sim_model.encode_from_text(codes=all_codes, obf_code=True, norm_L2=True)
    X = [code for code,_ in code_to_plan_op_data]
    code_enc = codebert_code_sim_model.encode_from_text(codes=X, obf_code=False, norm_L2=True)
    preds = [
        code_plan_op_mapping[all_codes[i]][0] for i in (
                code_enc @ plan_op_mat.T
            ).argmax(axis=-1).tolist()
        ]
    plan_op_preds = defaultdict(lambda: [])
    for code, plan_op_pred in zip(X, preds):
        plan_op_preds[plan_op_pred].append(code)
    plan_op_preds = dict(plan_op_preds)
    with open("./model/poc/zero_shot_code2code_plan_op_map.json", "w") as f:
        json.dump(plan_op_preds, f, indent=4)

def zero_shot_code2code_plan_op_map(code_to_plan_op_data):
    codebert_dense_searcher = CodeBERTDenseSearcher(
        ckpt_path="./experiments/CoNaLa_CSN_CodeBERT_CodeSearch2_CosSim/best_model.pt",
        faiss_index_path=None,
    )
    all_codes, code_plan_op_mapping = load_primer_plan_ops()
    plan_op_mat = torch.as_tensor(codebert_dense_searcher.encode(
        queries=all_codes, 
        text_query=False, 
        use_cos_sim=True,
    ))
    X = [code for code,_ in code_to_plan_op_data]
    code_enc = torch.as_tensor(codebert_dense_searcher.encode(
        queries=X, text_query=False, 
        use_cos_sim=True
    ))
    preds = [code_plan_op_mapping[all_codes[i]][0] for i in torch.argmax(code_enc @ plan_op_mat.T, axis=-1).tolist()]
    plan_op_preds = defaultdict(lambda: [])
    for code, plan_op_pred in zip(X, preds):
        plan_op_preds[plan_op_pred].append(code)
    plan_op_preds = dict(plan_op_preds)
    with open("./model/poc/zero_shot_code2code_plan_op_map.json", "w") as f:
        json.dump(plan_op_preds, f, indent=4)

def zero_shot_eval_codebert_searcher(code_to_plan_op_data, plan_op_names):
    codebert_dense_searcher = CodeBERTDenseSearcher(
        ckpt_path="./experiments/CoNaLa_CSN_CodeBERT_CodeSearch2_CosSim/best_model.pt",
        faiss_index_path=None,
    )
    plan_op_mat = torch.as_tensor(codebert_dense_searcher.encode(
        queries=plan_op_names, obf_code=True,
        text_query=True, use_cos_sim=True
    ))
    X = [code for code,_ in code_to_plan_op_data]
    y = [plan_ops for _,plan_ops in code_to_plan_op_data]
    code_enc = torch.as_tensor(codebert_dense_searcher.encode(
        queries=X, text_query=False, 
        use_cos_sim=True, obf_code=True
    ))
    preds = [plan_op_names[i] for i in torch.argmax(code_enc @ plan_op_mat.T, axis=-1).tolist()]
    print(len(code_to_plan_op_data))
    # print(preds)
    acc = 0
    tot = 0
    for p,t in zip(preds, y):
        if p in t: acc += 1
        tot += 1
    acc = acc/tot
    print(round(100*acc, 2))

# main 
if __name__ == "__main__":
    # plan_op_annotations = pd.read_csv("./data/FCDS/FCDS Plan Operator Annotations.csv")
    plan_ops, data = load_plan_op_data(
        filt_ops_list=[
            "data filtering", "get unique values", "count unique values", "aggregation",
            "data melting", "reset index", "data grouping", "drop missing values"
        ],
        filt_mode="keep",
    )
    print(len(data))
    # print(plan_ops)
    # zero_shot_eval_codebert_searcher(
    #     data, plan_ops
    # )
    # zero_shot_code2code_plan_op_map(data)
    zero_shot_ccsim_plan_op_map(data)