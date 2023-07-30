import os
import ast
import json
import parse
import torch
import statistics
import pandas as pd
from typing import *
from tqdm import tqdm
from collections import defaultdict
from sentence_transformers import util
from model.code_search import CodeBERTSimModel
from data.FCDS.code_chunking import extract_op_chunks, extract_plan_op_chunks_v2, sort_and_remap_chunks
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

METADATA = {}
for q, submissions in json.load(open("./data/FCDS/code_qa_submissions.json")).items():
    for rec in submissions:
        METADATA[int(rec["qa_id"])] = rec
def load_plan_op_data(
        path: str="./data/FCDS/FCDS Plan Operator Annotations.csv",
        canonicalize_plan_operator: bool=True, filt_ops_list: List[str]=[],
        filt_mode: str="remove", annotated_data_only: bool=True,
        filter_by_tasks: Union[List[str], None]=None,
    ):
    global METADATA
    assert filt_mode in ["keep", "remove"], "filt_mode should be `keep` or `remove`"
    annotations = pd.read_csv(path)
    plan_op_names = set() # unique plan operator names
    code_to_plan_op_data = []
    for rec in annotations.to_dict("records"):
        plan_op_string = rec["plan operator"]
        # handle NaN/missing values by substituting with empty string
        if not isinstance(plan_op_string, str): plan_op_string = ""
        if plan_op_string == "SKIP": continue
        plan_ops = []
        if filter_by_tasks is not None and METADATA[rec["id"]]["task_name"] not in filter_by_tasks: continue
        for plan_op in plan_op_string.split(";"):
            if plan_op == "": continue
            plan_op = plan_op.strip()
            if plan_op in filt_ops_list and filt_mode == "remove": continue
            # print(plan_op, filt_ops_list)
            elif plan_op not in filt_ops_list and filt_mode == "keep": continue
            if canonicalize_plan_operator:
                plan_op = generalize_plan_op(plan_op)
            plan_ops.append(plan_op)
            plan_op_names.add(plan_op)
        if len(plan_ops) > 0 or not(annotated_data_only):
            code_to_plan_op_data.append((rec["block"], plan_ops))
    plan_op_names = sorted(list(plan_op_names))

    return plan_op_names, code_to_plan_op_data



def majority_vote_topk(code_enc, plan_op_mat, all_codes: list, code_plan_op_mapping: dict, k: int=5):
    return [statistics.mode([code_plan_op_mapping[all_codes[i]][0] for i in row]) for row in torch.topk(code_enc @ plan_op_mat.T, axis=-1, k=k).indices.tolist()]

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
        code_plan_op_mapping[
            all_codes[i]][0] for i in (
                code_enc @ plan_op_mat.T
            ).argmax(axis=-1).tolist()
        ]
    # preds = majority_vote_topk(code_enc, plan_op_mat, all_codes, code_plan_op_mapping)
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

def assign_plan_ops(chunks, model, plan_op_mat, 
                    plan_op_codes, code_plan_op_mapping):
    code_enc = model.encode_from_text(codes=chunks, obf_code=False, norm_L2=True)
    max_return = (code_enc @ plan_op_mat.T).max(axis=-1)
    plan_preds = [
        code_plan_op_mapping[
            plan_op_codes[i]
        ][0] for i in max_return.indices.tolist()    
    ]
    score_preds = max_return.values.tolist()
    assert len(plan_preds) == len(score_preds)
    # print(preds)
    return plan_preds, score_preds

def load_submissions_and_extract_chunks(data_path: str, tasks: List[str]=[]):
    codebert_code_sim_model = CodeBERTSimModel()
    ckpt_path = "./experiments/CoNaLa_CSN_CodeBERT_CodeSim_CosSim/best_model.pt"
    state_dict = torch.load(ckpt_path, map_location="cpu")["model_state_dict"]
    codebert_code_sim_model.load_state_dict(state_dict)
    if torch.cuda.is_available(): codebert_code_sim_model.cuda()
    # create representation of plan operation.
    plan_op_codes, code_plan_op_mapping = load_primer_plan_ops()
    plan_op_mat = codebert_code_sim_model.encode_from_text(codes=plan_op_codes, obf_code=True, norm_L2=True)

    data = json.load(open(data_path))
    code_and_chunks = defaultdict(lambda: [])
    syntax_err_ctr = 0
    zero_cons_err_ctr = 0
    one_cons_err_ctr = 0
    for intent, submissions in data.items():
        for sub in tqdm(submissions, desc=intent): 
            if sub['task_name'] not in tasks: continue
            code = sub["answer"]
            try:
                nodecode2id, codecons2id, chunks = extract_plan_op_chunks_v2(code)
                chunk_codes = [chunk["META_code"] for chunk in chunks]
                if len(chunk_codes) == 0: 
                    zero_cons_err_ctr += 1
                    continue
                elif len(chunk_codes) == 1: 
                    one_cons_err_ctr += 1
                    continue
                assigned_plan_ops, assigned_plan_op_scores = assign_plan_ops(
                    chunk_codes, codebert_code_sim_model, plan_op_mat, 
                    plan_op_codes, code_plan_op_mapping
                )
                # cons_codes_to_plan_op = {cons_code: (assigned_plan_op, assigned_plan_op_score) for cons_code, assigned_plan_op, assigned_plan_op_score in zip(cons_codes, assigned_plan_ops, assigned_plan_op_scores)}
                for chunk, plan_op, score in zip(chunks, assigned_plan_ops, assigned_plan_op_scores):
                    chunk["META_plan_op"] = plan_op
                    chunk["META_plan_op_score"] = score
                    # print(chunk['META_code'])
                    # print(chunk['META_plan_op'])
                    # print("------------")
                chunks = sort_and_remap_chunks(chunks)
                for chunk in chunks:
                    assert "META_plan_op" in chunk, f"{chunk['META_code']}"
                    assert "META_plan_op_score" in chunk, f"{chunk['META_code']}"
                code_and_chunks[intent].append({
                    "id": sub["id"], "qa_id": sub["qa_id"],
                    "code": code, "chunks": chunks,
                })
            except IndexError:
                print(code)
                print(list(codecons2id.keys()))
                exit()
            except SyntaxError: syntax_err_ctr += 1
    print(syntax_err_ctr, "syntax errors")
    print(zero_cons_err_ctr, "zero constructs errors")
    print(one_cons_err_ctr, "one construct errors")

    return dict(code_and_chunks)

def test1():
    # plan_op_annotations = pd.read_csv("./data/FCDS/FCDS Plan Operator Annotations.csv")
    plan_ops, data = load_plan_op_data(
        # filt_ops_list=[
        #     "data filtering", "get unique values", "count unique values", "aggregation",
        #     "data melting", "reset index", "data grouping", "drop missing values"
        # ],
        # filt_mode="keep",
        filter_by_tasks=["Movie streaming service dataset"], 
        annotated_data_only=False,
    )
    print(len(data))
    # print(plan_ops)
    # zero_shot_eval_codebert_searcher(
    #     data, plan_ops
    # )
    # zero_shot_code2code_plan_op_map(data)
    zero_shot_ccsim_plan_op_map(data)

def test_plan_op_annot():
    code_and_chunks = load_submissions_and_extract_chunks(
        data_path="./data/FCDS/code_qa_submissions.json",
        tasks=["Movie streaming service dataset"]
    )
    with open("./data/FCDS/code_qa_submissions_and_chunks.json", "w") as f:
        json.dump(code_and_chunks, f, indent=4)

def visualize_plan_ops_and_chunks():
    code_and_chunks = json.load(open("./data/FCDS/code_qa_submissions_and_chunks.json"))
    intents = list(code_and_chunks.keys())
    chunks = code_and_chunks[intents[0]][1]["chunks"]
    def format_v(v):
        if isinstance(v, int):
            return f"\x1b[31;1m{v}\x1b[0m"
        elif isinstance(v, list):
            ret_out = []
            for subv in v:
                ret_out.append(format_v(subv))
            return "["+", ".join(ret_out)+"]"
        else: return v
    for chunk in chunks:
        chunk_args = [f'{k}={format_v(v)}' for k,v in chunk.items() if not(k.startswith('META'))]
        print(f"\x1b[31;1m{chunk['META_id']}\x1b[0m. \x1b[34;1m{chunk['META_chunktype']}\x1b[0m({', '.join(chunk_args)}) | {chunk['META_plan_op']} ({chunk['META_plan_op_score']:.2f})")
        # print(f"\x1b[31;1m{chunk['META_id']}\x1b[0m")
        print(chunk['META_code'])

# main 
if __name__ == "__main__":
    # visualize_plan_ops_and_chunks()
    test_plan_op_annot()