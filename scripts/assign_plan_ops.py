#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# assign plan ops: use ontology to do hierarchical search for plan operators.

import json
import faiss
import torch
import numpy as np
import pandas as pd
from typing import *
from tqdm import tqdm
from datautils import read_jsonl, load_ontology
from transformers import RobertaModel, RobertaTokenizerFast
from model.code_similarity_retrievers.dense import CodeBERTDenseSearcher
from model.code_similarity_retrievers.datautils import CodeDataset
from model.code_search import CodeBERTSearchModel, RobertaTokenizer, JuICeKBNNCodeBERTCodeSearchDataset

# dense encoder based on CodeBERT
class CodeBERTEncoder:
    def __init__(self, path: str="microsoft/codebert-base", device: str="cuda:0",
                 ckpt_path: str="./experiments/CoNaLa_CSN_CodeBERT_CodeSearch2_CosSim/best_model.pt", 
                 tok_args: str = {
                    "return_tensors": "pt", "padding": "max_length",
                    "truncation": True, "max_length": 100,
                }):
        self.tok_args = tok_args
        self.tokenizer = RobertaTokenizer.from_pretrained(path)
        self.model = CodeBERTSearchModel(
            path, device=device,
        )
        self.device = device
        # ckpt_path = os.path.join(ckp, "best_model.pt")
        ckpt_dict = torch.load(ckpt_path, map_location="cpu")
        print(f"loaded model from: {ckpt_path}")
        self.model.load_state_dict(ckpt_dict["model_state_dict"])
        self.model.to(self.device)

    def encode(self, queries: List[str], 
               use_cos_sim: bool=True,
               use_tqdm: bool=False,
               dtype: str="code"):
        # inp = self.tokenizer(query, **self.tok_args)
        dataset = JuICeKBNNCodeBERTCodeSearchDataset(
            tokenizer=self.tokenizer,
            queries=queries,
            **self.tok_args,
        )
        dataloader = dataset.get_docs_loader()
        doc_mat = []
        for step, batch in tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            disable=not(use_tqdm),
        ):
            self.model.zero_grad()
            with torch.no_grad():
                for j in range(len(batch)): batch[j] = batch[j].to(self.device)
                doc_enc = self.model.encode(*batch, dtype=dtype).cpu().detach().tolist()
                doc_mat += doc_enc
        doc_mat = torch.as_tensor(doc_mat).numpy()
        # normalize by L2 norm if cosine similarity is being used.
        if use_cos_sim: # doc_mat = np.array(doc_mat)
            faiss.normalize_L2(doc_mat)
        
        return torch.as_tensor(doc_mat)

# load ontology and plan operators.
plan_ops = json.load(open("./data/juice-dataset/plan_ops.json"))
ontology = load_ontology()

# # load codes and pick the test code and mapping from code cells to NBs.
# codes_KB = json.load(open("/home/arnaik/CodeCollab/JuICe_train_code_KB.json"))
# test_code = list(codes_KB.keys())[0]
# codes_to_nbids = json.load(open("/home/arnaik/CodeCollab/JuICe_train_code_to_nbids.json"))

def load_all_nb_seqs(path: str="./data/juice-dataset/traindedup.jsonl") -> List[List[str]]:
    # load the deduplicated train data.
    data = read_jsonl(path)
    all_nbs = []
    for rec in data:
        nb_cells = []
        for cell in rec["context"][::-1]:
            if cell["cell_type"] == "code":
                nb_cells.append(cell["code"])
        nb_cells.append(rec["code"])
        all_nbs.append(nb_cells)

    return all_nbs

# def fetch_plan_repr(plan_op: str):
#     global plan_op2index
#     global plan_ops_vecs
#     ind = plan_op2index[plan_op]
#     vec = plan_ops_vecs[ind]

#     return vec

# def fetch_scores_from_plan_ops(plan_op_scores, plan_ops: List[str]) -> List[float]:
#     global plan_op2index
#     global plan_ops_vecs
#     global index2plan_op
#     idx = [plan_op2index[plan_op] for plan_op in plan_ops]
#     scores = plan_op_scores[idx]

#     return scores
def assign_plan_ops_to_nbs(nbs: List[List[str]], encoder, cutoff: int=1000, 
                           k: int=5, save_path: str="plan_op_assgns.json",
                           use_cos_sim: bool=True) -> dict:
    # load indexed plan operator vectors
    if use_cos_sim: faiss_index_path = "./dense_indices/CoNaLa_CSN_CodeBERT_CodeSearch2_CosSim/codebert_plan_ops_cos_sim.index"
    else: faiss_index_path = "./dense_indices/CoNaLa_CSN_CodeBERT_CodeSearch2_CosSim/codebert_plan_ops.index"
    plan_ops_index = faiss.read_index(faiss_index_path)
    plan_ops_vecs = torch.as_tensor(plan_ops_index.reconstruct_n(0,len(plan_ops)))
    # plan_op2index = {plan_op: i for i,plan_op in enumerate(plan_ops)}
    # index2plan_op = {i: plan_op for i,plan_op in enumerate(plan_ops)}
    ctr = 0
    nb_plan_op_assgn = {
        "faiss_index": faiss_index_path,
        "nb_assgns": [],
    }
    for train_nb in tqdm(nbs):
        code_enc = encoder.encode(train_nb, use_cos_sim=use_cos_sim)
        plan_op_scores = (code_enc @ plan_ops_vecs.T).T
        # structure for assigning plan operators to notebooks.
        nb_assgn = [{
            "code": train_nb[cell_id], 
            "true_label": "",
            f"top{k}_preds": [plan_ops[ind] for ind in idx.tolist()]
        } for cell_id, idx in enumerate(plan_op_scores.topk(axis=0, k=k).indices.T)]
        nb_plan_op_assgn["nb_assgns"].append(nb_assgn)
        # print(nb_assgn)
        ctr += 1
        if ctr % cutoff == 0: break
    with open(save_path, "w") as f:
        json.dump(nb_plan_op_assgn, f, indent=4)

    return nb_plan_op_assgn

# main
if __name__ == "__main__":
    # train set notebooks.
    train_nbs = load_all_nb_seqs()
    # dense code encoder.
    codebert_encoder = CodeBERTEncoder()
    assign_plan_ops_to_nbs(train_nbs, codebert_encoder, cutoff=100)

    # plan_ops = plan_ops # [node for node in ontology["root"].children]
    # scores = fetch_scores_from_plan_ops(
    #     plan_op_scores, plan_ops
    # )
    # print(root_scores.shape)