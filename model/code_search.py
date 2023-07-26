#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# code for training/fine-tuning CodePLMs for code search using bi-encoder setups.
import os
import sys
import json
import faiss
import torch
import random
import argparse
import numpy as np
from typing import *
from tqdm import tqdm
import torch.nn as nn
from torch.optim import AdamW
from model.datautils import *
import torch.nn.functional as F
from datautils import load_plan_ops
from model.unixcoder import UniXcoder
from torch.utils.data import DataLoader
from transformers import RobertaModel, RobertaTokenizer
from torchmetrics.functional import pairwise_cosine_similarity
# import the dataset classes needed for code search for various datasets.
from sklearn.metrics import label_ranking_average_precision_score as MRR_score


# seed
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# single encoder for obfuscation invariant code similarity
class CodeBERTSimModel(nn.Module):
    """
    This class implements the classic bi-encoder architecture
    for code search using two CodePLM instances.
    """
    def __init__(self, path: str="microsoft/codebert-base", 
                 device: str="cuda", use_cos_sim: bool=False,
                 rev_ret: bool=False, sym_ret: bool=False):
        super(CodeBERTSimModel, self).__init__()
        self.code_encoder = RobertaModel.from_pretrained(path)
        self.ce_loss = nn.CrossEntropyLoss()
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"moving model to device: {self.device}")
        self.to(device)
        self.use_cos_sim = use_cos_sim
        self.rev_ret = rev_ret
        self.sym_ret = sym_ret
        assert not(self.rev_ret == True and self.sym_ret == True), "can't simultaneously train for reverse and symmetric retrieval"
        self.tokenizer = RobertaTokenizer.from_pretrained(path)

    def encode_from_text(self, codes: str, obf_code: bool=False, norm_L2: bool=False):
        # assert self.tokenizer is not None, "need to provide tokenizer"
        if obf_code: codes = [obfuscate_code(code) for code in codes]
        tok_dict = self.tokenizer(
            codes, return_tensors="pt", padding="max_length", 
            max_length=100, truncation=True,
        )
        iids = tok_dict["input_ids"].to(self.device)
        attn = tok_dict["attention_mask"].to(self.device)
        with torch.no_grad():
            vec = self.encode(iids, attn).squeeze()
        if norm_L2: # normalize by L2 norm for using with cosine similarity.
            vec = vec.cpu().numpy() # need to do this as normalize_L2 is an inplace operation.
            faiss.normalize_L2(vec) 
            vec = torch.as_tensor(vec).to(self.device)

        return vec

    def forward(self, query_iids, query_attn, code_iids, code_attn):
        query_enc = self.code_encoder(query_iids, query_attn).pooler_output
        code_enc = self.code_encoder(code_iids, code_attn).pooler_output
        batch_size, _ = query_enc.shape
        target = torch.as_tensor(range(batch_size)).to(self.device)
        # if self.use_cos_sim:
        #     code_enc = F.normalize(code_enc) # does L2 norm by default on axis-1 (768 vector dim axis)
        #     query_enc = F.normalize(query_enc)
        # if self.rev_ret: scores = pairwise_cosine_similarity(code_enc, query_enc)
        # elif self.sym_ret: 
        #     scores = pairwise_cosine_similarity(code_enc, query_enc) + pairwise_cosine_similarity(query_enc, code_enc)
        # else: scores = pairwise_cosine_similarity(query_enc, code_enc)
        if self.rev_ret: scores = code_enc @ query_enc.T
        elif self.sym_ret:
            scores = code_enc @ query_enc.T + query_enc @ code_enc.T
        else: scores = query_enc @ code_enc.T
        loss = self.ce_loss(scores, target)

        return query_enc, code_enc, scores, loss

    def encode(self, iids, attn, dtype: str="code"):
        return self.code_encoder(iids, attn).pooler_output

# bi-encoder for code search.
class CodeBERTSearchModel(nn.Module):
    """
    This class implements the classic bi-encoder architecture
    for code search using two CodePLM instances.
    """
    def __init__(self, path: str="microsoft/codebert-base", 
                 device: str="cuda", use_cos_sim: bool=False,
                 rev_ret: bool=False, sym_ret: bool=False):
        super(CodeBERTSearchModel, self).__init__()
        self.code_encoder = RobertaModel.from_pretrained(path)
        self.query_encoder = RobertaModel.from_pretrained(path)
        self.ce_loss = nn.CrossEntropyLoss()
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"moving model to device: {self.device}")
        self.to(device)
        self.use_cos_sim = use_cos_sim
        self.rev_ret = rev_ret
        self.sym_ret = sym_ret
        assert not(self.rev_ret == True and self.sym_ret == True), "can't simultaneously train for reverse and symmetric retrieval"
        self.tokenizer = RobertaTokenizer.from_pretrained(path)

    def encode_from_text(self, text_or_code: str, dtype: str="text"):
        # assert self.tokenizer is not None, "need to provide tokenizer"
        tok_dict = self.tokenizer(
            text_or_code, return_tensors="pt", 
            padding="max_length", max_length=100,
            truncation=True,
        )
        iids = tok_dict["input_ids"].to(self.device)
        attn = tok_dict["attention_mask"].to(self.device)
        vec = self.encode(iids, attn, dtype).squeeze()
        # print(vec.shape)
        return vec

    def forward(self, query_iids, query_attn, code_iids, code_attn):
        query_enc = self.query_encoder(query_iids, query_attn).pooler_output
        code_enc = self.code_encoder(code_iids, code_attn).pooler_output
        batch_size, _ = query_enc.shape
        target = torch.as_tensor(range(batch_size)).to(self.device)
        # if self.use_cos_sim:
        #     code_enc = F.normalize(code_enc) # does L2 norm by default on axis-1 (768 vector dim axis)
        #     query_enc = F.normalize(query_enc)
        # if self.rev_ret: scores = pairwise_cosine_similarity(code_enc, query_enc)
        # elif self.sym_ret: 
        #     scores = pairwise_cosine_similarity(code_enc, query_enc) + pairwise_cosine_similarity(query_enc, code_enc)
        # else: scores = pairwise_cosine_similarity(query_enc, code_enc)
        if self.rev_ret: scores = code_enc @ query_enc.T
        elif self.sym_ret:
            scores = code_enc @ query_enc.T + query_enc @ code_enc.T
        else: scores = query_enc @ code_enc.T
        loss = self.ce_loss(scores, target)

        return query_enc, code_enc, scores, loss

    def encode(self, iids, attn, dtype: str="text"):
        if dtype == "text": return self.query_encoder(iids, attn).pooler_output
        elif dtype == "code": return self.code_encoder(iids, attn).pooler_output

class GraphCodeBERTSearchModel(nn.Module):
    """
    This class implements the classic bi-encoder architecture
    for code search using two CodePLM instances.
    """
    def __init__(self, path: str="microsoft/graphcodebert-base",
                 device: str="cuda", use_cos_sim: bool=False,
                 rev_ret: bool=False, sym_ret: bool=False):
        super(GraphCodeBERTSearchModel, self).__init__()
        self.code_encoder = RobertaModel.from_pretrained(path)
        self.query_encoder = RobertaModel.from_pretrained(path)
        self.ce_loss = nn.CrossEntropyLoss()
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"moving model to device: {self.device}")
        self.to(device)
        self.use_cos_sim = use_cos_sim
        self.rev_ret = rev_ret
        self.sym_ret = sym_ret
        assert not(self.rev_ret == True and self.sym_ret == True), "can't simultaneously train for reverse and symmetric retrieval"
        self.tokenizer = RobertaTokenizer.from_pretrained(path)

    def forward(self, query_iids, code_iids, code_attn, code_pos_idx):
        query_enc = self.encode(iids=query_iids, dtype="text")
        code_enc = self.encode(
            iids=code_iids, attn=code_attn, 
            pos_idx=code_pos_idx, dtype="code",
        )
        batch_size, _ = query_enc.shape
        target = torch.as_tensor(range(batch_size)).to(self.device)
        # if self.use_cos_sim:
        #     code_enc = F.normalize(code_enc) # does L2 norm by default on axis-1 (768 vector dim axis)
        #     query_enc = F.normalize(query_enc)
        if self.rev_ret: scores = code_enc @ query_enc.T
        elif self.sym_ret:
            scores = code_enc @ query_enc.T + query_enc @ code_enc.T
        else: scores = query_enc @ code_enc.T
        loss = self.ce_loss(scores, target)

        return query_enc, code_enc, scores, loss

    def encode(self, iids=None, attn=None, 
               pos_idx=None, dtype: str="text"):
        if dtype == "text": return self.query_encoder(iids, attention_mask=iids.ne(1))[1]
        elif dtype == "code": 
            nodes_mask=pos_idx.eq(0)
            token_mask=pos_idx.ge(2)        
            inputs_embeddings=self.code_encoder.embeddings.word_embeddings(iids)
            nodes_to_token_mask=nodes_mask[:,:,None]&token_mask[:,None,:]&attn
            nodes_to_token_mask=nodes_to_token_mask/(nodes_to_token_mask.sum(-1)+1e-10)[:,:,None]
            avg_embeddings=torch.einsum("abc,acd->abd",nodes_to_token_mask,inputs_embeddings)
            inputs_embeddings=inputs_embeddings*(~nodes_mask)[:,:,None]+avg_embeddings*nodes_mask[:,:,None]    
            return self.code_encoder(inputs_embeds=inputs_embeddings, attention_mask=attn, position_ids=pos_idx)[1]

class UniXcoderSearchModel(nn.Module):
    """
    This class implements the classic bi-encoder architecture
    for code search using two CodePLM instances.
    """
    def __init__(self, path: str="microsoft/unixcoder-base", 
                 device: str="cuda", use_cos_sim: bool=False,
                 rev_ret: bool=False, sym_ret: bool=False):
        super(UniXcoderSearchModel, self).__init__()
        self.code_encoder = UniXcoder(path)
        self.query_encoder = UniXcoder(path)
        self.ce_loss = nn.CrossEntropyLoss()
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"moving model to device: {self.device}")
        self.to(device)
        self.use_cos_sim = use_cos_sim
        self.rev_ret = rev_ret
        self.sym_ret = sym_ret
        assert not(self.rev_ret == True and self.sym_ret == True), "can't simultaneously train for reverse and symmetric retrieval"
        self.tokenizer = RobertaTokenizer.from_pretrained(path)

    def forward(self, query_iids, code_iids):
        _,query_enc = self.query_encoder(query_iids)
        _,code_enc = self.code_encoder(code_iids)
        batch_size, _ = query_enc.shape
        target = torch.as_tensor(range(batch_size)).to(self.device)
        # if self.use_cos_sim:
        #     code_enc = F.normalize(code_enc) # does L2 norm by default on axis-1 (768 vector dim axis)
        #     query_enc = F.normalize(query_enc)
        if self.rev_ret: scores = code_enc @ query_enc.T
        elif self.sym_ret:
            scores = code_enc @ query_enc.T + query_enc @ code_enc.T
        else: scores = query_enc @ code_enc.T
        loss = self.ce_loss(scores, target)

        return query_enc, code_enc, scores, loss

    def encode(self, iids, dtype: str="text"):
        if dtype == "text": return self.query_encoder(iids)[1]
        elif dtype == "code": return self.code_encoder(iids)[1]

def codebert_codesearch_val_old(model, dataloader, log_file, args):
    model.eval()
    tot, matches, batch_losses = 0, 0, []
    pbar = tqdm(
        enumerate(dataloader), 
        total=len(dataloader),
    )
    for step, batch in pbar:
        model.zero_grad()
        with torch.no_grad():
            for j in range(len(batch)):
                batch[j] = batch[j].to(args.device)
            _, _, scores, loss = model(*batch)
            batch_losses.append(loss.detach().cpu().item())
            preds = scores.cpu().argmax(dim=-1)
            # compute train classification accuracy for code search.
            batch_tot = len(batch[0])
            batch_matches = (preds == torch.as_tensor(range(batch_tot))).sum().item()
            # epoch level accuracy.
            tot += batch_tot
            matches += batch_matches
            pbar.set_description(
                f"V: bl: {batch_losses[-1]:.3f} l: {np.mean(batch_losses):.3f} ba: {(100*batch_matches/batch_tot):.2f} a: {(100*matches/tot):.2f}"
            )
            if step%args.log_steps == 0 or (step+1) == len(dataloader): 
                with open(log_file, "a") as f:
                    f.write(json.dumps({
                        "step": step,
                        "bl": batch_losses[-1],
                        "l": np.mean(batch_losses),
                        "ba": 100*batch_matches/batch_tot,
                        "a": 100*matches/tot,
                    })+"\n")

    return matches/tot

def recall_at_k_score(doc_ranks, doc_ids, k: int=5):
    tot, hits = 0, 0
    for i in range(len(doc_ids)):
        tot += 1
        # print(doc_ids[i], doc_ranks[i][:k])
        for id in doc_ids[i]:
            if id in doc_ranks[i][:k]:
                hits += 1
                break

    return hits/tot

def codesearch_create_index(model, dataset, args):
    """create index of documents (code) or queries (NL/plan operators)."""
    model.eval()
    d_loader = dataset.get_docs_loader(batch_size=args.batch_size)
    dbar = tqdm(
        enumerate(d_loader), 
        total=len(d_loader),
    )
    # create a Faiss CPU index for large scale NN search (using inner product metric).
    index = faiss.IndexFlatIP(768) # TODO: change hard coded vector dim
    for step, batch in dbar:
        model.zero_grad()
        with torch.no_grad():
            for j in range(len(batch)): batch[j] = batch[j].to(args.device)
            q_enc = model.encode(*batch, dtype=args.index_modality).cpu().detach().numpy()
            if args.cos_sim: faiss.normalize_L2(q_enc)
            index.add(q_enc)
        # if step == 10: break
    # write the index to a file
    index_dir = os.path.join("./dense_indices", args.experiment_name)
    os.makedirs(index_dir, exist_ok=True)
    index_file_path = os.path.join(index_dir, args.index_file_name)
    faiss.write_index(index, index_file_path)
    
    # # search over the index with a query
    # query_vector = torch.randn(1, vector_dim)
    # D, I = index.search(query_vector.numpy(), k=5)

    # print("Query vector:\n", query_vector)
    # print("D (distances):\n", D)
    # print("I (indices):\n", I)
def codesearch_val(model, dataset, args):
    """use MRR to validate and pick best model (instead of batch level -ve sample acc.)"""
    model.eval()
    tot, matches, batch_losses = 0, 0, []
    q_loader = dataset.get_query_loader()
    d_loader = dataset.get_docs_loader()

    qbar = tqdm(
        enumerate(q_loader), 
        total=len(q_loader),
    )
    dbar = tqdm(
        enumerate(d_loader), 
        total=len(d_loader),
    )
    q_mat, d_mat = [], []
    for step, batch in qbar:
        model.zero_grad()
        with torch.no_grad():
            for j in range(len(batch)): batch[j] = batch[j].to(args.device)
            q_enc = model.encode(*batch, dtype="text").cpu().detach().tolist()
            q_mat += q_enc
    for step, batch in dbar:
        model.zero_grad()
        with torch.no_grad():
            for j in range(len(batch)): batch[j] = batch[j].to(args.device)
            d_enc = model.encode(*batch, dtype="code").cpu().detach().tolist()
            d_mat += d_enc
    q_mat = torch.as_tensor(q_mat).to(args.device)
    d_mat = torch.as_tensor(d_mat).to(args.device)
    if args.cos_sim:
        scores = pairwise_cosine_similarity(q_mat, d_mat).cpu()
    else: scores = (q_mat @ d_mat.T).cpu()
    doc_ids = dataset.doc_ids 
    trues = np.array(dataset.get_doc_ids_mask())
    print(scores.shape, trues.shape)
    doc_ranks = scores.argsort(dim=-1, descending=True).numpy()
    mrr = MRR_score(trues, scores.numpy())
    recalls = {}
    for k in [1,5,10]:
        recalls[f"@{k}"] = recall_at_k_score(doc_ranks, doc_ids, k)
    print(f"MRR: {mrr}")
    print(f"recalls: {recalls}")

    return mrr, recalls

def print_args(args):
    for k,v in vars(args).items():
        print(f"{k}: {v}")

def codesearch_test_only(args):
    device = args.device
    if args.model_type == "codebert":
        codesearch_biencoder = CodeBERTSearchModel(
            args.model_path, device=device,
            use_cos_sim=args.cos_sim,
            rev_ret=args.rev_ret,
            sym_ret=args.sym_ret,
        )
        tok_args = {
            "return_tensors": "pt",
            "padding": "max_length",
            "truncation": True,
            "max_length": 100,
        }
    elif args.model_type == "graphcodebert":
        codesearch_biencoder = GraphCodeBERTSearchModel(
            args.model_path, device=device,
            use_cos_sim=args.cos_sim,
            rev_ret=args.rev_ret,
            sym_ret=args.sym_ret,
        )
        tok_args = {
            "nl_length": 100, 
            "code_length": 100, 
            "data_flow_length": 64
        }
    elif args.model_type == "unixcoder":
        codesearch_biencoder = UniXcoderSearchModel(
            args.model_path, device=device,
            use_cos_sim=args.cos_sim,
            rev_ret=args.rev_ret,
            sym_ret=args.sym_ret,
        )
        tok_args = {
            "return_tensors": "pt",
            "padding": "max_length",
            "truncation": True,
            "max_length": 100,
            # "mode": "<encoder-only>",
        }
    if args.model_type in ["codebert", "graphcodebert", "unixcoder"]:
        tokenizer = RobertaTokenizer.from_pretrained(args.model_path)

    config = vars(args)
    config["tokenizer_args"] = tok_args
    if args.model_type == "codebert":
        # trainset = CodeSearchNetCodeBERTCodeSearchDataset(split="train", tokenizer=tokenizer, **tok_args)
        valset = CodeSearchNetCodeBERTCodeSearchDataset(split="val", tokenizer=tokenizer, **tok_args)
        testset = CodeSearchNetCodeBERTCodeSearchDataset(split="test", tokenizer=tokenizer, **tok_args)
    elif args.model_type == "graphcodebert":
        # trainset = CoNaLaGraphCodeBERTCodeSearchDataset(split="train", tokenizer=tokenizer, **tok_args)
        valset = CoNaLaGraphCodeBERTCodeSearchDataset(split="val", tokenizer=tokenizer, **tok_args)
        testset = CoNaLaGraphCodeBERTCodeSearchDataset(split="test", tokenizer=tokenizer, **tok_args)
    elif args.model_type == "unixcoder":
        # trainset = CoNaLaUniXcoderCodeSearchDataset(split="train", tokenizer=tokenizer, **tok_args)
        valset = CoNaLaUniXcoderCodeSearchDataset(split="val", tokenizer=tokenizer, **tok_args)
        testset = CoNaLaUniXcoderCodeSearchDataset(split="test", tokenizer=tokenizer, **tok_args)
    # testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)
    exp_folder = os.path.join(
        "experiments", 
        args.experiment_name,
    )
    # create new folder for the experiment.
    # os.makedirs(exp_folder, exist_ok=True)
    # config_path = os.path.join(exp_folder, "config.json")
    # checkpoint path
    ckpt_path = os.path.join(exp_folder, "best_model.pt")
    ckpt_dict = torch.load(ckpt_path, map_location="cpu")
    print(f"loaded model from: {ckpt_path}")
    codesearch_biencoder.load_state_dict(ckpt_dict["model_state_dict"])
    print_args(args)

    print("val metrics:")
    codesearch_val(codesearch_biencoder, valset, args)
    print("test metrics:")
    codesearch_val(codesearch_biencoder, testset, args)

def create_dense_index(args):
    device = args.device
    if args.model_type == "codebert":
        codesearch_biencoder = CodeBERTSearchModel(
            args.model_path, device=device,
            use_cos_sim=args.cos_sim,
            rev_ret=args.rev_ret,
            sym_ret=args.sym_ret,
        )
        tok_args = {
            "return_tensors": "pt",
            "padding": "max_length",
            "truncation": True,
            "max_length": 100,
        }
    elif args.model_type == "graphcodebert":
        codesearch_biencoder = GraphCodeBERTSearchModel(
            args.model_path, device=device,
            use_cos_sim=args.cos_sim,
            rev_ret=args.rev_ret,
            sym_ret=args.sym_ret,
        )
        tok_args = {
            "nl_length": 100, 
            "code_length": 100, 
            "data_flow_length": 64
        }
    elif args.model_type == "unixcoder":
        codesearch_biencoder = UniXcoderSearchModel(
            args.model_path, device=device,
            use_cos_sim=args.cos_sim,
            rev_ret=args.rev_ret,
            sym_ret=args.sym_ret,
        )
        tok_args = {
            "return_tensors": "pt",
            "padding": "max_length",
            "truncation": True,
            "max_length": 100,
            # "mode": "<encoder-only>",
        }
    if args.model_type in ["codebert", "graphcodebert", "unixcoder"]:
        tokenizer = RobertaTokenizer.from_pretrained(args.model_path)
    exp_folder = os.path.join(
        "experiments", 
        args.experiment_name,
    )
    ckpt_path = os.path.join(exp_folder, "best_model.pt")
    ckpt_dict = torch.load(ckpt_path, map_location="cpu")
    print(f"loaded model from: {ckpt_path}")
    codesearch_biencoder.load_state_dict(ckpt_dict["model_state_dict"])

    config = vars(args)
    config["tokenizer_args"] = tok_args    

    queries = None
    if args.index_modality == "text":
        queries = json.load(open("./data/juice-dataset/plan_ops.json"))
    if args.model_type == "codebert":
        dataset = JuICeKBNNCodeBERTCodeSearchDataset(
            tokenizer=tokenizer, queries=queries, 
            obf_code=args.obfuscate_code, 
            **tok_args,
        )
    elif args.model_type == "graphcodebert":
        dataset = JuICeKBNNGraphCodeBERTCodeSearchDataset(
            tokenizer=tokenizer, queries=queries, 
            obf_code=args.obfuscate_code,
            **tok_args,
        )
    elif args.model_type == "unixcoder":
        dataset = JuICeKBNNUniXcoderCodeSearchDataset(
            tokenizer=tokenizer, queries=queries, 
            obf_code=args.obfuscate_code, 
            **tok_args,
        )
    print_args(args)
    codesearch_create_index(codesearch_biencoder, dataset, args)

def codesearch_finetune(args):
    device = args.device
    if args.model_type == "codebert":
        if args.code_sim_mode:
            codesearch_biencoder = CodeBERTSimModel(
                args.model_path, device=device,
                use_cos_sim=args.cos_sim,
                rev_ret=args.rev_ret,
                sym_ret=args.sym_ret,
            )
        else:
            codesearch_biencoder = CodeBERTSearchModel(
                args.model_path, device=device,
                use_cos_sim=args.cos_sim,
                rev_ret=args.rev_ret,
                sym_ret=args.sym_ret,
            )
        tok_args = {
            "return_tensors": "pt",
            "padding": "max_length",
            "truncation": True,
            "max_length": 100,
        }
    elif args.model_type == "graphcodebert":
        codesearch_biencoder = GraphCodeBERTSearchModel(
            args.model_path, device=device,
        )
        tok_args = {
            "nl_length": 100, 
            "code_length": 100, 
            "data_flow_length": 64
        }
    elif args.model_type == "unixcoder":
        codesearch_biencoder = UniXcoderSearchModel(
            args.model_path, device=device,
        )
        tok_args = {
            "return_tensors": "pt",
            "padding": "max_length",
            "truncation": True,
            "max_length": 100,
            # "mode": "<encoder-only>",
        }
    
    # check if a pre-saved checkpoint should be loaded for the model before fine-tuning.
    if args.checkpoint_path is not None:
        ckpt_path = args.checkpoint_path
        ckpt_dict = torch.load(ckpt_path, map_location="cpu")
        print(f"loaded model from: {ckpt_path}")
        codesearch_biencoder.load_state_dict(ckpt_dict["model_state_dict"])
    
    # initialize the tokenizer.
    if args.model_type in ["codebert", "graphcodebert", "unixcoder"]:
        tokenizer = RobertaTokenizer.from_pretrained(args.model_path)

    config = vars(args)
    config["tokenizer_args"] = tok_args
    
    if args.model_type == "codebert":
        if args.dataset_name == "CoNaLa+CodeSearchNet": # use CoNaLa & CodeSearchNet.
            if not(args.code_sim_mode):
                trainset = CodeSearchNetCodeBERTCodeSearchDataset(split="train", obf_code=args.obfuscate_code, 
                                                                 folder=args.data_dir, tokenizer=tokenizer, **tok_args)
                valset = CodeSearchNetCodeBERTCodeSearchDataset(split="val", obf_code=args.obfuscate_code,
                                                                folder=args.data_dir, tokenizer=tokenizer, **tok_args)
                testset = CodeSearchNetCodeBERTCodeSearchDataset(split="test", obf_code=args.obfuscate_code,
                                                                folder=args.data_dir, tokenizer=tokenizer, **tok_args)
            else: 
                trainset = CSNCoNaLaCodeBERTCodeSimDataset(split="train", folder=args.data_dir, 
                                                           tokenizer=tokenizer, **tok_args)
                valset = CSNCoNaLaCodeBERTCodeSimDataset(split="val", folder=args.data_dir, 
                                                         tokenizer=tokenizer, **tok_args)
                testset = CSNCoNaLaCodeBERTCodeSimDataset(split="test", folder=args.data_dir, 
                                                          tokenizer=tokenizer, **tok_args)
        elif args.dataset_name == "JuICe": # use JuICe dataset.
            trainset = JuICeCodeBERTCodeSearchDataset(split="train", obf_code=args.obfuscate_code, 
                                                      tokenizer=tokenizer, **tok_args)
            valset = JuICeCodeBERTCodeSearchDataset(split="dev", obf_code=args.obfuscate_code, 
                                                    tokenizer=tokenizer, **tok_args)
            testset = JuICeCodeBERTCodeSearchDataset(split="test", obf_code=args.obfuscate_code, 
                                                     tokenizer=tokenizer, **tok_args)
    elif args.model_type == "graphcodebert":
        trainset = CodeSearchNetGraphCodeBERTCodeSearchDataset(split="train", obf_code=args.obfuscate_code,
                                                               folder=args.data_dir, tokenizer=tokenizer, **tok_args)
        valset = CodeSearchNetGraphCodeBERTCodeSearchDataset(split="val", obf_code=args.obfuscate_code,
                                                             folder=args.data_dir, tokenizer=tokenizer, **tok_args)
        testset = CodeSearchNetGraphCodeBERTCodeSearchDataset(split="test", obf_code=args.obfuscate_code,
                                                              folder=args.data_dir, tokenizer=tokenizer, **tok_args)
        # trainset = CoNaLaGraphCodeBERTCodeSearchDataset(split="train", obf_code=args.obfuscate_code,
        #                                                 folder=args.data_dir, tokenizer=tokenizer, **tok_args)
        # valset = CoNaLaGraphCodeBERTCodeSearchDataset(split="val", obf_code=args.obfuscate_code,
        #                                               folder=args.data_dir, tokenizer=tokenizer, **tok_args)
        # testset = CoNaLaGraphCodeBERTCodeSearchDataset(split="test", obf_code=args.obfuscate_code,
        #                                                folder=args.data_dir, tokenizer=tokenizer, **tok_args)
    elif args.model_type == "unixcoder":
        trainset = CodeSearchNetUniXcoderCodeSearchDataset(split="train", obf_code=args.obfuscate_code,
                                                           folder=args.data_dir, tokenizer=tokenizer, **tok_args)
        valset = CodeSearchNetUniXcoderCodeSearchDataset(split="val", obf_code=args.obfuscate_code,
                                                         folder=args.data_dir, tokenizer=tokenizer, **tok_args)
        testset = CodeSearchNetUniXcoderCodeSearchDataset(split="test", obf_code=args.obfuscate_code,
                                                          folder=args.data_dir, tokenizer=tokenizer, **tok_args)
        # trainset = CoNaLaUniXcoderCodeSearchDataset(split="train", obf_code=args.obfuscate_code,
        #                                             folder=args.data_dir, tokenizer=tokenizer, **tok_args)
        # valset = CoNaLaUniXcoderCodeSearchDataset(split="val", obf_code=args.obfuscate_code,
        #                                           folder=args.data_dir, tokenizer=tokenizer, **tok_args)
        # testset = CoNaLaUniXcoderCodeSearchDataset(split="test", obf_code=args.obfuscate_code,
        #                                            folder=args.data_dir, tokenizer=tokenizer, **tok_args)

    optimizer = AdamW(
        codesearch_biencoder.parameters(),
        lr=args.learning_rate,
    )
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    # valloader = DataLoader(valset, batch_size=args.batch_size, shuffle=False)
    # testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)
    exp_folder = os.path.join(
        "experiments", 
        args.experiment_name,
    )
    
    # create new folder for the experiment.
    os.makedirs(exp_folder, exist_ok=True)
    config_path = os.path.join(exp_folder, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    print_args(args)
    # create directory for logging stats.
    best_val_acc = 0
    logs_dir = os.path.join(exp_folder, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    for epoch in range(args.epochs):
        val_log_file = os.path.join(logs_dir, f"val_epoch_{epoch+1}.jsonl")
        train_log_file = os.path.join(logs_dir, f"train_epoch_{epoch+1}.jsonl")
        open(val_log_file, "w")
        open(train_log_file, "w")
        tot, matches, batch_losses = 0, 0, []
        pbar = tqdm(
            enumerate(trainloader),
            total=len(trainloader),
        )

        for step, batch in pbar:
            codesearch_biencoder.train()
            codesearch_biencoder.zero_grad()
            for j in range(len(batch)):
                batch[j] = batch[j].to(device)
            _, _, scores, loss = codesearch_biencoder(*batch)
            # model update.
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.detach().cpu().item())
            preds = scores.cpu().argmax(dim=-1)
            # compute train classification accuracy for code search.
            batch_tot = len(batch[0])
            batch_matches = (preds == torch.as_tensor(range(batch_tot))).sum().item()
            # print(batch_tot, batch_matches)
            # epoch level accuracy.
            tot += batch_tot
            matches += batch_matches
            pbar.set_description(
                f"T: {epoch+1}/{args.epochs}: bl: {batch_losses[-1]:.3f} l: {np.mean(batch_losses):.3f} ba: {(100*batch_matches/batch_tot):.2f} a: {(100*matches/tot):.2f}"
            )
            if step%args.log_steps == 0 or (step+1) == len(trainloader): 
                with open(train_log_file, "a") as f:
                    f.write(json.dumps({
                        "step": step,
                        "bl": batch_losses[-1],
                        "l": np.mean(batch_losses),
                        "ba": 100*batch_matches/batch_tot,
                        "a": 100*matches/tot,
                    })+"\n")
            # if step == 10:
            if (step+1)%args.eval_steps == 0 or (step+1) == len(trainloader):
                # val_log_file = os.path.join(logs_dir, f"val_epoch_{epoch+1}_step_{step}.jsonl")
                val_acc, val_recalls = codesearch_val(codesearch_biencoder, valset, args)
                with open(val_log_file, "a") as f:
                    f.write(json.dumps({
                        "step": step,
                        "MRR": val_acc,
                        "recalls": val_recalls,
                    })+"\n")
                # `val_acc` is MRR.
                # save the best model.
                if val_acc > best_val_acc:
                    save_dict = {
                        "step": step, "epoch": epoch+1, 
                        "val_MRR": best_val_acc, "val_recalls": val_recalls,
                        "model_state_dict": codesearch_biencoder.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    }
                    model_save_path = os.path.join(
                        exp_folder, "best_model.pt"
                    )
                    torch.save(save_dict, model_save_path)
    test_log_file = os.path.join(exp_folder, f"test_log.jsonl")
    MRR, recalls = codesearch_val(codesearch_biencoder, testset, args)
    with open(test_log_file, "a") as f:
        f.write(json.dumps({"MRR": MRR, "recalls": recalls})+"\n")

# get training/validation arguments.
def get_args():
    parser = argparse.ArgumentParser(
        description='''script to finetune CodeBERT for 
code search using a bi-encoder setup''')
    parser.add_argument('-d', '--device', default="cuda", type=str, 
                        help="device to be used for fine-tuning")
    parser.add_argument('-bs', '--batch_size', default=100, 
                        type=int, help="batch size used for training")
    parser.add_argument("-mt", "--model_type", type=str,
                        default="codebert", help="model class to be used",
                        choices=["codebert", "graphcodebert", "unixcoder"])
    parser.add_argument('-mp', "--model_path", type=str,
                        default="microsoft/codebert-base",
                        help="path to model/hf model name")
    parser.add_argument("-lr", "--learning_rate", type=float,
                        default=1e-5, help="learning rate for training")
    parser.add_argument("-exp", "--experiment_name", required=True, 
                        help="name of the experiment")
    parser.add_argument("-e", "--epochs", type=int, default=5, 
                        help="number of epochs for training")
    parser.add_argument("-ls", "--log_steps", type=int, default=20,
                        help="log training stats after these many steps")
    parser.add_argument("-es", "--eval_steps", type=int, default=500,
                        help="do validation after these many steps")
    parser.add_argument("--retrieval_result_dump_path", type=str,
                        help="where to store the inferred data")
    parser.add_argument("-sr", "--sym_ret", action="store_true", help="use symmetric loss")
    parser.add_argument("-rr", "--rev_ret", action="store_true", help="use reverse loss")
    parser.add_argument("-obf", "--obfuscate_code", action="store_true",
                        help="obfuscate variable names and function defs")
    parser.add_argument("--data_dir", type=str, required=False,
                        default="./data/CoNaLa", help="where to load the data")
    parser.add_argument("--mode", type=str, choices=['train', 'inference'], default='train',
                        help="should train or infer?")
    parser.add_argument("--cos_sim", action="store_true", help="use cosine similarity in contrastive loss")
    parser.add_argument("--index_modality", type=str, default="code", help="what kind of modality is indexed")
    parser.add_argument("--index_file_name", type=str, default="codebert_cos_sim.index")
    parser.add_argument("-ckpt", "--checkpoint_path", default=None, type=str, 
                        help="load the model from this checkpoint while fine-tuning")
    parser.add_argument("-ccs", "--code_sim_mode", action="store_true", help="code similarity training")
    parser.add_argument("-dn", "--dataset_name", type=str, default="CoNaLa+CodeSearchNet", help="dataset name")
    args = parser.parse_args()

    return args

# main
if __name__ == "__main__":
    args = get_args()
    if args.mode == 'train':
        codesearch_finetune(args)
    elif args.mode == "inference":
        create_dense_index(args)

# Various commands used:

# for CodeBERT continual fine-tuning training on JuICe:
# python -m model.code_search -exp JuICe_CodeBERT_CodeSearch2_CosSim -bs 200 --mode train -e 50 -ckpt ./experiments/CoNaLa_CSN_CodeBERT_CodeSearch2_CosSim/best_model.pt -dn JuICe

# python -m src.e_ret.code_search -exp CoNaLa_CodeBERT_CodeSearch
# python -m src.e_ret.code_search -mt unixcoder -exp CoNaLa_UniXcoder_CodeSearch -bs 85 -mp microsoft/unixcoder-base
# python -m src.e_ret.code_search -mt graphcodebert -exp CoNaLa_GraphCodeBERT_CodeSearch -bs 65 -mp microsoft/graphcodebert-base
# python -m src.e_ret.code_search -mt codebert -exp CoNaLa_CodeBERT_Python_CodeSearch -bs 100 -mp neulab/codebert-python 
# python -m model.code_search -exp CoNaLa_CodeBERT_CodeSearch -bs 100 --mode inference
# python -m model.code_search -exp CoNaLa_CodeBERT_CodeSearch_CosSim -bs 200 --mode train --cos_sim -e 5
# python -m model.code_search -exp CoNaLa_UniXcoder_CodeSearch_CosSim -bs 200 -mt unixcoder --mode train -e 50
# python -m model.code_search -exp CoNaLa_GraphCodeBERT_CodeSearch_CosSim -bs 200 -mt graphcodebert --mode train -e 50
# python -m model.code_search -exp CoNaLa_CSN_CodeBERT_ObfCodeSearch -bs 200 --mode train -obf --data_dir "./data/CoNaLa"
# python -m model.code_search -exp CoNaLa_CSN_CodeBERT_ObfCodeSearch2 -bs 500 --mode inference -obf --index_file_name codebert_obf_partial.index
# python -m model.code_search -exp CoNaLa_CSN_CodeBERT_ObfCodeSearch4_CosSim -bs 200 --mode train --cos_sim -obf -e 50
# python -m model.code_search -exp CoNaLa_CSN_CodeBERT_ObfCodeSearch4_CosSim -bs 500 --mode inference -obf --index_file_name codebert_obf_cos_sim.index --cos_sim
# python -m model.code_search -exp CoNaLa_CSN_CodeBERT_CodeSearch2.5_CosSim -bs 200 --mode train --cos_sim -e 50
# python -m model.code_search -exp CoNaLa_CSN_CodeBERT_CodeSearch2_Rev -bs 200 --mode train -e 50 --rev_ret
# python -m model.code_search -exp CoNaLa_CSN_CodeBERT_CodeSearch2_Sym -bs 200 --mode train -e 50 --sym_ret
# python -m model.code_search -exp CoNaLa_CSN_CodeBERT_CodeSearch3_CosSim -bs 200 --mode train --cos_sim -e 50
# python -m model.code_search -exp CoNaLa_CSN_CodeBERT_CodeSearch2_Sym -bs 500 --mode inference --index_file_name codebert_cos_sim.index --cos_sim
# python -m model.code_search -exp CoNaLa_CSN_CodeBERT_CodeSearch2_CosSim -bs 500 --mode inference --index_file_name codebert_cos_sim.index --cos_sim
# python -m model.code_search -exp CoNaLa_CSN_GraphCodeBERT_CodeSearch_CosSim -bs 500 -mt graphcodebert --mode inference --index_file_name graphcodebert_cos_sim.index --cos_sim
# python -m model.code_search -exp CoNaLa_CSN_UniXcoder_CodeSearch_CosSim -bs 500 -mt unixcoder --mode inference --index_file_name unixcoder_cos_sim.index --cos_sim
# python -m model.code_search -exp CoNaLa_CSN_CodeBERT_CodeSearch2_CosSim -bs 500 --mode inference --index_file_name codebert_plan_ops_cos_sim.index --cos_sim --index_modality text
# python -m model.code_search -exp CoNaLa_CSN_CodeBERT_CodeSearch2_Sym -bs 500 --mode inference --index_file_name codebert_plan_ops_cos_sim.index --cos_sim --index_modality text
# python -m model.code_search -exp CoNaLa_CSN_CodeBERT_CodeSim_CosSim -bs 200 --mode train --cos_sim -ccs -e 50
# python -m model.code_search -exp CoNaLa_CSN_CodeBERT_CodeSim2_CosSim -bs 200 --mode train --cos_sim -ccs -e 50