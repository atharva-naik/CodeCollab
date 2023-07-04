#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# dense retrievers
import os
import ast
import json
import faiss
import torch
import numpy as np
from typing import *
from tqdm import tqdm
import torch.nn as nn
from collections import defaultdict
from sentence_transformers import util
from torch.utils.data import DataLoader
from datautils.code_cell_analysis import obfuscate_code
from transformers import RobertaModel, RobertaTokenizerFast
from model.code_similarity_retrievers.datautils import CodeDataset
from model.code_search import CodeBERTSearchModel, GraphCodeBERTSearchModel, UniXcoderSearchModel, RobertaTokenizer, JuICeKBNNCodeBERTCodeSearchDataset, JuICeKBNNGraphCodeBERTCodeSearchDataset, JuICeKBNNUniXcoderCodeSearchDataset

tokenizer = RobertaTokenizerFast.from_pretrained("neulab/codebert-python")
LIST_OF_PUNCT_TOKENS = ["|","\\","/",",",")","(","!",";","`","|","{","}","'",'"',"[","]"]
def is_punct_token(token: str, thresh: float=0.5):
    global LIST_OF_PUNCT_TOKENS
    ctr = 0
    for c in token:
        if c in LIST_OF_PUNCT_TOKENS:
            ctr += 1
    if ctr/len(token) >= thresh: return True
    return False
LIST_OF_PUNCT_TOKEN = [k for k in tokenizer.vocab if is_punct_token(k)]
LIST_OF_PUNCT_TOKEN_IDS = [v for k,v in tokenizer.vocab.items() if is_punct_token(k)]
# extend roberta model to skip punctuation while mean pooling.
class NoPunctPoolRoberta(RobertaModel):
    def no_punct_pool(self, **batch):
        # emb_size is hard coded
        global LIST_OF_PUNCT_TOKEN_IDS
        punct_mask = torch.clone(batch["input_ids"]).cpu()
        punct_mask.apply_(lambda x: x in LIST_OF_PUNCT_TOKEN_IDS)
        punct_mask = punct_mask.unsqueeze(dim=-1).repeat(1,1,768)
        enc = self(**batch).last_hidden_state # batch_size x seq_len x emb_siE
        no_punct_pool_output = (punct_mask.to(enc.device) * enc).mean(dim=1)

        return no_punct_pool_output

class ZeroShotCodeBERTRetriever(nn.Module):
    """based on codebert-python, which is pre-trained further on python
    data to be used for computation of codebert score."""
    def __init__(self, model_path: str="neulab/codebert-python"):
        super().__init__()
        self.model = NoPunctPoolRoberta.from_pretrained(model_path)
        self.model.eval()
        self.tokenizer = RobertaTokenizerFast.from_pretrained(model_path)
        if torch.cuda.is_available(): self.model.cuda()

    def encode(self, codes: List[str], batch_size: int=32, 
               skip_punct: bool=True, show_progress_bar: bool=False):
        enc_dict = self.tokenizer.batch_encode_plus(
            codes, return_tensors="pt", 
            padding=True, truncation=True,
        )
        dataloader = DataLoader(
            CodeDataset(enc_dict),
            batch_size=batch_size, 
            shuffle=False,
        )
        embs = []
        for batch in tqdm(dataloader, disable=not(show_progress_bar)):
            # make sure the tensors are on the same device as the model.
            for k in batch: batch[k] = batch[k].to(self.model.device)
            with torch.no_grad():
                if skip_punct:
                    embs += self.model.no_punct_pool(**batch).detach().cpu().tolist()
                else: embs += self.model(**batch).pooler_output.detach().cpu().tolist()
        
        return torch.as_tensor(embs)

    def all_pairs_sim(self, c1: List[str], c2: List[str]):
        c1 = self.encode(c1)
        c2 = self.encode(c2)

        return util.cos_sim(c1, c2).cpu()

# dense retriever based on CodeBERT
class CodeBERTDenseSearcher:
    def __init__(self, path: str="microsoft/codebert-base", device: str="cuda:0",
                 ckpt_path: str="./experiments/CoNaLa_CSN_CodeBERT_ObfCodeSearch2/best_model.pt", 
                 tok_args: str={
                    "return_tensors": "pt", "padding": "max_length",
                    "truncation": True, "max_length": 100,
                }, model_type: str="codebert",
                faiss_index_path: str="./dense_indices/cos_sim.index"):
        if model_type == "codebert":
            tok_args = {
                "return_tensors": "pt",
                "padding": "max_length",
                "truncation": True,
                "max_length": 100,
            }
        elif model_type == "graphcodebert":
            tok_args = {
                "nl_length": 100, 
                "code_length": 100, 
                "data_flow_length": 64
            }
        elif model_type == "unixcoder":
            tok_args = {
                "return_tensors": "pt",
                "padding": "max_length",
                "truncation": True,
                "max_length": 100,
            }
        self.tok_args = tok_args
        self.model_type = model_type
        self.tokenizer = RobertaTokenizer.from_pretrained(path)
        if model_type == "codebert":
            self.model = CodeBERTSearchModel(path, device=device)
        elif model_type == "graphcodebert":
            self.model = GraphCodeBERTSearchModel(path, device=device)
        elif model_type == "unixcoder":
            self.model = UniXcoderSearchModel(path, device=device)
        self.device = device    
        self.faiss_index_path = faiss_index_path
        # ckpt_path = os.path.join(ckp, "best_model.pt")
        ckpt_dict = torch.load(ckpt_path, map_location="cpu")
        print(f"loaded model from: {ckpt_path}")
        self.model.load_state_dict(ckpt_dict["model_state_dict"])
        self.model.to(self.device)
        self.dense_index = faiss.read_index(faiss_index_path)

    def encode(self, queries: List[str], k: int=10, 
               text_query: bool=False, obf_code: bool=False,
               use_cos_sim: bool=True):
        # inp = self.tokenizer(query, **self.tok_args)
        assert not(text_query and obf_code), "code obfuscation not applicable in text query mode"
        if obf_code: queries = [obfuscate_code(q) for q in queries]
        if self.model_type == "codebert":
            dataset = JuICeKBNNCodeBERTCodeSearchDataset(
                tokenizer=self.tokenizer,
                queries=queries,
                **self.tok_args,
            )
        elif self.model_type == "graphcodebert":
            dataset = JuICeKBNNGraphCodeBERTCodeSearchDataset(
                tokenizer=self.tokenizer,
                queries=queries,
                **self.tok_args,
            )
        elif self.model_type == "unixcoder":
            dataset = JuICeKBNNUniXcoderCodeSearchDataset(
                tokenizer=self.tokenizer,
                queries=queries,
                **self.tok_args,
            )
        dataloader = dataset.get_docs_loader()
        q_mat = []
        for step, batch in tqdm(
            enumerate(dataloader),
            total=len(dataloader),
        ):
            self.model.zero_grad()
            with torch.no_grad():
                for j in range(len(batch)): batch[j] = batch[j].to(self.device)
                dtype = "text" if text_query else "code"
                q_enc = self.model.encode(*batch, dtype=dtype).cpu().detach().tolist()
                q_mat += q_enc
        q_mat = torch.as_tensor(q_mat).numpy()
        # normalize by L2 norm if cosine similarity is being used.
        if use_cos_sim: faiss.normalize_L2(q_mat)
        # print("constructed query matrix")
        return q_mat

    def search(self, queries: List[str], k: int=10, 
               text_query: bool=False, obf_code: bool=False,
               use_cos_sim: bool=True):
        # inp = self.tokenizer(query, **self.tok_args)
        assert not(text_query and obf_code), "code obfuscation not applicable in text query mode"
        if obf_code: queries = [obfuscate_code(q) for q in queries]
        if self.model_type == "codebert":
            dataset = JuICeKBNNCodeBERTCodeSearchDataset(
                tokenizer=self.tokenizer,
                queries=queries,
                **self.tok_args,
            )
        elif self.model_type == "graphcodebert":
            dataset = JuICeKBNNGraphCodeBERTCodeSearchDataset(
                tokenizer=self.tokenizer,
                queries=queries,
                **self.tok_args,
            )
        elif self.model_type == "unixcoder":
            dataset = JuICeKBNNUniXcoderCodeSearchDataset(
                tokenizer=self.tokenizer,
                queries=queries,
                **self.tok_args,
            )
        # print(f"dataset: {len(dataset.docs)}")
        dataloader = dataset.get_docs_loader()
        # print(f"dataloader: {len(dataloader)}")
        q_mat = []
        for step, batch in tqdm(
            enumerate(dataloader),
            total=len(dataloader),
        ):
            self.model.zero_grad()
            with torch.no_grad():
                for j in range(len(batch)): batch[j] = batch[j].to(self.device) # print(batch)
                dtype = "text" if text_query else "code"
                # print(len(batch), dtype)
                q_enc = self.model.encode(*batch, dtype=dtype).cpu().detach().tolist()
                q_mat += q_enc
        q_mat = torch.as_tensor(q_mat).numpy()
        # normalize by L2 norm if cosine similarity is being used.
        if use_cos_sim: faiss.normalize_L2(q_mat)
        print("constructed query matrix")

        return self.dense_index.search(q_mat, k=k)

class EnsembleDenseCodeSearcher:
    def __init__(self, path: str="microsoft/codebert-base", device: str="cuda:0",
                 ckpt_paths: str=[
                    "./experiments/CoNaLa_CSN_CodeBERT_CodeSearch2_CosSim/best_model.pt",
                    "./experiments/CoNaLa_CSN_CodeBERT_ObfCodeSearch4_CosSim/best_model.pt",
                 ], 
                 tok_args: str={
                    "return_tensors": "pt", "padding": "max_length",
                    "truncation": True, "max_length": 100,
                }, model_type: str="codebert", # requires_obf: List[bool]=[True, False],
                faiss_index_paths: List[str]=[
                    "./dense_indices/CoNaLa_CSN_CodeBERT_CodeSearch2_CosSim/cos_sim.index",
                    "./dense_indices/codebert_obf_cos_sim.index",
                ]):
        # NOTE: SANITY CHECKS 
        # assert to ensure there are equal number of checkpoints and saved dense indices:
        assert len(faiss_index_paths) == len(ckpt_paths)
        # # there also should be equal number of boolean flags (whether to obfuscated code or not) as there are faiss index paths.
        # assert len(requires_obf) == len(ckpt_paths)

        if model_type == "codebert":
            tok_args = {
                "return_tensors": "pt",
                "padding": "max_length",
                "truncation": True,
                "max_length": 100,
            }
        elif model_type == "graphcodebert":
            tok_args = {
                "nl_length": 100, 
                "code_length": 100, 
                "data_flow_length": 64
            }
        elif model_type == "unixcoder":
            tok_args = {
                "return_tensors": "pt",
                "padding": "max_length",
                "truncation": True,
                "max_length": 100,
            }
        self.tok_args = tok_args
        self.model_type = model_type
        self.tokenizer = RobertaTokenizer.from_pretrained(path)
        if model_type == "codebert":
            self.models = [CodeBERTSearchModel(path, device=device) for _ in range(len(ckpt_paths))]
        elif model_type == "graphcodebert":
            self.models = [GraphCodeBERTSearchModel(path, device=device) for _ in range(len(ckpt_paths))]
        elif model_type == "unixcoder":
            self.models = [UniXcoderSearchModel(path, device=device) for _ in range(len(ckpt_paths))]
        self.device = device    
        self.faiss_index_path = faiss_index_paths

        # load up the chechkpoints:
        for i, ckpt_path in enumerate(ckpt_paths):
            ckpt_dict = torch.load(ckpt_path, map_location="cpu")
            print(f"loaded model from: {ckpt_path}")
            self.models[i].load_state_dict(ckpt_dict["model_state_dict"])
            self.models[i].to(self.device)

        self.dense_indices = []
        for faiss_index_path in faiss_index_paths:
            self.dense_indices.append(faiss.read_index(faiss_index_path))

    def encode(self, queries: List[str], use_cos_sim: bool=True):
        """this function assumes only text queries will be asked."""
        if self.model_type == "codebert":
            dataset = JuICeKBNNCodeBERTCodeSearchDataset(
                tokenizer=self.tokenizer,
                queries=queries,
                **self.tok_args,
            )
        elif self.model_type == "graphcodebert":
            dataset = JuICeKBNNGraphCodeBERTCodeSearchDataset(
                tokenizer=self.tokenizer,
                queries=queries,
                **self.tok_args,
            )
        elif self.model_type == "unixcoder":
            dataset = JuICeKBNNUniXcoderCodeSearchDataset(
                tokenizer=self.tokenizer,
                queries=queries,
                **self.tok_args,
            )
        dataloader = dataset.get_docs_loader()
        ensemble_enc = []
        for model in self.models:
            q_mat = []
            for step, batch in tqdm(
                enumerate(dataloader),
                total=len(dataloader),
            ):
                self.model.zero_grad()
                with torch.no_grad():
                    for j in range(len(batch)): batch[j] = batch[j].to(self.device)
                    q_enc = model.encode(*batch, dtype="text").cpu().detach().tolist()
                    q_mat += q_enc
            q_mat = torch.as_tensor(q_mat).numpy()
            # normalize by L2 norm if cosine similarity is being used.
            if use_cos_sim: faiss.normalize_L2(q_mat)
            ensemble_enc.append(q_mat)
        # num_models x data_size x emb_size
        return ensemble_enc # list of encodings from each model of the ensemble

    def search(self, queries: List[str], k: int=10, use_cos_sim: bool=True):
        """this function assumes only text queries will be asked"""
        if self.model_type == "codebert":
            dataset = JuICeKBNNCodeBERTCodeSearchDataset(
                tokenizer=self.tokenizer,
                queries=queries,
                **self.tok_args,
            )
        elif self.model_type == "graphcodebert":
            dataset = JuICeKBNNGraphCodeBERTCodeSearchDataset(
                tokenizer=self.tokenizer,
                queries=queries,
                **self.tok_args,
            )
        elif self.model_type == "unixcoder":
            dataset = JuICeKBNNUniXcoderCodeSearchDataset(
                tokenizer=self.tokenizer,
                queries=queries,
                **self.tok_args,
            )
        dataloader = dataset.get_docs_loader()
        ensemble_scores = [defaultdict(lambda: 0) for _ in queries]
        for i, model in enumerate(self.models):
            q_mat = []
            for step, batch in tqdm(
                enumerate(dataloader),
                total=len(dataloader),
            ):
                model.zero_grad()
                with torch.no_grad():
                    for j in range(len(batch)): batch[j] = batch[j].to(self.device)
                    q_enc = model.encode(*batch, dtype="text").cpu().detach().tolist()
                    q_mat += q_enc
            q_mat = torch.as_tensor(q_mat).numpy()
            # normalize by L2 norm if cosine similarity is being used.
            if use_cos_sim: faiss.normalize_L2(q_mat)
            print(f"doing search for model-{i+1}")
            scores, indices = self.dense_indices[i].search(q_mat, k=k)
            # iterate over data/instances and aggregate scores per index
            for i in range(len(indices)):
                for ind, score in zip(indices[i], scores[i]):
                    ensemble_scores[i][ind] += score
        # get the top-k index predictions and scores from the aggregated scores.
        ensemble_inds = np.array(
            [
                [
                    k for k,v in sorted(
                        ind_scores.items(), 
                        key=lambda x: x[1],
                        reverse=True
                )
            ][:k] for ind_scores in ensemble_scores]
        )
        ensemble_scores = np.array(
            [
                [
                    v for k,v in sorted(
                        ind_scores.items(), 
                        key=lambda x: x[1],
                        reverse=True
                    )
                ][:k] for ind_scores in ensemble_scores
            ]
        )

        return ensemble_scores, ensemble_inds