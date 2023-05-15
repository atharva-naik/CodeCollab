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
from sentence_transformers import util
from torch.utils.data import DataLoader
from datautils.code_cell_analysis import obfuscate_code
from transformers import RobertaModel, RobertaTokenizerFast
from model.code_similarity_retrievers.datautils import CodeDataset
from model.code_search import CodeBERTSearchModel, RobertaTokenizer, JuICeKBNNCodeBERTCodeSearchDataset

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
                 tok_args: str = {
                    "return_tensors": "pt", "padding": "max_length",
                    "truncation": True, "max_length": 100,
                }, faiss_index_path: str="./dense_indices/codebert_partial.index"):
        self.tok_args = tok_args
        self.tokenizer = RobertaTokenizer.from_pretrained(path)
        self.model = CodeBERTSearchModel(
            path, device=device,
        )
        self.device = device    
        self.faiss_index_path = faiss_index_path
        # ckpt_path = os.path.join(ckp, "best_model.pt")
        ckpt_dict = torch.load(ckpt_path, map_location="cpu")
        print(f"loaded model from: {ckpt_path}")
        self.model.load_state_dict(ckpt_dict["model_state_dict"])
        self.model.to(self.device)
        self.dense_index = faiss.read_index(faiss_index_path)

    def search(self, queries: List[str], k: int=10, 
               text_query: bool=False, obf_code: bool=False,
               use_cos_sim: bool=True):
        # inp = self.tokenizer(query, **self.tok_args)
        assert not(text_query and obf_code), "code obfuscation not applicable in text query mode"
        if obf_code: queries = [obfuscate_code(q) for q in queries]
        dataset = JuICeKBNNCodeBERTCodeSearchDataset(
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
                for j in range(len(batch)): batch[j] = batch[j].to(self.device) # print(batch)
                dtype = "text" if text_query else "code"
                q_enc = self.model.encode(*batch, dtype=dtype).cpu().detach().tolist()
                q_mat += q_enc
        q_mat = torch.as_tensor(q_mat).numpy()
        # normalize by L2 norm if cosine similarity is being used.
        if use_cos_sim: faiss.normalize_L2(q_mat)
        print("constructed query matrix")

        return self.dense_index.search(q_mat, k=k)