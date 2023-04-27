#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# dense retrievers
import os
import ast
import json
import torch
import numpy as np
from typing import *
from tqdm import tqdm
import torch.nn as nn
from sentence_transformers import util
from torch.utils.data import DataLoader
from transformers import RobertaModel, RobertaTokenizerFast
from model.code_similarity_retrievers.datautils import CodeDataset

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