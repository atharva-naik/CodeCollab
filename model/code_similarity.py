#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import ast
import json
import torch
import numpy as np
from typing import *
from tqdm import tqdm
import torch.nn as nn
from scipy import sparse
from datautils import camel_case_split
from sentence_transformers import util
from torch.utils.data import Dataset, DataLoader
from datautils.code_cell_analysis import process_nb_cell
from transformers import RobertaModel, RobertaTokenizerFast
from sklearn.feature_extraction.text import TfidfVectorizer
# code-to-code similarity computation.

class CodeDataset(Dataset):
    def __init__(self, data_dict: Dict[str, torch.Tensor]):
        super().__init__()
        self.data = data_dict
        self.keys = list(data_dict.keys())

    def __len__(self):
        return len(self.data[self.keys[0]])

    def __getitem__(self, i):
        out = {}
        for key in self.data:
            out[key] = self.data[key][i]
        
        return out

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
# LIST_OF_PUNCT_TOKENS = ["(",")",",",'"',"'","\\","/",";","[","]","|",'("', '")', "('", "')", "['", "']", '["', '"]']
# LIST_OF_PUNCT_TOKEN_IDS = [tokenizer.vocab[token] for token in LIST_OF_PUNCT_TOKENS] # [1640, 43, 6, 113, 108, 37457, 73, 131, 10975, 742, 15483


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

# a sparse feature based retriever for matching up.
class SparseRetriever:
    def __init__(self):
        self.inverted_index = {}

    def build_inverted_index(self):
        pass

class BM25SparseRetriever(object):
    def __init__(self, b: float=0.75, k1: float=1.6, 
                code2words_path: str="./JuICe_train_code2words.json"):
        self.vectorizer = TfidfVectorizer(norm=None, smooth_idf=False)
        self.b = b
        self.k1 = k1
        self.code2words_path = code2words_path
        if os.path.exists(code2words_path):
            self.code2words = {k: v if v is not None else "" for k,v in json.load(open(code2words_path)).items()}

    def save_code2word(self, all_codes: List[str]):
        code2words = {}
        for code in tqdm(all_codes):
            text = self.transform_code_to_text(code)
            code2words[code] = text
        self.code2words = code2words
        with open(self.code2words_path, "w") as f:
            json.dump(self.code2words, f, indent=4)

    def transform_code_to_text(self, code: str):
        """convert a piece of code to a stream of variable names and API calls."""
        code = process_nb_cell(code)
        try:
            all_terms = []
            for node in ast.walk(ast.parse(code)):
                if isinstance(node, ast.Name):
                    name = ast.unparse(node)
                elif isinstance(node, ast.Call):
                    name = ast.unparse(node.func)
                elif isinstance(node, ast.alias):
                    if node.asname is not None:
                        name = node.name + " " + node.asname
                    else: name = node.name
                else: continue
                for dot_split_term in name.split("."): # split by dots first.        
                    for underscore_split_term in dot_split_term.split("_"): # split by underscore second.
                        for term in camel_case_split(underscore_split_term): # split by camel case finally.
                            all_terms.append(term.lower())

            return " ".join(all_terms)
        except SyntaxError as e: return ""
            # print(e, code)
    def fit(self, X: List[str], verbose: bool=False):
        """ Fit IDF to documents X 
        X: is a list of codes."""
        X = [self.code2words[x] for x in tqdm(X)]
        self.vectorizer.fit(X)
        y = super(TfidfVectorizer, self.vectorizer).transform(X)
        self.avdl = y.sum(1).mean()

    def transform(self, q, X):
        """ Calculate BM25 between query q and documents X """
        b, k1, avdl = self.b, self.k1, self.avdl
        try: q = self.code2words[q]
        except KeyError:
            new_q = self.transform_code_to_text(q)
            if new_q == "": new_q = q
            else: q = new_q
        X = [self.code2words[x] for x in tqdm(X)]
        # apply CountVectorizer
        X = super(TfidfVectorizer, self.vectorizer).transform(X)
        len_X = X.sum(1).A1
        q, = super(TfidfVectorizer, self.vectorizer).transform([q])
        assert sparse.isspmatrix_csr(q)

        # convert to csc for better column slicing
        X = X.tocsc()[:, q.indices]
        denom = X + (k1 * (1 - b + b * len_X / avdl))[:, None]
        # idf(t) = log [ n / df(t) ] + 1 in sklearn, so it need to be coneverted
        # to idf(t) = log [ n / df(t) ] with minus 1
        idf = self.vectorizer._tfidf.idf_[None, q.indices] - 1.
        numer = X.multiply(np.broadcast_to(idf, X.shape)) * (k1 + 1)                                                          
        scores = (numer / denom).sum(1).A1
        results = []
        for i in scores.argsort()[::-1]: results.append(i)

        return results