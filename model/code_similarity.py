#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
from typing import *
from tqdm import tqdm
import torch.nn as nn
from sentence_transformers import util
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaModel, RobertaTokenizerFast
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

class ZeroShotCodeBERTRetriever(nn.Module):
    """based on codebert-python, which is pre-trained further on python
    data to be used for computation of codebert score."""
    def __init__(self, model_path: str="neulab/codebert-python"):
        super().__init__()
        self.model = RobertaModel.from_pretrained(model_path)
        self.model.eval()
        self.tokenizer = RobertaTokenizerFast.from_pretrained(model_path)
        if torch.cuda.is_available(): self.model.cuda()

    def encode(self, codes: List[str], batch_size: int=32, show_progress_bar: bool=False):
        enc_dict = self.tokenizer.batch_encode_plus(
            codes, return_tensors="pt", 
            padding=True, truncation=True,
        )
        dataloader = DataLoader(CodeDataset(enc_dict), batch_size, shuffle=False)
        embs = []
        for batch in dataloader:
            # print(batch)
            # print(batch.keys())
            # make sure the tensors are on the same device as the model.
            for k in batch: batch[k] = batch[k].to(self.model.device)
            with torch.no_grad():
                embs += self.model(**batch).pooler_output.detach().cpu().tolist()
        
        return torch.as_tensor(embs)

    def all_pairs_sim(self, c1: List[str], c2: List[str]):
        c1 = self.encode(c1)
        c2 = self.encode(c2)

        return util.cos_sim(c1, c2).cpu()