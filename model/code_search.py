#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# code for training/fine-tuning CodePLMs for code search using bi-encoder setups.
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel, RobertaTokenizer

# bi-encoder for code search.
class CodeBERTSearchModel(nn.Module):
    """
    This class implements the classic bi-encoder architecture
    for code search using two CodePLM instances.
    """
    def __init__(self, path: str="microsoft/codebert-base", device: str="cuda"):
        super(CodeBERTSearchModel, self).__init__()
        self.code_encoder = RobertaModel.from_pretrained(path)
        self.query_encoder = RobertaModel.from_pretrained(path)
        self.ce_loss = nn.CrossEntropyLoss()
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"moving model to device: {self.device}")
        self.to(device)

    def forward(self, query_iids, query_attn, code_iids, code_attn):
        query_enc = self.query_encoder(query_iids, query_attn).pooler_output
        code_enc = self.code_encoder(code_iids, code_attn).pooler_output
        batch_size, _ = query_enc.shape
        target = torch.eye(batch_size).to(self.device)
        loss = self.ce_loss(query_enc @ code_enc.T, target)

        return query_enc, code_enc, loss