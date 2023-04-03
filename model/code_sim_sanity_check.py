import json
import torch
import random
import numpy as np
from typing import *
from sentence_transformers import util
from model.code_similarity import ZeroShotCodeBERTRetriever

random.seed(2023)
np.random.seed(2023)
torch.manual_seed(2023)

dense_retriever = ZeroShotCodeBERTRetriever()
codes_to_tutorials = json.load(open("./scrape_tutorials/unified_filt_code_to_path_KG.json"))
codes: List[str] = list(codes_to_tutorials.keys())

print("using punctuations:")
SKIP_PUNCT = False
tut_emb = dense_retriever.encode(codes, show_progress_bar=True, skip_punct=SKIP_PUNCT)
code = """bx = b.plot(kind='orange',figsize=(10,6))
bx.set_ylabel("test message"""
q_emb = dense_retriever.encode([code], skip_punct=SKIP_PUNCT)
cos_scores = util.cos_sim(q_emb, tut_emb)[0]
print("ref:\n")
print(code)
indices = torch.topk(cos_scores, k=5).indices.cpu().tolist()
scores = torch.topk(cos_scores, k=5).values.cpu().tolist()
# print(indices)
print("ret:\n")
for i, score, ind in zip(range(5), scores, indices):
    
    print(i, "|", ind, "|", score)
    print(codes[ind])
print("-"*10)

code = """bx = b.plot(kind='orange',figsize=(10,6))
bx.set_ylabel("test message")"""
q_emb = dense_retriever.encode([code], skip_punct=SKIP_PUNCT)
cos_scores = util.cos_sim(q_emb, tut_emb)[0]
print("ref:\n")
print(code)
indices = torch.topk(cos_scores, k=5).indices.cpu().tolist()
scores = torch.topk(cos_scores, k=5).values.cpu().tolist()
print("ret:\n")
for i, score, ind in zip(range(5), scores, indices):
    print(i, "|", ind, "|", score)
    print(codes[ind])
print("-"*10)

print("skipping punctuations:")
SKIP_PUNCT = True
tut_emb = dense_retriever.encode(codes, show_progress_bar=True, skip_punct=SKIP_PUNCT)
code = """bx = b.plot(kind='orange',figsize=(10,6))
bx.set_ylabel("test message"""
q_emb = dense_retriever.encode([code], skip_punct=SKIP_PUNCT)
cos_scores = util.cos_sim(q_emb, tut_emb)[0]
print("ref:\n")
print(code)
indices = torch.topk(cos_scores, k=5).indices.cpu().tolist()
scores = torch.topk(cos_scores, k=5).values.cpu().tolist()
# print(indices)
print("ret:\n")
for i, score, ind in zip(range(5), scores, indices):
    
    print(i, "|", ind, "|", score)
    print(codes[ind])
print("-"*10)

code = """bx = b.plot(kind='orange',figsize=(10,6))
bx.set_ylabel("test message")"""
q_emb = dense_retriever.encode([code], skip_punct=SKIP_PUNCT)
cos_scores = util.cos_sim(q_emb, tut_emb)[0]
print("ref:\n")
print(code)
indices = torch.topk(cos_scores, k=5).indices.cpu().tolist()
scores = torch.topk(cos_scores, k=5).values.cpu().tolist()
print("ret:\n")
for i, score, ind in zip(range(5), scores, indices):
    print(i, "|", ind, "|", score)
    print(codes[ind])
print("-"*10)