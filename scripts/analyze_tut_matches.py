# script to locate most similar code snippets from tutorials for JuICe sampled NBs.
import os
import json
from tqdm import tqdm
from model.code_similarity import ZeroShotCodeBERTRetriever

dense_retriever = ZeroShotCodeBERTRetriever()
tutorial_to_codes = json.load(open("./scrape_tutorials/unified_code_to_path_KG.json"))
tut_emb = dense_retriever.encode(list(tutorial_to_codes.keys()))
sampled_nbs = json.load(open("./data/juice-dataset/sampled_juice_train.json"))
for nb in tqdm(sampled_nbs.values()):
    code = nb['code']
    tutorial_to_codes.
    dense_retriever.encode()