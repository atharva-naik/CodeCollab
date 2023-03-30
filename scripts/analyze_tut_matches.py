# script to locate most similar code snippets from tutorials for JuICe sampled NBs.
import os
import json
import torch
from tqdm import tqdm
from sentence_transformers import util
from model.code_similarity import ZeroShotCodeBERTRetriever

dense_retriever = ZeroShotCodeBERTRetriever()
# encode the tutorial code snippets.
tutorial_to_codes = json.load(open("./scrape_tutorials/unified_code_to_path_KG.json"))
tut_emb = dense_retriever.encode(
    list(tutorial_to_codes.keys()),
    show_progress_bar=True,
    batch_size=64,
)
# encode the sampled NB codes (target cells only).
sampled_nbs = json.load(open("./data/juice-dataset/sampled_juice_train.json"))
sampled_nb_code = {k: nb['code'] for k, nb in sampled_nbs.items()}
sampled_nb_embs = dense_retriever.encode(
    list(sampled_nb_code.values()),
    show_progress_bar=True,
    batch_size=64,
)
# compute match scores:
k = 5
match_scores = util.cos_sim(sampled_nb_embs, tut_emb)
tut_paths = list(tutorial_to_codes.values())
tut_codes = list(tutorial_to_codes.keys())
matched_codes_and_paths = {}
nb_keys = list(sampled_nb_code.keys())
nb_codes = list(sampled_nb_code.values())
for i, ind_list in enumerate(torch.topk(match_scores, k=k).indices):
    key = nb_keys[i]
    matched_codes_and_paths[key] = {
        "jupyter_code_cell": nb_codes[i],
        "matched_tutorial_code_inds": ind_list.tolist(),
        "matched_tutorial_codes": [tut_codes[j] for j in ind_list],
        "matched_tutorial_paths": [tut_paths[j] for j in ind_list],
    }
with open("./scrape_tutorials/tut_codes_matched_with_sampled_NBs.json", "w") as f:
    json.dump(matched_codes_and_paths, f, indent=4)