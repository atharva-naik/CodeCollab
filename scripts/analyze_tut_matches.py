# script to locate most similar code snippets from tutorials for JuICe sampled NBs.
import os
import json
import torch
# from tqdm import tqdm
from sentence_transformers import util
from model.code_similarity import ZeroShotCodeBERTRetriever
from datautils.markdown_cell_analysis import extract_notebook_hierarchy

def retrieve_tutorial_codes_for_target_cells():
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

def compare_tut_paths_with_GT_paths():
    sampled_nb_matched_paths = json.load(open("./scrape_tutorials/tut_codes_matched_with_sampled_NBs.json"))
    sampled_nbs = json.load(open("./data/juice-dataset/sampled_juice_train.json"))
    fetched_tut_paths_and_GTs = []
    for id, nb in sampled_nbs.items():
        root, _ = extract_notebook_hierarchy(nb)
        triples = root.get_root_to_leaf_paths()[-1] # get root to leaf path for the target cell.
        plan_seq = [triple.content for triple in triples[1:-1] if triple.cell_type == 'markdown']+[triples[-1].content]
        GT_plan_seq = "->".join(plan_seq)
        pred_plan_seqs = []
        for code, seq_of_paths in zip(
            sampled_nb_matched_paths[id]["matched_tutorial_codes"], 
            sampled_nb_matched_paths[id]["matched_tutorial_paths"],
        ):
            pred_plan_seqs += [path+"->"+code for path in seq_of_paths]
        # print(GT_plan_seq)
        # print(pred_plan_seqs)
        fetched_tut_paths_and_GTs.append({
            "id": id, "GT": GT_plan_seq,
            "pred": pred_plan_seqs,
        })

    return fetched_tut_paths_and_GTs