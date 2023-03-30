# script to locate most similar code snippets from tutorials for JuICe sampled NBs.
import os
import json
import torch
# from tqdm import tqdm
from collections import defaultdict
from sentence_transformers import util
from model.code_similarity import ZeroShotCodeBERTRetriever
from datautils.markdown_cell_analysis import extract_notebook_hierarchy

FILT_LIST_STEPS = ['Effective Pandas','Articles','Introductory','Fast Pandas','PyTorch Recipes','NumPy Features','scipy','NumPy Applications','Data structures accepted by seaborn','User Notes','matplotlib','Tutorials','numpy','User guide and tutorial','torch','pandas_toms_blog','seaborn','sklearn','statsmodels', 'Examples']

def retrieve_tutorial_codes_for_target_cells(context_size: int=2):
    """
    Params:
    - context_size: number of code cells (including target one) to be included for matching (if present)
    """
    dense_retriever = ZeroShotCodeBERTRetriever()
    # encode the tutorial code snippets.
    tutorial_to_codes = json.load(open("./scrape_tutorials/unified_filt_code_to_path_KG.json"))
    tut_emb = dense_retriever.encode(
        list(tutorial_to_codes.keys()),
        show_progress_bar=True,
        batch_size=64,
    )
    # encode the sampled NB codes (target cells only).
    sampled_nbs = json.load(open("./data/juice-dataset/sampled_juice_train.json"))
    sampled_nb_code = {}
    for k, nb in sampled_nb_code:
        sampled_nb_code[k] = []
        ctr = context_size-1
        for cell in nb['context']:
            if cell["cell_type"] == "code":
                sampled_nb_code[k].append(cell["code"])
                ctr -= 1
            if ctr == 0: break
        sampled_nb_code[k].append(nb['code'])
        sampled_nb_code = "\n".join(sampled_nb_code)
    {k: nb['code'] for k, nb in sampled_nbs.items()}
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
    with open(f"./scrape_tutorials/tut_codes_matched_with_sampled_NBs_ctx_size_{context_size}.json", "w") as f:
        json.dump(matched_codes_and_paths, f, indent=4)

def aggregate_path_weights(path: str="./scrape_tutorials/tut_codes_matched_with_sampled_NBs.json"):
    ret_tut_paths = json.load(open(path))
    step_weights = defaultdict(lambda:0)
    path_weights = defaultdict(lambda:0)
    num_codes_per_path = defaultdict(lambda:0)
    code_to_path_KG = json.load(open("./scrape_tutorials/unified_filt_code_to_path_KG.json"))
    for path_list in code_to_path_KG.values():
        for path in path_list: num_codes_per_path[path] += 1
    for k, v in ret_tut_paths.items():
        for path_list in v["matched_tutorial_paths"]:
            for path in path_list:
                path_weights[path] += 1
                for step in path.split("->"):
                    step = step.strip()
                    step_weights[step] += 1
    path_weights = {k: v for k,v in sorted(path_weights.items(), key=lambda x: x[1])}
    step_weights = {k: v for k,v in sorted(step_weights.items(), key=lambda x: x[1])}
    num_codes_per_path = {k: v for k,v in sorted(num_codes_per_path.items(), key=lambda x: x[1])}

    return path_weights, step_weights, num_codes_per_path

def compare_tut_paths_with_GT_paths(tuts_path: str="./scrape_tutorials/tut_codes_matched_with_sampled_NBs.json"):
    sampled_nb_matched_paths = json.load(open(tuts_path))
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