# script to locate most similar code snippets from tutorials for JuICe sampled NBs.
import os
import re
import json
import torch
import tokenize
from io import StringIO
# from tqdm import tqdm
from collections import defaultdict
from sentence_transformers import util
from model.code_similarity import ZeroShotCodeBERTRetriever
from datautils.markdown_cell_analysis import extract_notebook_hierarchy

FILT_LIST_STEPS = ['Effective Pandas','Articles','Introductory','Fast Pandas','PyTorch Recipes','NumPy Features','scipy','NumPy Applications','Data structures accepted by seaborn','User Notes','matplotlib','Tutorials','numpy','User guide and tutorial','torch','pandas_toms_blog','seaborn','sklearn','statsmodels', 'Examples','','Intermediate','Examples based on real world datasets']

def remove_comments_and_docstrings(source, lang='python'):
    if lang in ['python']:
        """
        Returns 'source' minus comments and docstrings.
        """
        io_obj = StringIO(source)
        out = ""
        prev_toktype = tokenize.INDENT
        last_lineno = -1
        last_col = 0
        for tok in tokenize.generate_tokens(io_obj.readline):
            token_type = tok[0]
            token_string = tok[1]
            start_line, start_col = tok[2]
            end_line, end_col = tok[3]
            ltext = tok[4]
            if start_line > last_lineno:
                last_col = 0
            if start_col > last_col:
                out += (" " * (start_col - last_col))
            # Remove comments:
            if token_type == tokenize.COMMENT:
                pass
            # This series of conditionals removes docstrings:
            elif token_type == tokenize.STRING:
                if prev_toktype != tokenize.INDENT:
            # This is likely a docstring; double-check we're not inside an operator:
                    if prev_toktype != tokenize.NEWLINE:
                        if start_col > 0:
                            out += token_string
            else:
                out += token_string
            prev_toktype = token_type
            last_col = end_col
            last_lineno = end_line
        temp=[]
        for x in out.split('\n'):
            if x.strip()!="":
                temp.append(x)
        return '\n'.join(temp)
    elif lang in ['ruby']:
        return source
    else:
        def replacer(match):
            s = match.group(0)
            if s.startswith('/'):
                return " " # note: a space and not an empty string
            else:
                return s
        pattern = re.compile(
            r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
            re.DOTALL | re.MULTILINE
        )
        temp=[]
        for x in re.sub(pattern, replacer, source).split('\n'):
            if x.strip()!="":
                temp.append(x)
        return '\n'.join(temp)

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
    for k, nb in sampled_nbs.items():
        sampled_nb_code[k] = []
        ctr = context_size-1
        for cell in nb['context']:
            if cell["cell_type"] == "code":
                sampled_nb_code[k].append(cell["code"])
                ctr -= 1
            if ctr == 0: break
        sampled_nb_code[k].append(nb['code'])
        sampled_nb_code[k] = remove_comments_and_docstrings("\n".join(sampled_nb_code[k]))
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

def aggregate_path_weights(path: str="./scrape_tutorials/tut_codes_matched_with_sampled_NBs_ctx_size_2.json"):
    ret_tut_paths = json.load(open(path))
    step_weights = defaultdict(lambda:0)
    path_weights = defaultdict(lambda:0)
    hier_step_weights = defaultdict(lambda:defaultdict(lambda:0))
    num_codes_per_path = defaultdict(lambda:0)
    code_to_path_KG = json.load(open("./scrape_tutorials/unified_filt_code_to_path_KG.json"))
    for path_list in code_to_path_KG.values():
        for path in path_list: num_codes_per_path[path] += 1
    for k, v in ret_tut_paths.items():
        for path_list in v["matched_tutorial_paths"]:
            for path in path_list:
                path_weights[path] += 1
                for depth, step in enumerate(path.split("->")):
                    step = step.strip()
                    step_weights[step] += 1
                    hier_step_weights[depth][step] += 1
    path_weights = {k: v for k,v in sorted(path_weights.items(), key=lambda x: x[1])}
    step_weights = {k: v for k,v in sorted(step_weights.items(), key=lambda x: x[1])}
    num_codes_per_path = {k: v for k,v in sorted(num_codes_per_path.items(), key=lambda x: x[1])}
    hier_step_weights = {depth: {k: v for k,v in sorted(hier_step_weights[depth].items(), key=lambda x: x[1]) if k not in FILT_LIST_STEPS} for depth in hier_step_weights}

    return path_weights, step_weights, hier_step_weights, num_codes_per_path

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
# java -DoutputDir=<output dir to store JSON representation of graph> -DquadFile=<file name to write quads to - this file gets appended to, so all analyzed scripts end up in a single file> -cp codebreaker3.jar util.RunTurtleSingleAnalysis <python script to run on> <graph prefix> <graph qualifier> 

# java -DoutputDir="/home/atharva/CMU/sem2/Directed Study/CodeCollab/codegraphs" -cp codebreaker3.jar util.RunTurtleSingleAnalysis "/home/atharva/CMU/sem2/Directed Study/CodeCollab/scripts/infill_content_analysis.py" null null