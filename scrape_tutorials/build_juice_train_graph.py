# build KB from all of JuICe's NBs:
import os
import json
from typing import *
from tqdm import tqdm
from collections import defaultdict
from datautils import read_jsonl, camel_case_split
from scrape_tutorials.processing import KGPathsIndex
from datautils.markdown_cell_analysis import extract_notebook_hierarchy_from_seq

def get_name_from_path(path: str):
    name_terms = []
    for chunk in path.split("/")[-1].split(".")[0].strip().split("_"):
        for term in camel_case_split(chunk):
            name_terms.append(term.lower().strip())
    
    return " ".join(name_terms).lower().title()

def prune_empty_nodes(KG: dict):
    pruned_KG = {}
    for path_key, cells in KG.items():
        pruned_KG[path_key] = []
        for cell in cells:
            if cell[0].strip() != "":
                pruned_KG[path_key].append(cell)
    
    return pruned_KG

def get_uniq_steps(kg_paths: List[str]) -> Set[str]:
    uniq_steps = set()
    for path in kg_paths:
        for step in path.split("->"):
            step = step.strip()
            uniq_steps.add(step)

    return uniq_steps

def get_path_depths(kg_paths: List[str]) -> List[int]:
    path_depths = []
    for path in kg_paths: path_depths.append(len(path.split("->")))
    return path_depths
# # KB to path index converter class.
# class KBToPathIndexConverter:
#     def __init__(self, KB: dict):
#         self.KB = KB
#         self.path_index = self.build(KB, {})

#     def build(self, sub_KG: Union[list, dict, tuple], 
#               sub_path_index: Dict[str, List[Tuple[str, str]]]) -> Dict[str, List[Tuple[str, str]]]:
#         # base case:
#         if isinstance(sub_KG, tuple):
#             pass
#         elif isinstance(sub_KG, dict):
#             for k,v in sub_KG.items():
                 
#         path_index
path = "./data/juice-dataset/traindedup.jsonl"
train_data = read_jsonl(path)
KB = defaultdict(lambda:[])
for inst in tqdm(train_data):
    seq = inst["context"][::-1] + [{"code": inst["code"], "cell_type": "code"}]
    nb_json = extract_notebook_hierarchy_from_seq(seq)[0].serialize2()[""]
    name = get_name_from_path(inst["metadata"]["path"]) 
    KB[name].append(nb_json)
paths_KB = KGPathsIndex.from_data(KB)
task_decomp_keyed_dict = paths_KB.save("./JuICe_train_KB.json")