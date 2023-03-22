#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
from typing import *
from tqdm import tqdm
from collections import defaultdict
from scrape_tutorials.parsers import process_text, process_cell_text
# def process_text(text: str):
#     """remove residual html tags, &amp; etc. 
#     e.g.: `Reshaping &amp; Tidy Data<blockquote>` to `Reshaping & Tidy Data`"""
#     return text.replace("&amp;", "&").replace("<blockquote>", "")

# code for various kinds of processing on the KG.
class KGPathsIndex:
    def __init__(self, kg_path: str):
        self.kg_path = kg_path
        self.KG_json = json.load(open(kg_path))
        self.KG_paths = self.index_paths(self.KG_json, [[]])

    def index_paths(self, sub_KG: Union[dict, list], paths: List[Union[str, List[tuple]]]) -> List[Union[str, List[tuple]]]:
        """recursive function to uncover the task-decompositon paths from the nested JSON structure"""
        # if isinstance(sub_KG, list) and isinstance(sub_KG[0], list) and len(sub_KG[0]) == 2:
        #     exp_paths = []
        #     for value in sub_KG:
        #         for path in paths:
        #             exp_paths.append(path+[value])
            
        #     return exp_paths
        if isinstance(sub_KG, list) and len(sub_KG) == 2 and (sub_KG[-1] in ["markdown", "code"]):
            exp_paths = []
            for path in paths:
                exp_paths.append(path+[sub_KG])
            
            return exp_paths
            
        exp_paths = [] # expanded paths.
        if isinstance(sub_KG, dict):    
            for key, value in sub_KG.items():
                exp_sub_paths = []
                for path in paths: exp_sub_paths.append(path+[key])
                exp_paths += self.index_paths(value, exp_sub_paths)
        elif isinstance(sub_KG, list):
            for value in sub_KG:
                exp_paths += self.index_paths(value, paths)

        return exp_paths

    def save(self, save_path: str):
        task_decomp_keyed_dict = defaultdict(lambda:[])
        for path in self.KG_paths:
            leaf = path[-1]
            task_decomp = "->".join([process_text(ele) for ele in path[:-1]])
            cell_type = leaf[1]
            content = process_cell_text(leaf[0], cell_type)
            # content = leaf[0]
            task_decomp_keyed_dict[task_decomp].append((
                content, cell_type,
            ))
        task_decomp_keyed_dict = dict(task_decomp_keyed_dict)
        with open(save_path, "w") as f:
            json.dump(task_decomp_keyed_dict, f, indent=4)

# main
if __name__ == "__main__":
    # target_module = "numpy" # "pandas_toms_blog" # "seaborn"
    os.makedirs("./scrape_tutorials/KG_paths", exist_ok=True)
    module_list = ["sklearn"] #["numpy", "pandas_toms_blog", "seaborn", "torch", "scipy", "sklearn"]
    for target_module in tqdm(module_list):
        kg_paths = KGPathsIndex(f"./scrape_tutorials/KGs/{target_module}.json")
        kg_paths_save_path = f"./scrape_tutorials/KG_paths/{target_module}.json"
        kg_paths.save(kg_paths_save_path)