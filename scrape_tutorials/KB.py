#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# definition and utilities related to the KB (searching, splitting, joining etc.)
# and code to analyze KB statistics and properties.

import json
import torch
import numpy as np
from typing import *
from tqdm import tqdm
from collections import defaultdict
from sentence_transformers import SentenceTransformer

class TutorialPathsKB:
    def __init__(self, path: str="./scrape_tutorials/unified_KG.json"):
        self.data = json.load(open(path))
        self.path_index: Dict[str, List[dict]] = defaultdict(lambda:[]) # indexed by step names.
        self.path_phrases = [", ".join(k.split("->")) for k in self.data]
        
        self.sbert = SentenceTransformer("multi-qa-mpnet-base-dot-v1")
        if torch.cuda.is_available(): self.sbert.cuda()
        self.path_dense_matrix = []
        self.step_dense_matrix = []

        for path_decomp_key, value in tqdm(self.data.items()):
            sub_path = []
            full_decomp = path_decomp_key.split("->")
            for i, step in enumerate(full_decomp):
                step = step.strip()
                sub_path.append(step)
                self.path_index[step].append({
                    "prefix_decomp": sub_path,
                    "full_decomp": full_decomp,
                    "leaf_cells": value,
                    "step_index": i,
                    "next_step": full_decomp[i+1] if i<len(full_decomp)-1 else "",
                    "depth": i+1, # essentialy the same as step index, just begins at 1.
                    "module": full_decomp[0] if i>0 else "",
                })

    def build_semantic_step_index(self):
        self.step_dense_matrix = []                
        for step in tqdm(self.path_index):
            self.step_dense_matrix.append(self.sbert.encode(step))
        self.step_dense_matrix = np.stack(self.step_dense_matrix)

    def build_semantic_path_index(self):
        self.path_dense_matrix = []
        for path in tqdm(self.path_phrases, desc="encoding path phrases"):
            self.path_dense_matrix.append(self.sbert.encode(path))
        self.path_dense_matrix = np.stack(self.path_dense_matrix)

    def semantic_search_for_step(self, kp: str, k: int=5):
        if len(self.step_dense_matrix) == 0:
            self.build_semantic_step_index()
        kp_enc = self.sbert.encode(kp)
        scores = self.step_dense_matrix @ kp_enc.T
        result = []
        simplified_result = defaultdict(lambda: set())
        all_steps = list(self.path_index.keys())
        for id in scores.argsort()[::-1][:k]:
            step = all_steps[id]
            rec = self.path_index[step]
            result.append(rec)
            for d in rec:
                full_decomp = "->".join(d["full_decomp"])
                simplified_result[step].add(full_decomp)
        simplified_result = dict(simplified_result)

        return result, simplified_result

    def semantic_search(self, kp: str, k: int=10) -> List[str]:
        if len(self.path_dense_matrix) == 0:
            self.build_semantic_path_index()
        kp_enc = self.sbert.encode(kp)
        scores = self.path_dense_matrix @ kp_enc.T
        result = []
        all_paths = list(self.data.keys())
        for id in scores.argsort()[::-1][:k]:
            result.append(all_paths[id])

        return result

    def search(self, query, depth: int=1) -> List[Tuple[dict, float]]:
        """## Algorithm:
        1. do an exact search over the `path` inverted index.
        2. if not hits for above then do direct search, use fuzzy matching with threshold.
        3. if above also fails, try doing semantic search maybe (and also use value).
        
        - Returns: list of step matches and paired scores."""
        assert depth >= 1, "Depth should be >= 1"
        step_matches = self.path_index.get(query,[])
        if len(step_matches)>0: # order by absolute depth/distance.
            scores_and_matches = [] # order the step matches based on depth distance.
            for step in step_matches: scores_and_matches.append((step, abs(step["depth"]-depth)))
            scores_and_matches = sorted(scores_and_matches, key=lambda x: x[1], reverse=False) # least depth distance scored highest.
            return scores_and_matches
        else:
            # do fuzzy search
            return []

    def get_next_steps_from_matches(self, matches: List[dict]) -> Dict[str, float]:
        """score the next steps based on how many paths they occur in.
        ofc they will favor more decomposable paths as a result."""
        next_steps = defaultdict(lambda:0)
        for inst in matches:
            next_steps[inst["next_step"]] += 1
        Z = sum(next_steps.values())
        next_steps = {k: v/Z for k,v in sorted(next_steps.items(), reverse=True, key=lambda x: x[1])}

        return next_steps

# main
if __name__ == "__main__":
    KB_paths = [
        [
            step.strip() for step in task_decomp.split("->")
        ] for task_decomp in json.load(open("./scrape_tutorials/unified_KG_paths.json"))
    ]
    # Get general stats about the index
    # num paths:
    print("num paths:", len(KB_paths))
    # num modules:
    module_ctr = defaultdict(lambda:0)
    for path in KB_paths: module_ctr[path[0]] += 1
    module_ctr = {k: v for k,v in sorted(module_ctr.items(), reverse=True, key=lambda x: x[1])}
    print("num modules:", len(module_ctr))
    # num paths per module:
    print("num paths per module:")
    for module, num_paths in module_ctr.items():
        print(module, num_paths)
    # num steps:
    uniq_steps = set()
    path_depths = []
    for path in KB_paths:
        path_depths.append(len(path))
        for step in path:
            uniq_steps.add(step.strip())
    print(f"num steps: {len(uniq_steps)}")
    # average path depth/decomp steps:
    print("average path depth/decomp steps:", round(np.mean(path_depths), 3))
    # max path depth:
    print("max path depth:", np.max(path_depths))
    # min path depth:
    print("min path depth:", np.min(path_depths))