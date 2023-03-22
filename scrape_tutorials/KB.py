#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# definition and utilities related to the KB (searching, splitting, joining etc.)
# and code to analyze KB statistics and properties.

import json
import numpy as np
from typing import *
from tqdm import tqdm
from collections import defaultdict

class TutorialPathsKB:
    def __init__(self, path: str="./scrape_tutorials/unified_KG.json"):
        self.data = json.load(open(path))
        self.path_index: Dict[str, List[dict]] = defaultdict(lambda:[]) # indexed by step names.
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