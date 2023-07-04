# parse KB of procedures from markdown sequence in notebooks

import json
import numpy as np
from typing import *
from tqdm import tqdm
from datautils import read_jsonl
from scripts.gather_plan_operators import process_step
from datautils.markdown_cell_analysis import extract_notebook_hierarchy

def additional_processing_of_step(step: str):
    for term, s, e in [
        ("lab", 1, 1), 
        ("points", 1, 1), 
        ("point", 1, 1), 
        ("problem", 1, 0), 
        ('question', 1, 0),
        ("part", 1, 0),
        ("solution", 0, 1),
        ("solutions", 0, 1)
    ]:
        if step.startswith(f"{term} ") and s:
            step = step[len(term)+1:]
        if step.endswith(f" {term}") and e:
            step = step[:-(len(term)+1)]

    return step

def simplify_hier(nb_hier: dict):
    simple_hier = {}
    for child in nb_hier["children"]:
        if child["value"]["level"] == 1000: continue
        if child["value"]["type"] != "markdown": continue
        simple_child_hier = simplify_hier(child)
        key = additional_processing_of_step(process_step(
            child['value']["content"].split("\n")[0].strip()
        ))
        simple_hier[key] = simple_child_hier
    if len(simple_hier) == 0: return None 

    return simple_hier

def create_triples(nb_hier) -> Dict[str, Tuple[str, str, str]]:
    triples = {}
    if nb_hier is None: return []
    for k1, v1 in nb_hier.items():
        k1 = k1.strip()
        if not isinstance(v1, dict): continue
        prev_step = None
        for k2, v2 in v1.items():
            k2 = k2.strip()
            if prev_step is None: triples[f"{k1} has first step {k2}"] = (k1, "has first step", k2)
            else: 
                triples[f"{prev_step} has next step {k2}"] = (prev_step, "has next step", k2)
                triples[f"{k2} has prev step {prev_step}"] = (k2, "has prev step", prev_step)
            prev_step = k2        
            triples[f"{k1} has sub procedure {k2}"] = (k1, "has sub procedure", k2)
            triples[f"{k2} is sub procedure of {k1}"] = (k2, "is sub procedure of", k1)
            triples.update(create_triples({k2: v2}))
        if prev_step is not None:
            triples[f"{k1} has last step {k2}"] = (k1, "has last step", k2)
    
    return triples

# main
if __name__ == "__main__":
    # concat dev and test data.
    data = read_jsonl("./data/juice-dataset/traindedup.jsonl")
    data = read_jsonl("./data/juice-dataset/devdedup.jsonl")
    data += read_jsonl("./data/juice-dataset/testdedup.jsonl")
    all_triples = {}
    for inst in tqdm(data):
        root, triples = extract_notebook_hierarchy(inst)
        nb_hier = json.loads(root.to_json())
        triples = create_triples(simplify_hier(nb_hier))
        all_triples.update(triples)
        # print(len(triples))
    all_triples = list(all_triples.values())
    print(len(all_triples))
    with open("./data/juice-dataset/nb_kb_triples.json", "w") as f:
        json.dump(all_triples, f, indent=4)