#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# extract triples from KB jsons and unify them.

import os
import sys
import json
from typing import *
from collections import defaultdict

# dictionary of semantic types.
SEMANTIC_TYPES = json.load(open("./data/DS_TextBooks/semantic_types.json"))
SEMANTIC_TYPE_DIST = defaultdict(lambda: 0) 
RELATION_TYPE_DIST = defaultdict(lambda: 0)
UNIQUE_NODES = {}

def get_reverse_relation(rel_type: str) -> str:
    rel_type = rel_type.strip().upper()
    mapping = {"has part(s)": "part of", "influenced by": "influences", "uses": "used by",
               "has instance": "instance of", "modeled by": "can model", "superclass of": "subclass of",
               "evaluated by": "evaluates", "has language": "has dataset", "better than": "worse than"}
    inv_mapping = {v: k for k,v in mapping.items()}
    mapping.update(inv_mapping)
    for key, value in mapping.items():
        key = key.strip().upper()
        if rel_type == key: 
            return value.strip().upper()
    
    return "IS A PART OF"

def format_dist(d: dict):
    return "\n".join([f"{k}: {v}" for k,v in d.items()])

def predict_relation_type(sub: str, obj: str, ):
    pass

def extract_triples(kb_json: str) -> List[dict]:
    global UNIQUE_NODES
    global SEMANTIC_TYPES
    global RELATION_TYPE_DIST
    
    triples = {}
    for k1,v1 in kb_json.items():
        if k1 == "STEPS": continue # STEPS shouldn't be subject here.
        if k1 == "DESC": continue # DESC shouldn't be subject here.
        k1_s = k1.split("::")
        k1_desc = ""
        sub = k1_s[0].strip()
        sub_sem_type = k1_s[1].strip() # semantic type of subject
        UNIQUE_NODES[sub] = sub_sem_type
        if isinstance(v1, dict):
            for k2,v2 in v1.items():
                # a "HAS FIRST STEP" relation.
                if k2 == "DESC":
                    assert isinstance(v2, str)
                    k1_desc = v2
                    continue
                elif k2 == "STEPS":
                    assert isinstance(v2, list) and len(v2) > 0
                    triples[f"{sub} HAS FIRST STEP {v2[0]}"] = {
                        "sub": (sub, sub_sem_type, k1_desc),
                        "obj": (v2[0], "s", ""),
                        "e": "HAS FIRST STEP"
                    }
                    UNIQUE_NODES[v2[0]] = "STEP"
                    RELATION_TYPE_DIST["HAS FIRST STEP"] += 1
                    triples[f"{v2[0]} IS FIRST STEP OF {sub}"] = {
                        "sub": (v2[0], "s", ""),
                        "obj": (sub, sub_sem_type, k1_desc),
                        "e": "IS FIRST STEP OF"
                    }
                    RELATION_TYPE_DIST["IS FIRST STEP OF"] += 1
                    # "HAS NEXT STEP" relations.
                    for i in range(len(v2)-1):
                        triples[f"{v2[i]} HAS NEXT STEP {v2[i+1]}"] = {
                            "sub": (v2[i], "s", ""),
                            "obj": (v2[i+1], "s", ""),
                            "e": "HAS NEXT STEP"
                        }
                        UNIQUE_NODES[v2[i+1]] = "STEP"
                        RELATION_TYPE_DIST["HAS NEXT STEP"] += 1
                        triples[f"{v2[i+1]} HAS PREV STEP {v2[i]}"] = {
                            "sub": (v2[i+1], "s", ""),
                            "obj": (v2[i], "s", ""),
                            "e": "HAS PREV STEP"
                        }
                        RELATION_TYPE_DIST["HAS PREV STEP"] += 1
                    triples[f"{sub} HAS LAST STEP {v2[-1]}"] = {
                        "sub": (sub, sub_sem_type, k1_desc),
                        "obj": (v2[-1], "s", ""),
                        "e": "HAS LAST STEP"
                    }
                    RELATION_TYPE_DIST["HAS LAST STEP"] += 1
                    triples[f"{v2[-1]} IS LAST STEP OF {sub}"] = {
                        "sub": (v2[-1], "s", ""),
                        "obj": (sub, sub_sem_type, k1_desc),
                        "e": "IS LAST STEP OF"
                    }
                    RELATION_TYPE_DIST["IS LAST STEP OF"] += 1
                    continue
                k2_s = k2.split("::")
                obj = k2_s[0].strip()
                obj_sem_type = k2_s[1].strip() # semantic type of object
                UNIQUE_NODES[obj] = obj_sem_type
                rel_type = "CONSISTS OF" if len(k2_s) == 2 else k2_s[2].strip().upper()
                rev_rel_type = get_reverse_relation(rel_type)
                triples[f"{sub} {rel_type} {obj}"] = {
                    "sub": (sub, sub_sem_type, k1_desc),
                    "obj": (obj, obj_sem_type, ""),
                    "e": rel_type
                }
                RELATION_TYPE_DIST[rel_type] += 1
                triples[f"{obj} {rev_rel_type} {sub}"] = {
                    "sub": (obj, obj_sem_type, ""),
                    "obj": (sub, sub_sem_type, k1_desc),
                    "e": rev_rel_type
                }
                RELATION_TYPE_DIST[rev_rel_type] += 1
                # skip recursion step if v2 is not a dictionary.
                if not isinstance(v2, dict): continue
                # recursion step.
                triples.update(extract_triples(kb_json={k2: v2}))

    return triples

# main
if __name__ == "__main__":
    unified_triples = {}
    for path in os.listdir("./data/DS_TextBooks"):
        if not path.endswith(".json"): continue
        if path == "semantic_types.json": continue
        if path == "unified_triples.json": continue
        if path == "sample.json": continue
        full_path = os.path.join("./data/DS_TextBooks", path)
        try: kb_json = json.load(open(full_path))
        except json.decoder.JSONDecodeError as e:
            print(f"{path}: {e}")
            continue
        try:
            triples = extract_triples(kb_json=kb_json)
            print(f"{path}: \x1b[34;1m{len(triples)}\x1b[0m triples")
            unified_triples.update(triples)
        except Exception as e: print(f"{path}: {e}")
    for semantic_type in UNIQUE_NODES.values():
        SEMANTIC_TYPE_DIST[SEMANTIC_TYPES.get(semantic_type, semantic_type)] += 1
    SEMANTIC_TYPE_DIST = {k: v for k,v in sorted(SEMANTIC_TYPE_DIST.items(), reverse=True, key=lambda x: x[1])}
    RELATION_TYPE_DIST = {k: v for k,v in sorted(RELATION_TYPE_DIST.items(), reverse=True, key=lambda x: x[1])}
    # print distribution of semantic types and relation types.
    print("\x1b[34;1mSemantic Type Distribution:\x1b[0m")
    print(format_dist(SEMANTIC_TYPE_DIST))
    print("\x1b[34;1mRelation Type Distribution:\x1b[0m")
    print(format_dist(RELATION_TYPE_DIST))
    print(f"|N|: \x1b[34;1m{len(UNIQUE_NODES)}\x1b[0m nodes")
    print(f"|E|: \x1b[34;1m{len(unified_triples)}\x1b[0m triples")
    unified_triples = list(unified_triples.values())
    with open("./data/DS_TextBooks/unified_triples.json", "w") as f:
        json.dump(unified_triples, f, indent=4)