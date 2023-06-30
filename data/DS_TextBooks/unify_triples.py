#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# extract triples from KB jsons and unify them.

import sys
import json
from typing import *
from collections import defaultdict

# dictionary of semantic types.
SEMANTIC_TYPES = json.load(open("./data/DS_TextBooks/semantic_types.json"))
REL_TYPES = defaultdict(lambda:[])

def predict_relation_type(sub: str, obj: str, ):
    pass

def extract_triples(kb_json: str) -> List[dict]:
    global SEMANTIC_TYPES
    triples = []
    for k1,v1 in kb_json.items():
        if k1 == "STEPS": continue # a step shouldn't be subject here.
        sub = k1.split("::")[0].strip()
        sub_sem_type = k1.split("::")[1].strip() # semantic type of subject
        if isinstance(v1, dict):
            for k2,v2 in v1.items():
                # a "HAS FIRST STEP" relation.
                if k2 == "STEPS":
                    assert isinstance(v2, list) and len(v2) > 0
                    triples.append({
                        "sub": (sub, sub_sem_type),
                        "obj": (v2[0], "s"),
                        "e": "HAS FIRST STEP"
                    })
                    triples.append({
                        "sub": (v2[0], "s"),
                        "obj": (sub, sub_sem_type),
                        "e": "IS FIRST STEP OF"
                    })
                    # "HAS NEXT STEP" relations.
                    for i in range(len(v2)-1):
                        triples.append({
                            "sub": (v2[i], "s"),
                            "obj": (v2[i+1], "s"),
                            "e": "HAS NEXT STEP"
                        })
                        triples.append({
                            "sub": (v2[i+1], "s"),
                            "obj": (v2[i], "s"),
                            "e": "HAS PREV STEP"
                        })
                    triples.append({
                        "sub": (sub, sub_sem_type),
                        "obj": (v2[-1], "s"),
                        "e": "HAS LAST STEP"
                    })
                    triples.append({
                        "sub": (v2[-1], "s"),
                        "obj": (sub, sub_sem_type),
                        "e": "IS LAST STEP OF"
                    })
                    continue
                obj = k2.split("::")[0].strip()
                obj_sem_type = k2.split("::")[1].strip() # semantic type of object
                triples.append({
                    "sub": (sub, sub_sem_type),
                    "obj": (obj, obj_sem_type),
                    "e": "CONSISTS OF"
                })
                triples.append({
                    "sub": (obj, obj_sem_type),
                    "obj": (sub, sub_sem_type),
                    "e": "IS A PART OF"
                })
                REL_TYPES[f"{sub_sem_type}::{obj_sem_type}"].append(
                    f"{sub} CONSISTS OF {obj}"
                )
                REL_TYPES[f"{obj_sem_type}::{sub_sem_type}"].append(
                    f"{obj} IS A PART OF {sub}"
                )
                # REL_TYPES.add(f"{obj_sem_type} has {sub_sem_type}")
                # skip recursion step if v2 is not a dictionary.
                if not isinstance(v2, dict): continue
                # recursion step.
                triples += extract_triples(kb_json={k2: v2})

    return triples

# main
if __name__ == "__main__":
    kb_path = sys.argv[1]
    kb_json = json.load(open(kb_path))
    triples = extract_triples(kb_json=kb_json)
    with open("./data/DS_TextBooks/unified_triples.json", "w") as f:
        json.dump(triples, f, indent=4)
    
    # visualize serializations of triples
    # for rel_type, instances in REL_TYPES.items():
    #     print("\x1b[34;1m#", rel_type+"\x1b[0m")
    #     for i, inst in enumerate(instances):
    #         print(f"{i+1}. {inst}")