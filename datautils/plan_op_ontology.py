#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# script to process 

import json
import spacy
from typing import *
from tqdm import tqdm
from datautils.plan_op_splitting import dissolve_compound_plan_ops
from datautils import load_plan_ops, collapse_plan_operators, build_ontology, simplify_ontology, save_ontology

# main
if __name__ == "__main__":
    plan_ops = load_plan_ops()
    # spacy model loaded `en_core_web_lg`
    nlp = spacy.load("en_core_web_lg")
    plan_ops = dissolve_compound_plan_ops(plan_ops, nlp)
    with open("./data/juice-dataset/plan_ops.json", "w") as f:
        json.dump(plan_ops, f, indent=4)
    print("plan_ops:", len(plan_ops))
    collapse_pairs, top_parents = collapse_plan_operators(plan_ops)
    root, name_to_node = build_ontology(plan_ops, top_parents, collapse_pairs)
    name_to_node = simplify_ontology(name_to_node)
    save_ontology(name_to_node)