import json
import treelib
import pandas as pd
from typing import *
from collections import defaultdict
from data.FCDS.code_chunking import extract_op_chunks

def convert_chunk_tree_to_plan(chunk_tree: dict, plan_op_mapping: dict):
    """recursively convert a tree/nested dictionary of code chunks to a nested dictionary of plan step names."""
    plan_tree = {}
    for chunk, sub_chunk_tree in chunk_tree.items():
        plan_op = plan_op_mapping[chunk]
        plan_tree[plan_op] = convert_chunk_tree_to_plan(sub_chunk_tree, plan_op_mapping)

    return plan_tree

def rec_build_tree(plan_tree, tree, root):
    """recursively build plan tree"""
    for key, subtree in plan_tree.items():
        tree.create_node(key, key, parent=root)
        rec_build_tree(subtree, tree, key)

def print_plan_tree(plan_tree):
    tree = treelib.Tree()
    tree.create_node("root", "root")
    rec_build_tree(plan_tree["root"], tree, "root")
    tree.show()

# main
if __name__ == "__main__":
    annotations = pd.read_csv("./data/FCDS/FCDS Plan Operator Annotations.csv")
    annotations_grouped_by_id = defaultdict(lambda: [])
    annot_ctr = 0
    visualizations = {}
    plan_op_to_code = defaultdict(lambda: [])
    for rec in annotations.to_dict("records"):
        plan_op = rec["plan operator"]
        if isinstance(plan_op, str) and plan_op.strip() == "": continue
        elif isinstance(plan_op, float): continue # probably a float/nan.
        annotations_grouped_by_id[rec["id"]].append(rec)
        annot_ctr += 1
    # print(annot_ctr)
    # exit()
    for id, recs in annotations_grouped_by_id.items():
        code = recs[0]["answer"]
        # print(code)
        chunk_tree, flat_chunk_list = extract_op_chunks(code)
        plan_op_mapping = {}
        for rec in recs:
            plan_op_mapping[rec["block"]] = rec["plan operator"]
            plan_op_to_code[rec["plan operator"]].append(rec["block"])
        try: assert len(flat_chunk_list) == len(plan_op_mapping), f"{len(flat_chunk_list)}, {len(plan_op_mapping)}"
        except AssertionError as e: # to analyze errors.
            print(e)
            print(f"\x1b[34;1m#plan_op_mapping:\x1b[0m")
            for chunk in plan_op_mapping:
                if chunk not in flat_chunk_list:
                    print("-"*30)
                    print(chunk)
            print(f"\x1b[34;1m#flat_chunk_list:\x1b[0m")
            for chunk in flat_chunk_list:
                if chunk not in plan_op_mapping:
                    print("-"*30)
                    print(chunk)
            exit() 
        plan_tree = convert_chunk_tree_to_plan(chunk_tree, plan_op_mapping)
        # treelib doesn't preserve order
        # print_plan_tree({"root": plan_tree})
        visualizations[id] = {
            "chunk_tree": chunk_tree,
            "plan_tree": plan_tree,
            "chunk_to_plan_op": plan_op_mapping,
        }
    with open("./data/FCDS/plan_op_annotations.json", "w") as f:
        json.dump(plan_op_to_code, f, indent=4)
    with open("./data/FCDS/plan_op_visualizations.json", 'w') as f:
        json.dump(visualizations, f, indent=4)
