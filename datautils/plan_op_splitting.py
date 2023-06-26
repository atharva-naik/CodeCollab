#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# script to process 

import json
import spacy
from typing import *
from tqdm import tqdm
from datautils import load_plan_ops

EXCEPTION_LIST = [
    "impute missing data values, in order to preserve as much of the time series as possible",
    "set the index to a datetimeindex type",
    "computing alpha, beta, and r squared in python",
    "get the sse by using the predictions for every x y_hats and the true y values",
    "helper to create two separate dataframes for classification and regression",
    "using principal component analysis to plot in two dimensions",
    "how many rows and columns are in kallisto",
]
SKIP_LIST = set([
    "introduction", "linear", "load", "loading",
    "do exploratory data analysis", "bar charts",
])
def split_plan_op(op: str, nlp, plan_ops: List[str]):
    # get 2 candidate operators from op
    if op in EXCEPTION_LIST: return op, None, None
    try: 
        op1, op2 = op.split(" to ")
        if op1 in plan_ops: return op1, op2, "TO"
        elif op2 in plan_ops: return op1, op2, "TO"
    except ValueError:
        if len(op.split(" to ")) > 2:
            op1 = op.split(" to ")[0]
            op2 = " to ".join(op.split(" to ")[1:])
            if op1 in plan_ops: return op1, op2, "TO"
            elif op2 in plan_ops: return op1, op2, "TO"
        else: op1, op2 = "", ""
    try:
        if " & " in op: op3, op4 = op.split(" & ")
        else: op3, op4 = op.split(" and ")
        if op3 in plan_ops: return op3, op4, "AND"
        elif op4 in plan_ops: return op3, op4, "AND"
    except ValueError: op3, op4 = "", ""
    # has_verb1 = any([w.pos_ == "VERB" for w in nlp(op1)])
    # has_verb2 = any([w.pos_ == "VERB" for w in nlp(op2)])
    # don't split case:
    if len(op2.split()) <= 2 or len(op1.split()) <= 2:
        if len(op3.split()) > 2 and len(op4.split()) > 2:
            # print(len(op3), len(op4))
            return op3, op4, "AND"
        else: return op, None, None
    else: return op1, op2, "TO"

def dissolve_compound_plan_ops(plan_ops: List[str], nlp):
    # plan ops having 'and' and 'or'.
    has_to_or_and = [w for w in plan_ops if (" to " in w) or (" and " in w)]
    rest_of_plan_ops = [w for w in plan_ops if w not in has_to_or_and]
    # plan ops which are split.
    split_ops = set()
    # write plan operator splits to a text file.
    with open("./data/juice-dataset/plan_op_splits.txt", "w") as f:
        for op in tqdm(has_to_or_and):
            op1, op2, JOIN_op = split_plan_op(op, nlp, plan_ops)
            if op2 is None: 
                split_ops.add(op1)
                f.write(f"{op}\n")
            else: 
                split_ops.add(op1)
                split_ops.add(op2)
                f.write(f"{op1}   |{JOIN_op}|   {op2}\n")
    # split compound plan operators and find change in the number of plan operators.
    print(f"original no. of plan ops: {len(plan_ops)}")
    plan_ops = sorted(list(
        split_ops.union(set(rest_of_plan_ops)).difference(SKIP_LIST)
    ))

    return plan_ops

# main
if __name__ == "__main__":
    # spacy model loaded `en_core_web_lg`
    nlp = spacy.load("en_core_web_lg")
    # load plan operators.
    plan_ops = load_plan_ops()
    plan_ops = dissolve_compound_plan_ops(plan_ops, nlp)
    # print and save final no. of plan ops.
    print(f"final no. of plan ops: {len(plan_ops)}")
    with open("./data/juice-dataset/plan_ops.json", "w") as f:
        json.dump(plan_ops, f, indent=4)