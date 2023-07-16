#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# sample Q/A submissions data for plan operator annotations.
import json
import random
import pandas as pd
from typing import *
from data.FCDS.code_chunking import chunk_code_to_variable_blocks, extract_op_chunks

# main
if __name__ == "__main__":
    random.seed(42) # random.seed(2023)
    stratum_size: int = 10 # number of sample per question
    qa_sub_data = json.load(open("./data/FCDS/code_qa_submissions.json"))
    print(f"Q/A submissions data has {len(qa_sub_data)} questions")
    annot_df = []
    num_sub = 0
    for q,a in qa_sub_data.items():
        sampling_k = min(stratum_size, len(a))
        if len(a) < stratum_size: print(q, len(a))
        subs: List[dict] = random.sample(a, k=sampling_k)
        # for sub in subs:
        #     var_blocks = chunk_code_to_variable_blocks(sub["answer"])
        #     block_id = 0
        #     num_sub += 1
        #     for var_name, block in var_blocks.items():
        #         block_id += 1
        #         rec = {
        #             "id": sub["qa_id"], "blockid": block_id,
        #             "block": block, "plan operator": "TODO",
        #             "variable": var_name, "answer": sub["answer"],
        #             "answer with test": sub["answer_with_test"],

        #         }
        #         annot_df.append(rec)
        for sub in subs:
            _, flat_chunk_list = extract_op_chunks(sub["answer"])
            block_id = 0
            num_sub += 1
            for chunk in flat_chunk_list:
                block_id += 1
                rec = {
                    "id": sub["qa_id"], "blockid": block_id, "block": chunk, 
                    "plan operator": "", "answer": sub["answer"],
                    "answer with test": sub["answer_with_test"],
                }
                annot_df.append(rec)
    annot_df = pd.DataFrame(annot_df)
    annot_df.to_csv("./data/FCDS/plan_op_code_chunks_annot.csv", index=False)
    print(f"sampling with stratum size: {stratum_size}")
    print(f"data contains {num_sub} submissions")
    print(f"{len(annot_df)} annotations to be done")