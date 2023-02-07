import json
import random
import pandas as pd
from typing import *
from datautils import read_jsonl

BUCKET_SIZE = 25
# bucket size number of instances for each context size: 2,5,10.
# reapeat for markdown and code as the next target cell.
# repeat for both train and val split of the data.
# total = 2 x 2 x 3 x 25 = 300 

def prepare_instance(inst, ctxt_size: int=2, 
                     target_type: str="markdown") -> dict:
    sample = {"context": "", "context_size": ctxt_size, "pred": ""}
    if target_type == "markdown":
        ctxt = inst["context"][::-1][:-1][-ctxt_size:]
    else: ctxt = inst["context"][::-1][-ctxt_size:]
    ctxt_pairs = [(cell.get("nl_original", cell.get("code")), 
                  cell['cell_type'], cell['distance_target']) for cell in ctxt]
    # type_to_tag = {"markdown": "text"}
    sample['context'] = "\n".join([f'''\n{cell_type}:\n{content}\n''' for content, cell_type, dist in ctxt_pairs])

    return sample

def prepare_instances(list_of_inst: List[dict], split_name: str):
    true_vales = {}
    df = []
    cell_type_ctxt_size_info = []
    for i in range(BUCKET_SIZE):
        for cell_type in ["markdown", "code"]:
            for ctxt_size in [2,5,10]:
                cell_type_ctxt_size_info.append((
                    cell_type, ctxt_size
                ))
    N = len(cell_type_ctxt_size_info)
    shuffled_idx = random.sample(range(N), k=N)
    index_ctr = 0
    for i in shuffled_idx:
        cell_type, ctxt_size = cell_type_ctxt_size_info[i]
        inst = list_of_inst[i]
        if cell_type == "markdown":
            target_content = inst["context"][::-1][-1]['nl_original']
        else: target_content = inst["code"]
        rec = prepare_instance(list_of_inst[i], 
        target_type=cell_type, ctxt_size=ctxt_size)
        rec["id"] = index_ctr
        rec["split_name"] = split_name
        true_vales[index_ctr] = (cell_type, target_content)
        index_ctr += 1
        df.append(rec)
    df = pd.DataFrame(df)

    return df, true_vales

if __name__ == "__main__":
    # from train.
    train: List[dict] = read_jsonl("./data/juice-dataset/train.jsonl", cutoff=80000)
    sampled_train = random.sample(train, k=BUCKET_SIZE*6)
    df, true_values = prepare_instances(sampled_train, "train")
    df.to_excel("./analysis/nctp_human_study_train.xlsx", index=False)
    with open("./analysis/nctp_human_study_train_trues.json", "w") as f:
        json.dump(true_values, f, indent=4)
    # from validation.
    dev: List[dict] = read_jsonl("./data/juice-dataset/dev.jsonl", cutoff=80000)
    sampled_dev = random.sample(dev, k=BUCKET_SIZE*6)
    df, true_values = prepare_instances(sampled_train, "val")
    df.to_excel("./analysis/nctp_human_study_val.xlsx", index=False)
    with open("./analysis/nctp_human_study_val_trues.json", "w") as f:
        json.dump(true_values, f, indent=4)