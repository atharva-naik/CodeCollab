# script to sample data for P1-P2-P3 style annotation.
import json
import random
import pandas as pd
from datautils import read_jsonl

random.seed(2023)

# main
if __name__ == "__main__":
    code_KB = json.load(open("./JuICe_train_code_KB.json"))
    codes = list(code_KB.keys())
    paths = list(code_KB.values())
    knns = read_jsonl("./analysis/codebm25_topic_and_struct_10nn.jsonl")
    data = []
    for i, rec in enumerate(knns):
        p1_code = codes[i]
        cands = []
        p1_paths = "\n".join(paths[i])
        for id, score in rec["combined"]:
            id = int(id)
            cands.append(id)
        if len(cands) == 0: continue
        pos_id = random.sample(cands, k=1)[0]
        p2_code = codes[pos_id]
        p2_paths = "\n".join(paths[pos_id])
        neg_id = i
        while (neg_id in cands) or (paths[neg_id] == paths[i]) or (codes[neg_id] == codes[i]):
            neg_id = random.sample(range(len(codes)), k=1)[0]
        p3_code = codes[neg_id]
        p3_paths = "\n".join(paths[neg_id])
        data.append({
            "id": i, "p1_code": p1_code, 
            "p2_code": p2_code, "p3_code": p3_code, #"p4_code": p4_code,
            "op_name": "", "p1_paths": p1_paths, 
            "p2_paths": p2_paths, "p3_paths": p3_paths, #"p4_paths": p4_paths,
        })
    sampled_data = random.sample(data, k=1000)
    sampled_data_df = pd.DataFrame(sampled_data)
    sampled_data_df.to_csv("./analysis/p1p2p3_study.csv", index=False)