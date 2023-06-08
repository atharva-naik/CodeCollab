# sample data for annotation for testing/comparing retrievers from the op2code data.
import json
import random
import pandas as pd
from collections import defaultdict

random.seed(2023)
# main
if __name__ == "__main__":
    op2code_data_obf: dict = json.load(open("./experiments/CoNaLa_CSN_CodeBERT_ObfCodeSearch4_CosSim/seed_query_op2code.json"))
    op2code_data_reg: dict = json.load(open("./experiments/CoNaLa_CSN_CodeBERT_CodeSearch2_CosSim/seed_query_op2code.json"))
    op2code_data = defaultdict(lambda: [])
    
    num_plan_ops = len(list(op2code_data_reg.keys()))
    assert list(op2code_data_reg.keys()) == list(op2code_data_obf.keys())
    print(f"found {num_plan_ops} plan operators!")
    for op, codes in op2code_data_obf.items():
        op2code_data[op] += codes[:20]
    for op, codes in op2code_data_reg.items():
        op2code_data[op] += codes[:20]
    for op, codes in op2code_data.items():
        dedup_codes = {code: score for code, score in codes}
        op2code_data[op] = list(sorted(dedup_codes.items(), reverse=True, key=lambda x: x[1]))
    op2code_data = dict(op2code_data)
    # print(len(op2code_data[op]))
    idx = random.sample(range(len(op2code_data)), k=200)
    all_queries = list(op2code_data.keys())
    data = []
    for ind in idx:
        query = all_queries[ind]
        docs = op2code_data[query]
        for doc in docs:
            data.append({"id": ind, "query": query, "doc": doc[0], "score": doc[1], "is_rel": ""})
    df = pd.DataFrame(data)
    df.to_csv("./data/annot_data_for_ret.csv", index=False)
    print(len(df))