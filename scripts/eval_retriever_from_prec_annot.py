# evaluate retriever.
import json
import pandas as pd
from collections import defaultdict

# main
if __name__ == "__main__":
    annot_path = "./data/prec_annot_data_for_ret.csv"
    preds_path = "./experiments/CoNaLa_CSN_CodeBERT_CodeSearch2_CosSim/seed_query_op2code.json"
    # preds_path = "./experiments/CoNaLa_CSN_CodeBERT_ObfCodeSearch4_CosSim/seed_query_op2code.json"
    preds = json.load(open(preds_path))
    annots = pd.read_csv(annot_path).to_dict("records")
    query_to_codes = defaultdict(lambda:[])
    for rec in annots:
        if rec["is_rel"] == 1:
            q = rec["query"]
            d = rec["doc"]
            query_to_codes[q].append(d)
    # print(len(query_to_codes))
    recall_at_10 = 0
    for q, labels in query_to_codes.items():
        ret = preds[q]
        print(q)
        # print(len(labels))
        for code, score in ret[:10]:
            # print(code)
            if code in labels:
                recall_at_10 += 1
                break
    recall_at_10 /= len(query_to_codes)
    print(recall_at_10)