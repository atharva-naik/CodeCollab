# evaluate retriever.
import json
import numpy as np
import pandas as pd
from collections import defaultdict

# main
if __name__ == "__main__":
    annot_path: str = "./data/prec_annot_data_for_ret2.csv"
    preds_path: str = "./experiments/CodeBERT_ensemble/seed_query_op2code.json"
    # preds_path: str = "./experiments/CoNaLa_CSN_UniXcoder_CodeSearch_CosSim/seed_query_op2code.json"
    # preds_path: str = "./experiments/CoNaLa_CSN_GraphCodeBERT_CodeSearch_CosSim/seed_query_op2code.json"
    # preds_path = "./experiments/CoNaLa_CSN_CodeBERT_CodeSearch2_Sym/seed_query_op2code.json"
    #  preds_path = "./experiments/CoNaLa_CSN_CodeBERT_CodeSearch2_CosSim/seed_query_op2code.json"
    # preds_path = "./experiments/CoNaLa_CSN_CodeBERT_ObfCodeSearch4_CosSim/seed_query_op2code.json"
    preds = json.load(open(preds_path))
    annots = pd.read_csv(annot_path).to_dict("records")
    query_to_codes = defaultdict(lambda: [])

    for rec in annots:
        if rec["is_rel"] == 1:
            q = rec["query"]
            d = rec["doc"]
            query_to_codes[q].append(d)
    print(len(query_to_codes), "samples")
    # measure recalls:
    recall_at_5 = 0
    recall_at_10 = 0
    recall_at_20 = 0
    # measure precision:
    prec_at_5 = []
    prec_at_10 = []
    prec_at_20 = []
    # distribution of number of labels
    # num_labels_list = []
    for q, labels in query_to_codes.items():
        ret = preds[q]
        # print(q)
        # print(labels[0])
        labels = [label.strip() for label in labels]
        # num_labels_list.append(len(labels))
        # compute recalls
        for code, score in ret[:5]:
            if code in labels:
                recall_at_5 += 1
                break
        for code, score in ret[:10]:
            if code in labels:
                recall_at_10 += 1
                break
        for code, score in ret[:20]:
            if code in labels:
                recall_at_20 += 1
                break
        # compute precision
        prec = 0
        for code, score in ret[:5]:
            if code in labels:
                prec += 1
        prec /= 5
        prec_at_5.append(prec)
        
        prec = 0
        for code, score in ret[:10]:
            if code in labels:
                prec += 1
        prec /= 10
        prec_at_10.append(prec)

        prec = 0
        for code, score in ret[:20]:
            if code in labels:
                prec += 1
        prec /= 20
        prec_at_20.append(prec)

    recall_at_5 /= len(query_to_codes)
    recall_at_10 /= len(query_to_codes)
    recall_at_20 /= len(query_to_codes)

    print("recall@5:", round(100*recall_at_5, 2))
    print("recall@10:", round(100*recall_at_10, 2))
    print("recall@20:", round(100*recall_at_20, 2))

    print("prec@5:", round(100*np.mean(prec_at_5), 2))
    print("prec@10:", round(100*np.mean(prec_at_10), 2))
    print("prec@20:", round(100*np.mean(prec_at_20), 2))

    # print("num labels min:", np.min(num_labels_list))
    # print("num labels max:", np.max(num_labels_list))
    # print("num labels mean:", round(np.mean(num_labels_list), 2))