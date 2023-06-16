# moved to datautils

# # build tree like ontology structure from operator to code mapping.
# import json
# import numpy as np
# from tqdm import tqdm
# from collections import defaultdict

# # main
# if __name__ == "__main__":
#     op2code = json.load(open("./experiments/CoNaLa_CSN_CodeBERT_CodeSearch2_CosSim/seed_query_op2code.json"))
#     overlap_map = defaultdict(lambda: defaultdict(lambda: 0))
#     all_overlaps = []
#     pc_thresh = 5
#     pc_pairs = []
#     for i in tqdm(range(len(op2code)), desc="computing overlap"):
#         for j in range(i+1, len(op2code)):
#             op_i = list(op2code)[i]
#             op_j = list(op2code)[j]
#             code_set_i = set(code.strip() for code,_ in list(op2code.values())[i])
#             code_set_j = set(code.strip() for code,_ in list(op2code.values())[j])
#             overlap_score = len(code_set_i.intersection(code_set_j))
#             all_overlaps.append(overlap_score)
#             if overlap_score > pc_thresh:
#                 pc_pairs.append((op_i, op_j))
#             overlap_map[op_i][op_j] = overlap_score
#     avg_overlap_score = np.mean(all_overlaps)
#     print(f"average overlap score = {avg_overlap_score}")
#     with open("./experiments/CoNaLa_CSN_CodeBERT_CodeSearch2_CosSim/seed_query_op_overlap_scores.json", "w") as f:
#         json.dump(overlap_map, f, indent=4)
#     with open("./experiments/CoNaLa_CSN_CodeBERT_CodeSearch2_CosSim/seed_query_op_pc_pairs.json", "w") as f:
#         json.dump(pc_pairs, f, indent=4)