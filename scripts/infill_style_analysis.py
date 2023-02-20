from typing import *
from tqdm import tqdm
from datautils import read_jsonl
from datautils.markdown_cell_analysis import get_title_hierarchy_and_stripped_title
# from datautils.code_cell_analysis import ast_parse, get_uniq_vars_and_funcs

if __name__ == "__main__":
    # do hierarchy mismatch analysis.
    infill = read_jsonl("./analysis/incoder_markdown_infill_val.jsonl")
    has_hier_tot = 0
    hier_match_tot = 0
    hier_pred_tot = 0
    for rec in tqdm(infill):
        true = rec["true"]
        pred = rec["pred"]
        ctr_t, title_t = get_title_hierarchy_and_stripped_title(true)
        ctr_p, title_p = get_title_hierarchy_and_stripped_title(pred)
        if ctr_p != 1000:
            hier_pred_tot += 1
        if ctr_t != 1000:
            has_hier_tot += 1
            if ctr_t == ctr_p:
                hier_match_tot += 1
            else:
                print(true)
                print("$"*10)
                print(pred)
                break
    print(f"cells starting with hierarchical titles: {has_hier_tot}/{len(infill)}")
    print(f"predicted cells with hierarchy {hier_pred_tot}/{len(infill)}")
    print(f"{hier_match_tot}/{has_hier_tot}")
    print(f"%age of hierarchy match: {100*hier_match_tot/has_hier_tot:.2f}%")