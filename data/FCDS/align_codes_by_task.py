# align code sequences belonging to the same task

import os
import json
import warnings
import numpy as np
from typing import *
from tqdm import tqdm
from scipy import stats
from collections import defaultdict

warnings.simplefilter(action='ignore', category=FutureWarning)

def get_prev_md(code_cells: List[dict], i: int=0):
    while code_cells[i]["cell_type"] != "markdown" and i >= 0: i -= 1
    if i >= 0: return code_cells[i]["markdown"]
    return None

def get_content(cell: dict) -> str:
    # print(cell)
    return cell[cell["cell_type"]].strip()

# main
if __name__ == "__main__":
    fcds_data = json.load(open("./data/FCDS/code_submissions_dataset.json"))
    print("empty NBs:", sum([len(rec["code"]) == 0 for rec in fcds_data]))
    print(len(fcds_data))
    task_wise_data = defaultdict(lambda: [])
    for rec in fcds_data:
        task_wise_data[rec["task_name"]].append(rec)
    aligned_data = defaultdict(lambda: defaultdict(lambda: []))
    avg_code_cells_per_delta = []
    avg_deltas_per_nb = []
    for task_name, student_submissions in task_wise_data.items():
        try: assert len(student_submissions) > 1
        except AssertionError: continue
        nb_lens = [len(rec['code']) for rec in student_submissions if len(rec["code"]) != 0]
        ref_sub = student_submissions[0]
        ref_len = stats.mode(nb_lens).mode[0]
        # min_len = np.min(nb_lens)
        # assert min_len == ref_len, f"{ref_len} {min_len}"
        ti = 0
        while len(student_submissions[ti]["code"]) != ref_len: ti += 1
        tj = ti + 1
        while len(student_submissions[tj]["code"]) != ref_len: tj += 1

        try:
            check_len_0 = len(student_submissions[ti]["code"])            
            check_len_1 = len(student_submissions[tj]["code"])
            assert check_len_0 == ref_len 
            assert check_len_1 == ref_len
        except AssertionError: 
            print(f"AssertionError at line 26: {ref_len}, {check_len_0}, {check_len_1}")
        ref_nb = student_submissions[ti]
        # assert len(delta_codes) == len(delta_mds)
        blank_ctr = 0 # number of blank notebooks.
        inc_ctr = 0 # incomplete counter (notebooks for which all mds couldn't be matched)
        # iterate over all notebooks submitted for a given task.
        for index in tqdm(range(len(student_submissions))):
            nb = student_submissions[index]
            metadata = {k: v for k,v in nb.items() if k != "code"}
            # skip blank notebooks
            if len(nb["code"]) == 0: 
                blank_ctr += 1
                continue
            nb_i = 0
            ref_i = 0
            code_content = defaultdict(lambda: [])
            cell_ids = defaultdict(lambda: [])
            while nb_i < len(nb["code"]) and ref_i < ref_len:
                c_nb = get_content(nb["code"][nb_i])
                c_ref = get_content(ref_nb["code"][ref_i])
                if c_nb == c_ref: # skip if the cells are aligned
                    nb_i += 1
                    ref_i += 1
                else: # unequal cells.
                    # ignore unequal markdown cells (slight variation of the same cell)
                    if len(nb["code"]) == ref_len:
                        if nb["code"][nb_i]["cell_type"] == "code":
                            md = get_prev_md(nb["code"], nb_i)
                            code_content[md].append(c_nb)
                            cell_ids[md].append(nb_i)
                        nb_i += 1
                        ref_i += 1
                    elif len(nb["code"]) < ref_len:
                        ref_i += 1
                    else:
                        if nb["code"][nb_i]["cell_type"] == "code":
                            md = get_prev_md(nb["code"], nb_i)
                            code_content[md].append(c_nb)
                            cell_ids[md].append(nb_i)
                        nb_i += 1
            num_deltas = 0
            for question, answer_cells in code_content.items():
                answer = "\n".join(answer_cells)
                num_deltas += len(answer_cells)
                rec = {
                    "question": question, "answer": answer,
                    "answer_cell_ids": cell_ids[question], # slightly heuristic based.
                }
                avg_code_cells_per_delta.append(len(cell_ids[question]))
                rec.update(metadata)
                aligned_data[task_name][nb["person_id"]].append(rec)
            avg_deltas_per_nb.append(num_deltas)
            # for j in diff_indices:
            #     try: probably_markdown = student_submissions[i]["code"][j-1]["markdown"]
            #     except KeyError: continue
            #     try: code = student_submissions[i]["code"][j]["code"]
            #     except KeyError: continue
            #     aligned_data[task_name][person_id] = {
            #         "question": probably_markdown,
            #         "answer": code, "diff_index": j,
            #     }
            #     for key, value in student_submissions[i].items():
            #         if key == "code": continue
            #         aligned_data[task_name][person_id][key] = value
    print(f"inc_ctr: {inc_ctr}")
    print(f"blank_ctr: {blank_ctr}")
    # save the aligned student submissions data.
    with open("./data/FCDS/aligned_student_submissions_data.json", "w") as f:
        json.dump(aligned_data, f, indent=4)
    aligned_data_len = 0
    print(f"avg code cells per delta: {round(np.mean(avg_code_cells_per_delta), 3)}")
    print(f"num deltas: {sum(avg_deltas_per_nb)}")
    print(f"avg deltas: {round(np.mean(avg_deltas_per_nb), 3)}")
    for task_name in aligned_data: # print(task_name)
        for person_id in aligned_data[task_name]:
            aligned_data_len += len(aligned_data[task_name][person_id])
    print(f"aligned_data_len: {aligned_data_len}")