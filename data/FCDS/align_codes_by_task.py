# align code sequences belonging to the same task

import os
import json
from tqdm import tqdm
from collections import defaultdict

# main
if __name__ == "__main__":
    fcds_data = json.laod(open("./data/FCDS/code_submissions_data.json"))
    task_wise_data = defaultdict(lambda: [])
    for rec in fcds_data:
        task_wise_data[rec["task_name"]].append(rec)
    aligned_data = defaultdict(lambda: {})
    aligned_data_len = 0
    for task_name, student_submissions in task_wise_data.items():
        try: assert len(student_submissions) > 1
        except AssertionError: continue
        ref_sub = student_submissions[0]
        ref_len = len(student_submissions[0]["code"])
        try: assert len(student_submissions[1]["code"]) == ref_len
        except AssertionError: continue
        diff_indices = []
        ind = 0
        for cell1, cell2 in zip(student_submissions[0]["code"],
                                student_submissions[1]["code"]):
            cell_type1 = cell1["cell_type"]
            cell_type2 = cell2["cell_type"]
            try: assert cell_type1 == cell_type2
            except AssertionError: continue
            if cell_type1 == "markdown":
                c1 = cell1["markdown"] 
                c2 = cell2["markdown"]
            else: 
                c1 = cell1["code"]
                c2 = cell2["code"]
            if c1.strip() != c2.strip():    
                diff_indices.append(ind)
            ind += 1
        for i in tqdm(range(len(student_submissions))):
            person_id = student_submissions[i]["person_id"]
            try: assert len(student_submissions[i]["code"]) == ref_len
            except AssertionError: continue
            if len(student_submissions[i]["code"]) == 0: continue
            for j in diff_indices:
                try: probably_markdown = student_submissions[i]["code"][j-1]["markdown"]
                except KeyError: continue
                try: code = student_submissions[i]["code"][j]["code"]
                except KeyError: continue
                aligned_data[task_name][person_id] = {
                    "question": probably_markdown,
                    "answer": code, "diff_index": j,
                }
                for key, value in student_submissions[i].items():
                    if key == "code": continue
                    aligned_data[task_name][person_id][key] = value
                    aligned_data_len += 1
    # save the aligned student submissions data.
    with open("./data/FCDS/aligned_student_submissions_data.json", "w") as f:
        json.dump(aligned_data, f, indent=4)
    print(aligned_data_len)