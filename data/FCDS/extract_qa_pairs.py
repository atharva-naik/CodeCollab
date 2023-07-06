# align code sequences belonging to the same task

import os
import json
import warnings
import numpy as np
from typing import *
from tqdm import tqdm
from scipy import stats
from collections import defaultdict

# filter FutureWarning(s)
warnings.simplefilter(action='ignore', category=FutureWarning)

# unique instance IDs.
UNIQ_IDS = set()

def get_content(cell: dict) -> str:
    # print(cell)
    return cell[cell["cell_type"]].strip()

def extract_intent_from_question(question: str):
    return question.split("Question")[-1].split(":")[-1].split("(")[0].strip()

# main
if __name__ == "__main__":
    fcds_data = json.load(open("./data/FCDS/code_submissions_dataset.json"))
    print("empty NBs:", sum([len(rec["code"]) == 0 for rec in fcds_data]))
    
    task_name_to_questions = defaultdict(lambda: {})
    task_name_to_other_mds = defaultdict(lambda: {})
    data = defaultdict(lambda: []) # intent to student submissions mapping.
    intent_not_covered = defaultdict(lambda: True)

    inst_ctr = 0 # count unique instances in the Q/A pairs.
    for rec in tqdm(fcds_data):
        metadata = {k: v for k,v in rec.items() if k != "code"}
        cells = rec["code"]
        i = 0
        while i < len(cells):
            cell = cells[i]
            if cell["cell_type"] == "markdown":
                # filter out question like markdowns
                question = cell["markdown"].strip("#").strip().split("\n")[0].strip()
                if question.lower().startswith("question"):
                    # first line of the markdown
                    intent = extract_intent_from_question(question)
                    task_name_to_questions[rec["task_name"]][intent] = None
                    rec = {"intent": intent}
                    answer_cells = []
                    while cell["cell_type"] != "code":
                        i += 1
                        if i >= len(cells): break
                        cell = cells[i]
                    while cell["cell_type"] == "code":
                        if cell["code"].strip() != "":
                            answer_cells.append(cell["code"])
                        i += 1
                        if i >= len(cells): break
                        cell = cells[i]
                    # sanity checking
                    # if intent_not_covered[intent] == True:
                    #     print(f"\x1b[34;1m{intent}\x1b[0m")
                    #     print(answer_cells[0])
                    #     print("-"*50+"\n"+"\n".join(answer_cells))
                    #     intent_not_covered[intent] = False
                    inst_ctr += 1
                    rec.update(metadata)
                    rec["qa_id"] = inst_ctr
                    rec["answer"] = answer_cells[0]
                    rec["answer_with_test"] = "\n".join(answer_cells)
                    UNIQ_IDS.add(rec["id"])
                    data[intent].append(rec)
            i += 1
                # else:
                #     task_name_to_other_mds[rec["task_name"]][
                #         cell["markdown"].split("\n")[0].strip()
                #     ] = None                    
    
    # analyze list of questions per task name:
    for task_name, questions in task_name_to_questions.items():
        print(f"\x1b[34;1m{task_name}\x1b[0m")
        for i, question in enumerate(questions):
            print(f"\t{i+1}. {question}")
    # print the length of the QA pairs:
    tot = 0
    for q,a in data.items(): tot += len(a)
    print(f"{tot} Q/A pairs")
    with open("./data/FCDS/code_qa_submissions.json", "w") as f:
        json.dump(data, f, indent=4)
    print(f"UNIQ_IDs: {len(UNIQ_IDS)}")
    print(f"QA_IDs: {inst_ctr}")
    # for task_name, mds in task_name_to_other_mds.items():
    #     print(f"\x1b[34;1m{task_name}\x1b[0m")
    #     for i, md in enumerate(mds):
    #         print(f"\t{i+1}. {md}")