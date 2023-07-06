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
    
    task_name_to_questions = defaultdict(lambda: {})
    task_name_to_other_mds = defaultdict(lambda: {})
    for rec in fcds_data:
        for cell in rec["code"]:
            if cell["cell_type"] == "markdown":
                # filter out question like markdowns
                if cell["markdown"].strip("#").strip().lower().startswith("question"):
                    # first line of the markdown
                    task_name_to_questions[rec["task_name"]][
                        cell["markdown"].split("\n")[0].strip()
                    ] = None
                else:
                    task_name_to_other_mds[rec["task_name"]][
                        cell["markdown"].split("\n")[0].strip()
                    ] = None                    
    
    # analyze list of questions per task name:
    for task_name, questions in task_name_to_questions.items():
        print(f"\x1b[34;1m{task_name}\x1b[0m")
        for i, question in enumerate(questions):
            print(f"\t{i+1}. {question}")

    # for task_name, mds in task_name_to_other_mds.items():
    #     print(f"\x1b[34;1m{task_name}\x1b[0m")
    #     for i, md in enumerate(mds):
    #         print(f"\t{i+1}. {md}")