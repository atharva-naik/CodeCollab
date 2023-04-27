# gather plan operators.
import os
import ast
import json
from typing import *
from tqdm import tqdm
from datautils import read_jsonl
from collections import defaultdict
from datautils.code_cell_analysis import process_nb_cell
from scripts.sample_data_for_annot_situated_steps import process_step

def assemble_plan_paths(id: int, knn: Dict[str, Union[int, List[str]]], 
                        codes_and_paths: List[Tuple[str, List[str]]]) -> List[str]:
    operator_names, all_codes = {}, set()
    for path in codes_and_paths[id][1]:
        step = path.split("->")[-1] 
        # gather potential operator names:
        operator_names[path] = process_step(step)
    # collect all codes
    all_codes.add(codes_and_paths[id][0])
    # iterate over the retrieval hits:
    for matched_id in knn[id]["matched"]:
        matched_id = int(matched_id)
        # collect all codes
        all_codes.add(codes_and_paths[matched_id][0])
        for path in codes_and_paths[matched_id][1]:
            step = path.split("->")[-1]
            # gather potential operator names:
            operator_names[path] = process_step(step)

    return {"operator_names": operator_names, 'codes': sorted(list(all_codes), key=lambda x: len(x), reverse=True)}

def invert_code_KB(code_KB: Dict[str, List[str]]) -> Dict[str, List[str]]:
    step_to_codes_KB = defaultdict(lambda:set())
    for code, path_list in code_KB.items():
        for path in path_list:
            step = path.split("->")[-1].strip()
            step_to_codes_KB[step].add(code)
    temp = {}
    for k,v in step_to_codes_KB.items(): temp[k] = list(v)
    step_to_codes_KB = temp

    return step_to_codes_KB

def get_codes_from_step_seqs(steps: List[str], step_to_codes_KB) -> Dict[str, str]:
    steps_to_codes_result = defaultdict(lambda:[])
    for step in steps:
        for code in step_to_codes_KB[step.strip()]: 
            code = ast.unparse(ast.parse(process_nb_cell(code)))
            steps_to_codes_result[step].append(code)
    result = {}
    for step in steps:
        result[step] = "\n".join(steps_to_codes_result[step])

    return result

# main
if __name__ == "__main__":
    knn = read_jsonl("./JuICe_train_code_knn.jsonl")
    code_KB = json.load(open("./JuICe_train_code_KB.json"))
    codes_and_paths = list(code_KB.items())

    assemble_plan_paths(0, knn, codes_and_paths)
    steps_to_codes_KB = invert_code_KB(code_KB)
    get_codes_from_step_seqs([], steps_to_codes_KB)