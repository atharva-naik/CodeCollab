import json
from typing import *
from tqdm import tqdm
from datautils import read_jsonl
from collections import defaultdict

def load_cell_to_nb_map(path: str="./data/juice-dataset/traindedup.jsonl") -> Dict[str, List[Dict[str, int]]]:
    mapping = defaultdict(lambda:[]) 
    f = open(path, "r")
    nbid = 0
    for line in tqdm(f, desc="loading cell to NB map"):
        cellid = 0
        line = line.strip()
        if line == "": continue
        nb = json.loads(line)
        for cell in nb["context"][::-1]: 
            if cell["cell_type"] != "markdown":
                mapping[cell["code"]].append({
                    "nbid": nbid,
                    "cellid": cellid,
                })
            cellid += 1
        mapping[nb["code"]].append({
            "nbid": nbid,
            "cellid": cellid,
        })
        nbid += 1

    return dict(mapping)

# main
if __name__ == "__main__":
    code_KB = json.load(open("./JuICe_train_code_KB.json"))
    codes = list(code_KB.keys())
    mapping = load_cell_to_nb_map()
    cell_to_ids_map = {}
    missing_ctr = 0
    for id, code in tqdm(enumerate(codes), total=len(codes)):
        try: cell_to_ids_map[code] = mapping[code]
        except KeyError: 
            cell_to_ids_map[code] = []
            missing_ctr += 1
            # print(f"missing cell at: {id}")
    cell_to_ids_map = dict(cell_to_ids_map)
    print(f"missing NB IDs for {missing_ctr} cells")
    # save code to JuICe NB and cell ids.
    with open("./JuICe_train_code_to_nbids.json", "w") as f:
        json.dump(cell_to_ids_map, f, indent=4)