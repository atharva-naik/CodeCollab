import os
import csv
import sys
import json
import tarfile
from typing import *
from tqdm import tqdm
from jupyter_notebook_parser import JupyterNotebookParser

def save_code(codeUrl: str, writePath: str) -> str:
    import requests
    resp = requests.get(codeUrl)
    with open(writePath, "wb") as f:
        f.write(resp.content)

def create_dataset(metadata_dir: str, codes_path: str):
    data = []
    inst_ctr = 0
    for file in os.listdir(metadata_dir):
        if not file.endswith(".csv"): continue
        metadata_path = os.path.join(metadata_dir, file)
        course_id,_ = os.path.splitext(file)
        for row in tqdm(csv.DictReader(open(metadata_path)), desc=course_id):
            code_file = course_id + "_person_id_" + row["person_id"] + "_submission_id_" + row["submission_id"] + ".tar.gz"
            code_path = os.path.join(metadata_dir, "codes", code_file)
            code = load_code_from_tar_file(tar_path=code_path)
            inst_ctr += 1
            ID = inst_ctr
            rec = {
                "id": ID, "code": code, "person_id": row["person_id"], "submission_id": row["submission_id"],
                "module_name": row["module_name"], "module_slug": row["module_slug"], 
                "task_name": row["task_name"], "task_slug": row["task_slug"], 
                "event_id": row["event_id"], "event_time": row["event_time"],
                "module_date_to_activate": row["module_date_to_activate"],
                "module_date_to_submit": row["module_date_to_submit"],
                "score_obtained": row["score_obtained"],    
            }
            data.append(rec)
    with open("./data/FCDS/code_submissions_dataset.json", "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    return data

def download_code_data_from_metadata(path: str, course_id: str, save_dir: str="./data/FCDS/codes") -> List[dict]:
    for row in tqdm(csv.DictReader(open(path))):
        ext = json.loads(row["extensions"])
        fname = course_id + "_person_id_" + row["person_id"] + "_submission_id_" + row["submission_id"] + ".tar.gz"
        codeUrl = ext["codeUrl"]
        tar_save_path = os.path.join(save_dir, fname)
        save_code(codeUrl=codeUrl, writePath=tar_save_path)

def download_all_codes():
    for file in os.listdir("./data/FCDS"):
        if not file.endswith(".csv"): continue
        course_id,_ = os.path.splitext(file)
        path = os.path.join("./data/FCDS", file)
        download_code_data_from_metadata(
            path, course_id=course_id
        )

CTR = 0
TOT = 0
def load_code_from_tar_file(tar_path: str) -> List[dict]:
    global CTR
    global TOT
    TOT += 1
    try: tar = tarfile.open(tar_path)
    except tarfile.ReadError as e:
        CTR += 1
        print(f"TarFileReadError({CTR}):", e)
        return []
    content = None
    for member in tar.getmembers():
        if member.name.endswith(".ipynb"):
            content = tar.extractfile(member).read()
            break
    # if no .ipynb file is found.
    if content is None: return []
    temp_file_path = "DELETE_ME.ipynb"
    with open(temp_file_path, "wb") as f:
        f.write(content)
    try: parsed = JupyterNotebookParser(temp_file_path)
    except (UnicodeDecodeError, json.decoder.JSONDecodeError, ValueError) as e: 
        CTR += 1
        print(f"JupyterParseError({CTR}):", e)
        return []
    tar.close()
    code_cells = []
    for cell in parsed.get_all_cells():
        rec = {
            "cell_type": cell["cell_type"], 
            "metadata": cell["metadata"], 
            "id": cell.get("id")
        }
        if cell["cell_type"] == "markdown":
            rec["markdown"] = "\n".join(cell["source"])
        elif cell["cell_type"] == "code":
            rec["code"] = "\n".join(cell["source"])
        code_cells.append(rec)

    return code_cells
 
# main
if __name__ == "__main__":
    # code_cells = load_code_from_tar_file("./data/FCDS/codes/s23_computersystems_person_id_05ad4713423be680cb99ded021f17d88032dab8ad91afea30dfb584e7f528875_submission_id_049f0cf6fbb007be524fa0849f5d224ece111bb3799f79b9a4e1663bbd4310b7.tar.gz")
    create_dataset(
        metadata_dir="./data/FCDS/",
        codes_path="./data/FCDS/codes/"
    )
    print(f"{TOT} instances, error in {CTR} and {TOT-CTR} fixed instances")