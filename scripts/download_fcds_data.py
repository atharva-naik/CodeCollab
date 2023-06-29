import os
import csv
import sys
import json
from typing import *
from tqdm import tqdm

def save_code(codeUrl: str, writePath: str) -> str:
    import requests
    resp = requests.get(codeUrl)
    with open(writePath, "wb") as f:
        f.write(resp.content)

def create_dataset(metadata_path: str, codes_path: str):
    for row in csv.DictReader(open(path)):
        ext = json.loads(row["extensions"])
        codeUrl = ext["codeUrl"]
        code = ""
        rec = {
            "module_slug": row["module_slug"], 
            "task_slug": row["task_slug"], 
            "score_obtained": row["score_obtained"], 
            "codeURL": codeUrl,
            "code": code,
        }

def download_code_data_from_metadata(path: str, course_id: str, save_dir: str="./data/FCDS/codes") -> List[dict]:
    for row in tqdm(csv.DictReader(open(path))):
        ext = json.loads(row["extensions"])
        fname = course_id + "_person_id_" + row["person_id"] + "_submission_id_" + row["submission_id"] + ".tar.gz"
        codeUrl = ext["codeUrl"]
        tar_save_path = os.path.join(save_dir, fname)
        save_code(codeUrl=codeUrl, writePath=tar_save_path)

# main
if __name__ == "__main__":
    for file in os.listdir("./data/FCDS"):
        if not file.endswith(".csv"): continue
        course_id,_ = os.path.splitext(file)
        path = os.path.join("./data/FCDS", file)
        data = download_code_data_from_metadata(
            path, course_id=course_id
        )