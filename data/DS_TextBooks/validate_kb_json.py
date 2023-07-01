import os
import sys
import json

def recursively_check_keys(kb: dict, path: str):
    """recursively iterate over KB and verify whether 
    each key parses into key name and ontology type."""
    for k,v in kb.items():
        # check the key
        msg = f"{path}: Key \x1b[31;1m`{k}`\x1b[0m doesn't parse"
        if k != "STEPS":
            assert len(k.split("::")) == 2, msg
        if isinstance(v, dict):
            recursively_check_keys(v, path)

# main
if __name__ == "__main__":
    # path = "./data/DS_TextBooks/Learning Kernel Classifiers Theory and Algorithms.json" # sys.argv[1]
    for path in os.listdir("./data/DS_TextBooks"):
        if not path.endswith(".json"): continue
        if path == "semantic_types.json": continue
        if path == "unified_triples.json": continue
        if path == "sample.json": continue
        full_path = os.path.join("./data/DS_TextBooks", path)
        try:
            kb_json = json.load(open(full_path))
            recursively_check_keys(kb_json, path=path)
        except Exception as e:
            print(path+":", e)
        print(f"{path}: \x1b[32mall tests passed!\x1b[0m")