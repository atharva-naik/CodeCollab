import sys
import json

def recursively_check_keys(kb: dict):
    """recursively iterate over KB and verify whether 
    each key parses into key name and ontology type."""
    for k,v in kb.items():
        # check the key
        msg = f"Key \x1b[31;1m`{k}`\x1b[0m doesn't parse"
        if k != "STEPS":
            assert len(k.split("::")) == 2, msg
        if isinstance(v, dict):
            recursively_check_keys(v)

# main
if __name__ == "__main__":
    path = "./data/DS_TextBooks/Learning Kernel Classifiers Theory and Algorithms.json" # sys.argv[1]
    kb_json = json.load(open(path))
    recursively_check_keys(kb_json)
    print("\x1b[32mall tests passed!\x1b[0m")