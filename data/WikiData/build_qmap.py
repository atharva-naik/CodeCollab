# map entity codes (begin with Q to strings/english names).
import json
from tqdm import tqdm

# main 
if __name__ == "__main__":
    lexemes =  json.load(open("/data/NO-BACKUP/arnaik/WikiData/wikidata-20230628-lexemes.json"))
    en_lexemes = [l for l in tqdm(lexemes) if "en" in l["lemmas"]]
    qmap = {}
    for l in en_lexemes:
        key = l["senses"][0]["claims"]["P5137"][0]["mainsnak"]["datavalue"]["value"]["id"]
        value = l["senses"][0]["en"]["glosses"]["value"]
        qmap[key] = value
    with open("./data/WikiData/qmap.json", "w") as f:
        json.dump(qmap, f, indent=4)