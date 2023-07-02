# map entity codes (begin with Q to strings/english names).
import json
import requests
from tqdm import tqdm

def initial_build():
    lexemes =  json.load(open("/data/NO-BACKUP/arnaik/WikiData/wikidata-20230628-lexemes.json"))
    en_lexemes = [l for l in tqdm(lexemes) if "en" in l["lemmas"]]
    qmap = {}
    for l in tqdm(en_lexemes):
        # skip non nouns.
        if l["lexicalCategory"] != "Q1084": continue
        value = l["lemmas"]["en"]["value"]
        q_json = json.loads(requests.get(f"https://en.wikipedia.org/w/api.php?action=query&prop=pageprops&titles={value.replace(' ','_')}&format=json").text)
        for qcode in list(q_json['query']['pages'].values()):
            try: key = qcode['pageprops']['wikibase_item']
            except KeyError as e: continue
            qmap[key] = value
    with open("./data/WikiData/qmap.json", "w") as f:
        json.dump(qmap, f, indent=4)

def update_build():
    curr_qmap = json.load(open("./data/WikiData/qmap.json"))
    print(len(curr_qmap))
    qids = json.load(open("./data/WikiData/qids.json"))
    for qid in tqdm(qids):
        if qid not in curr_qmap:
            qid = qid.strip()
            if not qid.startswith("Q"): continue
            qjson = json.loads(requests.get(f"https://www.wikidata.org/w/api.php?action=wbgetentities&format=json&ids={qid}&props=labels&languages=en").text)
            try: 
                try: 
                    labels = qjson["entities"][qid]["labels"]
                    if len(labels) == 0: value = "NO_LABEL"
                    else: value = labels["en"]["value"]
                except KeyError: value = "MISSING"
                #["en"]["value"]
            except KeyError:
                print(qjson)
                exit()
            curr_qmap[qid] = value
    print(len(curr_qmap))
    with open("./data/WikiData/qmap1.json", "w") as f:
        json.dump(curr_qmap, f, indent=4)

# main 
if __name__ == "__main__":
    update_build()