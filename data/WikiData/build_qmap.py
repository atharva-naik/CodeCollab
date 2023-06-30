# map entity codes (begin with Q to strings/english names).
import json
import requests
from tqdm import tqdm

# main 
if __name__ == "__main__":
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