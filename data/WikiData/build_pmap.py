# API call for Q code: https://www.wikidata.org/w/api.php?action=wbgetentities&format=json&ids=Q11660
# API call for P code: https://www.wikidata.org/w/api.php?action=wbgetproperties&format=json&ids=P1000

import json
import requests
from tqdm import tqdm

pmap = {}
pids = json.load(open("./data/WikiData/pids.json"))
for pid in tqdm(pids):
    url = f"https://www.wikidata.org/w/api.php?action=wbgetentities&format=json&ids={pid}"
    pjson = json.loads(requests.get(url).text)
    value = pjson["entities"][pid]["labels"]["en"]["value"]
    pmap[pid] = value
with open("./data/WikiData/pmap.json", "w") as f:
    json.dump(pmap, f, indent=4)