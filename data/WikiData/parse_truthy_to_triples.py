# parse truthy.nt (it is a plain text file) file into Q,P,Q triples.

import re 
import json
from tqdm import tqdm

truthy_path = "/data/NO-BACKUP/arnaik/WikiData/latest-truthy.nt"
regex_pattern = "<http://www\.wikidata\.org/entity/Q\d+> <http://www.wikidata.org/prop/direct/P\d+> <http://www\.wikidata\.org/entity/Q\d+> \."
# re.match(regex_pattern, "<http://www.wikidata.org/entity/Q31> <http://www.wikidata.org/prop/direct/P1344> <http://www.wikidata.org/entity/Q1088364> .")
triples = []
g = open("./data/WikiData/QPQ_triples.jsonl", "w")
with open(truthy_path, "r") as f:
    for line in tqdm(f):
        line = line.strip()
        if re.match(regex_pattern, line) is not None:
            entities = re.findall("Q\d+", line)
            try: assert len(entities) == 2
            except AssertionError as e:
                print("EntityMismatchError:", e)
            prop = re.findall("P\d+", line)
            try: assert len(prop) == 1
            except AssertionError as e:
                print("PropMismatchError:", e)
            triple = (entities[0], prop[0], entities[1])
            triples.append(triple)
            g.write(json.dumps(triple)+"\n")