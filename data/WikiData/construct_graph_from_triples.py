# create a graph structure of neighbors and relation types from triples.

import os
import json
from tqdm import tqdm
from collections import defaultdict

# (inflow, outflow) for each Q code
q_counts = defaultdict(lambda: [0,0])
# how many times is a P code a relationship 
p_counts = defaultdict(lambda: 0) 
triples_graph = defaultdict(lambda: [])

if __name__ == "__main__":
    with open("./data/WikiData/qpq_triples.jsonl") as f:
        for line in tqdm(f):
            Q1, P, Q2 = json.loads(line.strip())
            q_counts[Q1][0] += 1 
            q_counts[Q2][1] += 1
            p_counts[P] += 1
            triples_graph[Q1].append((Q2,P))

with open("./data/WikiData/qids.json", "w") as f:
    json.dump(dict(q_counts), f, indent=4)
with open("./data/WikiData/pids.json", "w") as f:
    json.dump(dict(p_counts), f, indent=4)
with open("./data/WikiData/qpq_triples_graph.json", "w") as f:
    json.dump(dict(triples_graph), f, indent=4)