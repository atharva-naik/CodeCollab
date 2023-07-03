# create a graph structure of neighbors and relation types from triples.

import os
import json
from tqdm import tqdm
from collections import defaultdict

# (outflow, inflow) for each Q code
q_counts = defaultdict(lambda: [0,0])
# how many times is a P code a relationship 
p_counts = defaultdict(lambda: 0) 
triples_graph = defaultdict(lambda: {"c": 0, "E": []})
qmap = json.load(open("./data/WikiData/qmap1.json"))
pmap = json.load(open("./data/WikiData/pmap.json"))
# list of relevant nodes (for data science, CS, ML, etc.)
NODE_COSTS = {
    "Q11660": 0, # artificial intelligence
    "Q2539": 0, # machine learning
    "Q2374463": 0, # data science
    "Q12483": 0, # statistics
    "Q35308049": 0, # statistical data
    "Q208042": 0, # regression analysis
    "Q12718609": 0, # statistical method
    "Q1988917": 0, # data analysis
    # "Q485396": "", # analytics 
    "Q1149776": 0, # data management
    # "Q11661": "", # information technology
    # "Q77293133": "", # data analyst
    "Q42848": 0, # data
    "Q15088675": 0, # data curation
    # "Q190087": "", # data type
    # "Q5227257": "", # data classification (data management)
    # "Q494823": "", # data format
    # "Q112598603": "", # data professional
    # "Q188889": "", # code
    "Q1417149": 0, # rule-based system
    "Q59154708": 0, # data export
    "Q1783551": 0, # data conversion
    "Q6661985": 0, # data processing
    # "Q750843": "", # information processing
    "Q107491038": 0, # data processor
    "Q8366": 0, # algorithm
    "Q5157286": 0, # computational complexity
    "Q1296251": 0, # algorithmic efficiency
    "Q59154760": 0, # data import
    # "Q235557": "", # file format
    # "Q65757353": "", # transformation
    "Q7595718": 0, # algorithmic stability
    "Q1412694": 0, # knowledge-based system
    # "Q217602": "", # analysis
}

PROPS = [
    "P361", # part of
    # "P910", # topic's main category
    "P279", # subclass of
    # "P1424", # topic's main template 
    # "P1482", # Stack Exchange Tag
    "P527", # has part(s)
    "P1889", # different from
    # "P373", # Commons Category.
    # "P3095", # practiced by
    "P31", # instance of
    "P1552", # has quality
    # "P2184", # history of topic
    # "P6541", # stack exchange site url
    "P366", # has use
    "P737", # influenced by
    "P797", # significant event
    "P3712", # has goal
    "P155", # follows
    "P2578", # is the study of
    "P2283", # uses
    "P1382", # partially coincident with
    "P460", # said to be the same as
    "P2737", # union of
]
BLOCK_LIST = [
    "Q11862829", # significant event
    "Q268592", # industry 
    "Q120208", # emerging technology
    "Q112057532", # type of technology
    "Q14623823", # artificiality 
    "Q11024", # communication
    "Q194253", # Evangelicalism
    "Q373069", # evangelism
    "Q1074953", # microscopy
    "Q2695280", # technique
    "Q11016", # technology
    "Q1078438",
    "Q3707847", # bookkeeping
    "Q1132131", # loyalty
    "Q11862829", # academic discipline
    "Q374814", # certification
    "Q43015", # finance
    "Q2465832", # branch of science
    "Q336", # science
    "Q395", # mathematics
    "Q4671286", # academic major
    "Q816264", # formal science
    "Q205663", # process
    "Q3353185", # business intelligence
    "Q1128340", # subject heading
    "Q11028", # information
    "Q41689629", # procedure
    "Q386724", # work 
    "Q212469", # souvenir
    # "",
]
if __name__ == "__main__":
    with open("./data/WikiData/qpq_triples.jsonl") as f:
        pbar = tqdm(f, desc="")
        all_ctr = 0
        edge_ctr = 0 # count number of edges
        for line in pbar:
            all_ctr += 1
            if all_ctr % 10000000 == 0:
                print(f"N: {len(q_counts)} E: {edge_ctr}")

            Q1, P, Q2 = json.loads(line.strip())            
            if Q1 not in NODE_COSTS and Q2 not in NODE_COSTS: continue
            if Q1 in BLOCK_LIST or Q2 in BLOCK_LIST: continue

            if P not in PROPS: continue
            if Q1 not in NODE_COSTS:
                NODE_COSTS[Q1] = NODE_COSTS[Q2] + 1
            if Q2 not in NODE_COSTS:
                NODE_COSTS[Q2] = NODE_COSTS[Q1] + 1
            if max(NODE_COSTS[Q1], NODE_COSTS[Q2]) > 5: continue
            cost =  NODE_COSTS[Q1] # node cost.

            q_counts[Q1][0] += 1 
            q_counts[Q2][1] += 1
            p_counts[P] += 1

            Q1 = qmap.get(Q1, Q1)
            Q2 = qmap.get(Q2, Q2)
            P = pmap[P]

            triples_graph[Q1]["c"] = cost
            triples_graph[Q1]["E"].append((Q2,P))
            edge_ctr += 1
            
print(f"NODE_COSTS: {len(NODE_COSTS)}")
with open("./data/WikiData/qids.json", "w") as f:
    print(f"qids: {len(q_counts)}")
    json.dump(dict(q_counts), f, indent=4)
with open("./data/WikiData/pids.json", "w") as f:
    print(f"pids: {len(p_counts)}")
    json.dump(dict(p_counts), f, indent=4)
with open("./data/WikiData/ds_qpq_graph.json", "w") as f:
    json.dump(dict(triples_graph), f, indent=4)