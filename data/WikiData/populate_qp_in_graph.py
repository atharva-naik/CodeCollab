# populate Q/P codes with their english labels.

import os 
import json

qmap = json.load(open("./data/WikiData/qmap1.json"))
pmap = json.load(open("./data/WikiData/pmap.json"))

# main
if __name__ == "__main__":
    graph = json.load(open("./data/WikiData/ds_qpq_graph.json"))
    pop_graph = {} # populated graph
    for q1, adj_list in graph.items():
        pop_adj_list = []
        if q1.startswith("Q") and len(q1) > 1 and q1[1].isdigit(): 
            q1 = qmap[q1]
        for q2, p in adj_list["E"]:
            if q2.startswith("Q") and len(q2) > 1 and q2[1].isdigit(): 
                q2 = qmap[q2]
            if p.startswith("P"): p = pmap[p]
            pop_adj_list.append((q2,p))
        pop_graph[q1] = {"c": adj_list['c'], "E": pop_adj_list}
    # print(json.dumps(pop_graph, indent=4)[:200])
    with open("./data/WikiData/ds_qpq_graph.json", "w") as f:
        json.dump(pop_graph, f, indent=4)