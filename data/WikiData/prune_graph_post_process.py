# post processing to prune KB
import os
import json
from typing import *

class GraphPostProcessor:
    def __init__(self, filter_conditions: List[Dict[str, str]]=[]):
        self.filter_conditions = filter_conditions
    
    def add_filter(self, *filters: Dict[str, str]):
        for filter_ in filters:
            assert "P" in filter_
            assert "Q" in filter_
            self.filter_conditions.append(
                f"{filter_['P'].strip()} {filter_['Q'].strip()}"
            )

    def __repr__(self):
        return f"GraphPostProcessor({', '.join(self.filter_conditions)})"

    def prune(self, graph: Dict[str, List[Tuple[str, str]]]):
        pruned_graph = {}
        pruned_out = {}
        for Q1, adj_list in graph.items():
            if self.is_disease_node(Q1, adj_list): 
                pruned_out[Q1] = graph[Q1] 
                continue
            skip_this = False
            for Q2, P in adj_list["E"]:
                condition = f"{P.strip()} {Q2.strip()}"
                # skip a node if it triggers a filter condition
                if condition in self.filter_conditions: skip_this = True; break
            if skip_this: pruned_out[Q1] = graph[Q1]; continue
            pruned_graph[Q1] = graph[Q1]
        # with open("./data/WikiData/DELETE_diseases.json", "w") as f:
        #     print(f"{len(disease_nodes)} disease related nodes")
        #     json.dump(disease_nodes, f, indent=4)
        return pruned_graph, pruned_out

    def is_disease_node(self, node_name, adj_list):
        if "disease" in node_name.lower(): return True
        for Q2, P in adj_list["E"]:
            if "disease" in Q2.lower(): return True
        return False

# main
if __name__ == "__main__":
    qpq_graph = json.load(open("./data/WikiData/ds_qpq_graph.json"))
    graph_proc = GraphPostProcessor()
    graph_proc.add_filter(
        {"P": "instance of", "Q": "occurrence"},
        {"P": "instance of", "Q": "disease"},
        {"P": "subclass of", "Q": "anthropogenic hazard"},
        {"Q": "waste management process", "P": "subclass of"},
        {"Q": "class of disease", "P": "instance of"},
        {"Q": "viral infectious disease", "P": "subclass of"},
        {"Q": "stomach disease", "P": "subclass of"},
        {"Q": "nail disease", "P": "subclass of"},
        {"Q": "viral infectious disease", "P": "instance of"},
        {"Q": "teeth hard tissue disease", "P": "subclass of"},
        {"Q": "heavy chain disease", "P": "subclass of"},
        {"Q": "bone disease", "P": "subclass of"},
        {"Q": "ecology", "P": "subclass of"},
        {"Q": "urinary system disease", "P": "subclass of"},
        {"Q": "autoimmune disease of gastrointestinal tract", "P": "subclass of"},
        {"Q": "heart valve disease", "P": "subclass of"},
        {"Q": "adrenal gland disease", "P": "subclass of"},
        {"Q": "pancreas disease", "P": "subclass of"},
        {"Q": "damage", "P": "subclass of"},
        {"Q": "granuloma annulare", "P": "subclass of"},
        {"Q": "disease ecology", "P": "subclass of"},
        {"Q": "disease", "P": "subclass of"},
        {"Q": "public company", "P": "instance of"},
        {"Q": "financial institution", "P": "instance of"},
        {"Q": "financial institution", "P": "subclass of"},
        {"Q": "securities law", "P": "subclass of"},
        {"Q": "area of law", "P": "part of"},
        {"Q": "area of law", "P": "subclass of"},
        {"Q": "area of law", "P": "instance of"},
        {"Q": "area of law in France", "P": "instance of"},
        {"Q": "competition law", "P": "subclass of"},
        # {"Q": "study of history", "P": "has goal"},
        {"Q": "study of history", "P": "subclass of"},
        {"Q": "cultural history", "P": "subclass of"},
        {"Q": "archaeology", "P": "instance of"},
        {"Q": "economic activity", "P": "instance of"},
        {"Q": "economic activity", "P": "subclass of"},
        {"Q": "film", "P": "instance of"},
        {"Q": "business", "P": "instance of"},
        {"Q": "company", "P": "instance of"},
        {"Q": "biological process", "P": "instance of"},
        {"Q": "family tree", "P": "instance of"},
        {"Q": "infectious disease", "P": "instance of"},
        {"Q": "infectious disease", "P": "subclass of"},
        {"Q": "occupation", "P": "instance of"},
        {"Q": "article", "P": "instance of"},
        {"Q": "article", "P": "subclass of"},
        {"Q": "media franchise", "P": "instance of"},
        {"Q": "food", "P": "instance of"},
        {"Q": "food", "P": "subclass of"},
        {"Q": "meeting", "P": "instance of"},
        {"Q": "terrorist attack", "P": "instance of"},
        {"Q": "acid", "P": "instance of"},
        {"Q": "trial", "P": "instance of"},
        {"Q": "trial", "P": "subclass of"},
        {"Q": "folklore", "P": "instance of"},
        {"Q": "folklore", "P": "subclass of"},
        {"Q": "folklore", "P": "part of"},
        {"Q": "rare disease", "P": "instance of"},
        {"Q": "rare disease", "P": "subclass of"},
        {"Q": "Christianity", "P": "instance of"},
        {"Q": "Christianity", "P": "subclass of"},
        {"Q": "religious conversion", "P": "subclass of"},
        {"Q": "Christian denomination", "P": "subclass of"},
        {"Q": "Christian denomination", "P": "instance of"},
        {"Q": "Christianity of an area", "P": "instance of"},
        {"Q": "Christianity of an area", "P": "subclass of"},
        {"Q": "Christianity in Angola", "P": "subclass of"},
        {"Q": "Christianity in Nigeria", "P": "subclass of"},
        {"Q": "Christianity in Mongolia", "P": "subclass of"},
        {"Q": "Protestantism", "P": "subclass of"},
        {"Q": "Christianity in the 13th century", "P": "follows"},
        {"Q": "Christianity in the 14th century", "P": "follows"},
        {"Q": "creative work", "P": "instance of"},
        {"Q": "RNA", "P": "subclass of"},
        {"Q": "chemical substance", "P": "instance of"},
        {"Q": "chemical substance", "P": "subclass of"},
        {"Q": "chemical reaction", "P": "instance of"},
        {"Q": "chemical reaction", "P": "subclass of"},
        {"Q": "Islam of an area", "P": "instance of"},
        {"Q": "flood", "P": "instance of"},
        {"Q": "flood", "P": "subclass of"},
        {"Q": "Christianity", "P": "part of"},
        {"Q": "glass", "P": "instance of"},
        {"Q": "glass", "P": "subclass of"},
        {"Q": "avalanche", "P": "instance of"},
        {"Q": "avalanche", "P": "subclass of"},
        {"Q": "sports technique", "P": "instance of"},
        {"Q": "sports technique", "P": "subclass of"},
    )
    print(graph_proc)
    print(len(qpq_graph))
    pruned_graph, pruned_out = graph_proc.prune(qpq_graph)
    assert (len(pruned_out) + len(pruned_graph)) == len(qpq_graph)
    print(len(pruned_graph))
    print(len(pruned_out))
    with open("./data/WikiData/ds_qpq_graph_pruned.json", "w") as f:
        json.dump(pruned_graph, f, indent=4, ensure_ascii=False)
    # with open("DELETE.txt", "w") as f:
    #     f.write("\n".join([k for k in pruned_out]))