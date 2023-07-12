# file containing various node type classifiers for different KB sources (as needed).

import os
import json
from typing import *
from tqdm import tqdm

INIT_POINTS = {"decision tree": "M", "decision tree learning": "M", "machine learning":"C", "artificial intelligence": "C", "epistemology": "C", "cognitive psychology": "C", "psychology": "C", "cognitive linguistics": "C", "digital data": "D", "data": "D", "statistics": "S", "scientific theory": "C", "data structure": "C", "array data structure": "C", "norm": "C", "problem": "T", "informatics": "C", "Computational physiology": "C", "computational science": "C", "hypothesis testing": "S", "algorithm": "M", "ontology": "C", "design": "C", "error": "E", "error detection and correction": "C", "process": "C", "information retrieval": "C", "sampling bias": "C", "cognitive bias": "C", "engineering": "C", "research project": "C", "mathematical model": "M", "partial differential equation": "C", "nonlinear partial differential equation": "C", "conjecture": "C", "poisson bracket": "C", "graph": "C", "method": "M", "generative model": "M", "physics terminology": "C", "area of mathematics": "C", "theory": "C", "discrete mathematics": "C", "logic gate": "M", "polynomial root": "C", "lemma": "C", "computer science": "C", "computer network protocol": "C", "nonparametric regression": "M", "nonparametric statistics": "S", "statistical method": "S", "data scrubbing": "C", "data management": "C", "data extraction": "C", "data processing": "C", "type of test": "E", "modular exponentiation": "M", "integer factorization": "M", "bounded lattice": "C", "maximum": "C", "minimum": "C", "model-free reinforcement learning": "T", "physics": "C", "chemical analysis": "C", "LR parser": "M", "parsing": "T", "field of study": "C", "neuroscience": "C", "applied science": "C"}

class WikiDataNodeClassifier:
    def __init__(self):
        global INIT_POINTS
        self.known_points = INIT_POINTS
        self.curated_graph = json.load(open("./data/DS_TextBooks/unified_triples.json"))
        self.curated_nodes = {}
        for rec in self.curated_graph:
            self.curated_nodes[rec["sub"][0].lower()] = rec["sub"][1]
            self.curated_nodes[rec["obj"][0].lower()] = rec["obj"][1] 
        # print(self.curated_nodes)
    def override_clf_from_curated_info(self, node_name):
        return self.curated_nodes.get(node_name.lower())

    def __call__(self, node_name: str, adj_list: List[Tuple[str, str]]=[]) -> str:
        children = [x for x,_ in adj_list]
        overridden_class = self.override_clf_from_curated_info(node_name)
        if overridden_class is not None:
            # print(f"overrode class for {node_name}")
            self.known_points[node_name] = overridden_class
            return overridden_class
        if node_name.lower().endswith(" problem"): return "T"
        if node_name.lower().endswith("engineering"): return "C"
        if node_name.lower().endswith(" science"): return "Cs"
        if node_name.lower().endswith("statistics"): return "S"
        if node_name.lower().endswith(" distribution") or node_name.lower().endswith(" distributions"): # or " distribution " in node_name.lower(): 
            self.known_points[node_name] = "S"
            return "S"
        if "conjecture" in node_name.lower():
            self.known_points[node_name] = "C"
            return "C"
        if node_name.lower().endswith("theory"): "C"
        if node_name.lower().endswith("algorithm"):
            self.known_points[node_name] = "M"
            return "M"
        for name in self.known_points:
            if name in children+[node_name]:
                return self.known_points[name]
        match_conds = {f"subclass of {name}": class_ for name, class_ in self.known_points.items()}
        match_conds.update({f"instance of {name}": class_ for name, class_ in self.known_points.items()})
        # print(match_conds)
        for Q,P in adj_list:
            Q = Q.strip()
            P = P.strip()
            cond = f"{P} {Q}"
            if cond == "instance of concept": 
                self.known_points[node_name] = "C"
                return "C"
            elif cond == "has use statistics":
                self.known_points[node_name] = "S"
                return "S"
            elif cond == "subclass of concept": 
                self.known_points[node_name] = "C"
                return "C"
            elif cond == "instance of inequality": 
                self.known_points[node_name] = "C"
                return "C"
            elif cond == "instance of mathematical concept": 
                self.known_points[node_name] = "C"
                return "C"
            elif cond == "instance of algorithm": 
                self.known_points[node_name] = "M"
                return "M"
            elif cond == "subclass of algorithm": 
                self.known_points[node_name] = "M"
                return "M"
            class_ = match_conds.get(cond, None)
            if class_ is not None: 
                self.known_points[node_name] = class_
                return class_
        if "sampling" in node_name:
            self.known_points[node_name] = "S"
            return "S"
        if node_name.lower().endswith("logy"): 
            self.known_points[node_name] = "C" 
            return "C"

        return "U"

# main
if __name__ == "__main__":
    wiki_clf = WikiDataNodeClassifier()
    node_to_class_mapping = {}
    wikidata_graph = json.load(open("./data/WikiData/ds_qpq_graph_pruned.json"))
    for Q1, adj_list in tqdm(wikidata_graph.items()):
        adj_list = adj_list["E"]
        node_to_class_mapping[Q1] = wiki_clf(Q1, adj_list)
        for Q2, _ in adj_list:
            if Q2 in node_to_class_mapping: continue
            node_to_class_mapping[Q2] = wiki_clf(Q2, [])
    # successfully classified node percentage.
    unk_count = sum([int(v == 'U') for v in node_to_class_mapping.values()])
    tot = len(node_to_class_mapping)
    succ_count = tot - unk_count
    print(f"{succ_count}/{tot} and {100*(succ_count/tot):.2f}% are successfully classified ({unk_count} are still unknown)")
    with open("./data/DS_KB/wikidata_pred_node_classes.json", "w") as f:
        json.dump(node_to_class_mapping, f, indent=4)