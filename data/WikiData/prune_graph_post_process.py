# post processing to prune KB
import os
import json
from typing import *

FILTER_LIST = ["Walgreens", "Bihar school meal poisoning incident", "Scientology", "scientology in Norway", "lead paragraph", "abstract", "emissions", "pollution", "human tooth", "dental notation", "article"]
# class for post processing the graph.
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
            if self.is_medical_term(Q1, adj_list) or Q1 in FILTER_LIST or self.is_animal_term(Q1) or self.is_food_term(Q1) or self.is_religion(Q1) or self.is_legal_term(Q1, adj_list) or Q1.startswith("reserved for private use") or self.is_date_term(Q1): 
                pruned_out[Q1] = graph[Q1] 
                continue
            skip_this = False
            subG = {"E": [], "c": graph[Q1]["c"]}
            for Q2, P in adj_list["E"]:
                if Q2 in FILTER_LIST: continue
                condition = f"{P.strip()} {Q2.strip()}"
                # skip a node if it triggers a filter condition
                if not(self.is_animal_term(Q2) or self.is_medical_term(Q2) or self.is_legal_term(Q2) or self.is_food_term(Q2) or self.is_religion(Q2) or Q2.startswith("reserved for private use") or self.is_date_term(Q2)): 
                    subG["E"].append((Q2, P))
                if condition in self.filter_conditions: skip_this = True; break
            if len(subG["E"]) == 0: 
                pruned_out[Q1] = graph[Q1]
                continue
            if skip_this: pruned_out[Q1] = graph[Q1]; continue
            pruned_graph[Q1] = subG
        # with open("./data/WikiData/DELETE_diseases.json", "w") as f:
        #     print(f"{len(disease_nodes)} disease related nodes")
        #     json.dump(disease_nodes, f, indent=4)
        return pruned_graph, pruned_out

    def is_date_term(self, node_name: str):
        if "january" in node_name.lower(): return True
        if "february" in node_name.lower(): return True
        if "march" in node_name.lower(): return True
        if "april" in node_name.lower(): return True
        if "may" in node_name.lower(): return True
        if "june" in node_name.lower(): return True
        if "july" in node_name.lower(): return True
        if "august" in node_name.lower(): return True
        if "september" in node_name.lower(): return True
        if "october" in node_name.lower(): return True
        if "november" in node_name.lower(): return True
        if "december" in node_name.lower(): return True
        if "monday" in node_name.lower(): return True
        if "tuesday" in node_name.lower(): return True
        if "wednesday" in node_name.lower(): return True
        if "thursday" in node_name.lower(): return True
        if "friday" in node_name.lower(): return True
        if "saturday" in node_name.lower(): return True
        if "sunday" in node_name.lower(): return True
        if "years" in node_name.lower(): return True
        return False

    def is_religion(self, node_name: str):
        if "battle" in node_name.lower(): return True
        if "christian" in node_name.lower(): return True
        if "buddhism" in node_name.lower(): return True
        if "religio" in node_name.lower(): return True
        if "muslim" in node_name.lower(): return True
        if "islam" in node_name.lower(): return True
        if "hindu" in node_name.lower(): return True
        return False

    def is_food_term(self, node_name: str):
        return "food" in node_name.lower()

    def is_animal_term(self, node_name: str):
        if "ecolog" in node_name.lower(): return True
        if "dinosaur" in node_name.lower(): return True
        if "agricultur" in node_name.lower(): return True
        return False

    def is_legal_term(self, node_name: str, adj_list: Dict[str, Union[int, List[Tuple[str, str]]]]={"E": []}):
        if "legal" in node_name.lower(): return True
        if " bank" in node_name.lower(): return True
        # if " festival" in node_name.lower(): return True
        for Q2, P in adj_list["E"]: 
            if "legal" in Q2.lower(): return True
        return False

    def is_medical_term(self, node_name: str, adj_list: Dict[str, Union[int, List[Tuple[str, str]]]]={"E": []}):
        if "skin " in node_name.lower(): return True
        if "bone" in node_name.lower(): return True
        if "foot" in node_name.lower(): return True
        if "disease" in node_name.lower(): return True
        if "rbcl" in node_name.lower(): return True
        if "syndrome" in node_name.lower(): return True
        if "(deoxy)" in node_name.lower(): return True
        if "lipoma" in node_name.lower(): return True
        if "carcinoma" in node_name.lower(): return True
        if "cancer" in node_name.lower(): return True
        if "melanoma" in node_name.lower(): return True
        if "lymphoma" in node_name.lower(): return True
        if "hyperplas" in node_name.lower(): return True
        # not medical term (related to olypmics)
        if "olympic" in node_name.lower(): return True
        if node_name.lower().endswith("[cytosol]"): return True
        if node_name.lower().endswith("[nucleoplasm]"): return True
        if node_name.lower().endswith("[plasma membrane]"): return True
        if node_name.lower().endswith("[extracellular region]"): return True
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
        {"Q": "Request for Comments", "P": "instance of"},
        {"Q": "agriculture", "P": "instance of"},
        {"Q": "agriculture", "P": "subclass of"},
        {"Q": "agriculture by country or territory", "P": "instance of"},
        {"Q": "agriculture and forestry", "P": "subclass of"},
        {"Q": "evolution", "P": "instance of"},
        {"Q": "evolution", "P": "subclass of"},
        {"Q": "anthropology", "P": "instance of"},
        {"Q": "anthropology", "P": "subclass of"},
        {"Q": "S&P 500", "P": "part of"},
        {"Q": "mental state", "P": "instance of"},
        {"Q": "mental state", "P": "subclass of"},
        {"Q": "ecology", "P": "instance of"},
        {"Q": "ecology", "P": "subclass of"},
        {"Q": "woodworking", "P": "has use"},
        {"Q": "woodworking", "P": "part of"},
        {"Q": "woodworking", "P": "subclass of"},
        {"Q": "review", "P": "instance of"},
        {"Q": "Scientology", "P": "instance of"},
        {"Q": "Scientology", "P": "subclass of"},
        {"Q": "Scientology", "P": "said to be the same as"},
        {"Q": "abstract", "P": "instance of"},
        {"Q": "strike", "P": "instance of"},
        {"Q": "strike", "P": "subclass of"},
        {"Q": "medicinal plant", "P": "has use"},
        {"Q": "nursing", "P": "instance of"},
        {"Q": "permanent mission", "P": "instance of"},
        {"Q": "incident", "P": "instance of"},
        {"Q": "company", "P": "instance of"},
        {"Q": "company", "P": "subclass of"},
        {"Q": "business", "P": "instance of"},
        {"Q": "business", "P": "subclass of"},
        {"Q": "sexual assault", "P": "instance of"},
        {"Q": "sexual assault", "P": "subclass of"},
        {"Q": "patronal festival", "P": "instance of"},
        {"Q": "rivalry", "P": "subclass of"},
        {"Q": "symptom or sign", "P": "instance of"},
        {"Q": "military term", "P": "instance of"},
        {"Q": "military term", "P": "subclass of"},
        {"Q": "policy", "P": "instance of"},
        {"Q": "policy", "P": "subclass of"},
        {"Q": "social policy", "P": "part of"},
        {"Q": "social policy", "P": "instance of"},
        {"Q": "social policy", "P": "subclass of"},
        {"Q": "politics", "P": "instance of"},
        {"Q": "politics", "P": "subclass of"},
        {"Q": "politics", "P": "part of"},
        {"Q": "political science", "P": "instance of"},
        {"Q": "political science", "P": "subclass of"},
    )
    # print(graph_proc)
    print(len(qpq_graph))
    pruned_graph, pruned_out = graph_proc.prune(qpq_graph)
    assert (len(pruned_out) + len(pruned_graph)) == len(qpq_graph)
    print(len(pruned_graph))
    print(len(pruned_out))
    with open("./data/WikiData/ds_qpq_graph_pruned.json", "w") as f:
        json.dump(pruned_graph, f, indent=4, ensure_ascii=False)
    # with open("DELETE.txt", "w") as f:
    #     f.write("\n".join([k for k in pruned_out]))