import json
import spacy
import random
import numpy as np
from typing import *   
from tqdm import tqdm
from spacy.matcher import Matcher
from spacy.util import filter_spans
from collections import defaultdict

class NaiveEventGraph:
    """A Naive Event Graph representation based on 
    directed event sequence occurences frequencies"""
    def __init__(self, path: Union[str, None]=None):
        self.adj = defaultdict(lambda: defaultdict(lambda: 0))
        if path: self.read(path)

    def serialize_node(self, cell_type: str, event: str):
        return json.dumps({"cell_type": cell_type, "event": event})

    def deserialize_node(self, string: str):
        node_dict = json.loads(string)
        return (node_dict["cell_type"], node_dict["event"])
        
    def continue_nl(self, nl, model, algo: str="greedy"):
        verb_phrases = get_verb_phrases(nl, model)
        cont_seq = [("markdown", phrase) for phrase in verb_phrases]
        if algo == "greedy":
            decoded_cont_seq = self.greedy_decode(cont_seq[-1])
        return cont_seq+decoded_cont_seq

    def greedy_decode(self, node):
        cont_seq = [] # continuation sequence.
        key = self.serialize_node(*node)
        next_node_dist = self.adj.get(key)
        while next_node_dist:
            counts = np.array(list(next_node_dist.values()))
            p = counts / counts.sum() # normalized probabilities.
            next_node_optns = list(next_node_dist.keys())
            key = np.random.choice(next_node_optns, p=p)
            cont_seq.append(self.deserialize_node(key))
            next_node_dist = self.adj.get(key)

        return cont_seq

    def read(self, path: str):
        with open(path, 'r') as f:
            _dict = json.load(f)
            self.adj.update(_dict)

    def add_edge(self, node1: Tuple[str, str], node2: Tuple[str, str]):
        node1 = self.serialize_node(*node1)
        node2 = self.serialize_node(*node2)
        self.adj[node1][node2] += 1

    def write(self, path: str):
        with open(path, "w") as f:
            json.dump(self.adj, f, indent=4)

def get_noun_phrases(sentence, model):
    return [c.text for c in model(sentence).noun_chunks]

def get_verb_phrases(sentence, model):
    # sentence = 'The cat sat on the mat. He quickly ran to the market. The dog jumped into the water. The author is writing a book.'
    pattern = [{'POS': 'VERB', 'OP': '?'},
               {'POS': 'ADV', 'OP': '*'},
               {'POS': 'AUX', 'OP': '*'},
               {'POS': 'VERB', 'OP': '+'}]
    # instantiate a Matcher instance
    matcher = Matcher(model.vocab)
    matcher.add("Verb phrase", [pattern])
    doc = model(sentence) 
    # call the matcher to find matches 
    matches = matcher(doc)

    return [span.text for span in filter_spans([doc[start:end] for _, start, end in matches])]

def instance_to_event_seq(inst: dict, model):
    context = inst["context"]
    plan = []
    for cell in context[::-1]:
        cell_type = cell["cell_type"]
        if cell_type == "markdown":
            sent = " ".join(cell["nl"])
            plan.append(("markdown", get_verb_phrases(sent, model)))
        else: plan.append(("code", cell["api_sequence"]))
    plan.append(("code", inst["api_sequence"]))

    return plan

def flatten_event_seq(event_seq: List[Tuple[str, List[str]]]):
    flat_event_seq = []
    for cell_type, event_list in event_seq:
        if event_list is None:
            print(cell_type)
            continue
        for event in event_list:
            flat_event_seq.append((cell_type, event))

    return flat_event_seq

if __name__ == "__main__":
    d = {}
    model = spacy.load('en_core_web_sm') 
    sentence = "Dimensionality Reduction PCA allows us to reduce the dimensionality of the data -- this will result in the less computation cost . As a side effect , total variance in the data will also decrease.We will use 2 components since it captures 95 % of total vairance ."
    get_verb_phrases(sentence, model)
    model = spacy.load("en_core_web_lg")
    graph = NaiveEventGraph()
    for id, inst in tqdm(d.items()):
        event_seq = instance_to_event_seq(inst, model)
        flat_event_seq = flatten_event_seq(event_seq)
        for i in range(len(flat_event_seq)-1):
            node1 = flat_event_seq[i]
            node2 = flat_event_seq[i+1]
            graph.add_edge(node1, node2)