import copy
import json
import nltk
import gensim
import pprint
import graphviz
import numpy as np
import dataclasses
from typing import *
import gensim.corpora as corpora
from nltk.corpus import stopwords
from collections import defaultdict
from gensim.utils import simple_preprocess
from datautils.plan_graph_extraction import get_noun_phrases, get_verb_phrases

# download stopwords
nltk.download('stopwords')

def get_title_hierarchy_and_stripped_title(title: str):
    ctr = 0
    for char in title:
        if char == "#": ctr += 1
    if ctr == 0: ctr = 1000 # base level is taken as 1000 randomly.
    return ctr, title[ctr:].strip()

# node type object.
class NBTreeNode:
    """The primitive node type for the notebook hierarchy tree."""
    def __init__(self, triple, parent=None):
        self.parent = parent
        self.children = []
        self.triple = triple

    def set_parent(self, parent=None):
        self.parent = parent

    def add_child(self, child):
        self.children.append(child)
        child.parent = self
    
    def add_children(self, children):
        for child in children: self.add_child(child)

    def serialize(self):
        return {
            "value": self.triple.to_dict(),
            "children": [child.serialize() for child in self.children]
        }
        
    def to_json(self):
        return json.dumps(self.serialize())

    def populate_digraph(self, dot):
        cell_type = self.triple.cell_type
        content = self.triple.content[:30]
        dot.node(
            str(self.triple.id), 
            f"{cell_type}({self.triple.id})",
            # f"{cell_type}: {content}",
        )
        for child in self.children:
            dot.edge(str(self.triple.id), str(child.triple.id), constraint='false')
            dot = child.populate_digraph(dot)

        return dot

    def plot(self, path: str, view=False) -> str:
        dot = graphviz.Digraph(comment='Notebook Hierarchy')
        dot = self.populate_digraph(dot)
        print(dot.source)
        # render image and return the path to the rendered image.
        return dot.render(path+".gv", view=view).replace('\\', '/')

# triple object.
@dataclasses.dataclass
class NBNodeTriple:
    id: int=0 # unique integer id.
    level: int=-1
    content: str=""
    cell_type: str="root"

    def __getitem__(self, i: int):
        if i == 0: return self.level
        elif i == 1: return self.content
        elif i == 2: return self.cell_type

    def tuple(self):
        return (self[0], self[1], self[2])

    def to_dict(self):
        return {
            "id": self.id, "level": self.level, 
            "content": self.content, "type": self.cell_type
        }

    def __repr__(self):
        return f'{self.cell_type}({self.level})'

    def __str__(self):
        return json.dumps((self.id, self[0], self[1], self[2]))

def extract_notebook_hierarchy(inst: dict):
    ctxt = inst["context"][::-1]
    triples = []
    id = 1
    for cell in ctxt:
        cell_type = cell["cell_type"]
        if cell_type == "markdown":
            title = cell["nl_original"] # the title/original NL of the markdown cell
            level, stripped_title = get_title_hierarchy_and_stripped_title(title)
            triples.append(
                NBNodeTriple(id, level, stripped_title, cell_type)
            )
        else: 
            if cell_type == "code": content = cell["code"]
            elif cell_type == "heading": content = cell["code"]
            elif cell_type == "raw": content = cell["code"]
            triples.append(
                NBNodeTriple(id, 1000, content, cell_type)
            )
        id += 1 
    triples.append(
        NBNodeTriple(id, 1000, inst["code"], "code")
    )
    root = NBTreeNode(NBNodeTriple())
    curr_top_g = root
    for triple in triples:
        node = NBTreeNode(triple)
        if curr_top_g.triple.level < triple.level: 
            # if current top_g (lowest most senior node) is more senior (less level).
            curr_top_g.add_child(node)
            curr_top_g = node
        elif curr_top_g.triple.level >= triple.level:
            # if current top_g (lowest most senior node) is less senior or equally (more level)
            while curr_top_g.triple.level >= triple.level: # keep moving to parent till more senior node is found.
                curr_top_g = curr_top_g.parent
            curr_top_g.add_child(node)
            curr_top_g =  node

    return root, triples
# def extract_notebook_hierarchy(inst: dict):
#     ctxt = inst["context"][::-1]
#     triples = []
#     for cell in ctxt:
#         cell_type = cell["cell_type"]
#         if cell_type == "markdown":
#             title = cell["nl_original"] # the title/original NL of the markdown cell
#             level, stripped_title = get_title_hierarchy_and_stripped_title(title)
#             triples.append(
#                 NBNodeTriple(level, stripped_title, cell_type)
#             )
#         else: 
#             if cell_type == "code": content = cell["code"]
#             elif cell_type == "heading": content = cell["code"]
#             elif cell_type == "raw": content = cell["code"]
#             triples.append(
#                 NBNodeTriple(1000, content, cell_type)
#             ) 
#     triples.append(
#         NBNodeTriple(1000, inst["code"], "code")
#     )
#     children = []
#     for triple in triples[::-1]:
#         if children == []: 
#             children.append(NBTreeNode(triple))
#         else:
#             child_level = min(child.triple.level for child in children)
#             if triple.level >= child_level:
#                 children.append(NBTreeNode(triple))
#             elif triple.level < child_level:
#                 new_node = NBTreeNode(triple)
#                 new_node.add_children(children)
#                 children = [new_node]
#     root = NBTreeNode(NBNodeTriple())
#     root.add_children(children)

#     return root, triples
    # pass_1, temp, prev_level = [], [], -1
    # for triple in level_cell_type_triples:
    #     if not(triple[0] >= prev_level):
    #         if temp != []: pass_1.append(temp)
    #         temp = [triple]
    #     else: temp.append(triple)
    #     prev_level = triple[0]
    # pass_1.append(temp)

    # return level_cell_type_triples, pass_1
            # new_node = NBTreeNode(
            #     title=stripped_title, 
            #     level=level, cell_type=cell_type
            # )
def extract_title_phrases(data: List[dict], model, path: str):
    titles = defaultdict(lambda:[])
    for rec in data.values():
        for cell in rec["context"]:
            if cell["cell_type"] != "markdown": continue
            for line in cell["nl_original"].split("\n"):
                line = line.strip()
                if not(line.startswith("#")): continue
                level, stripped_title = get_title_hierarchy_and_stripped_title(line)
                verb_phrases = get_verb_phrases(stripped_title, model)
                noun_phrases = get_noun_phrases(stripped_title, model)
                titles[level].append({
                    "verbs": verb_phrases,
                    "nouns": noun_phrases,
                    "title": stripped_title,
                })
    with open(path, "w") as f:
        json.dump(titles, f, indent=4)

def sent_to_words(sentences):    
    for sentence in sentences:        
        # deacc=True removes punctuations        
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

stop_words = stopwords.words('english')
def remove_stopwords(texts):
    global stop_words
    return [[word for word in simple_preprocess(str(doc)) 
             if word not in stop_words] for doc in texts]

def get_lda_topics(data: List[str], num_topics: int=10):
    data_words = remove_stopwords(list(sent_to_words(data)))
    id2word = corpora.Dictionary(data_words)
    texts = data_words
    corpus = []
    for text in texts:
        corpus.append(id2word.doc2bow(text))
    lda_model = gensim.models.LdaMulticore(
        corpus=corpus, 
        id2word=id2word,
        num_topics=num_topics,
    )
    pprint.pprint(lda_model.print_topics())

def cluster_titles(data: List[dict], model, path: str):
    titles = defaultdict(lambda:[])
    for rec in data.values():
        for cell in rec["context"]:
            if cell["cell_type"] != "markdown": continue
            for line in cell["nl_original"].split("\n"):
                line = line.strip()
                if not(line.startswith("#")): continue
                level, stripped_title = get_title_hierarchy_and_stripped_title(line)
                titles[level].append(stripped_title)
    with open(path, "w") as f:
        json.dump(titles, f, indent=4)

def compute_interleave_stats(data: List[dict]):
    seq_lens = {"code": [], "markdown": [], "raw": []}
    switches = [] # number of switches.
    for rec in data.values():
        prev_cell_type = None
        cell_types = [cell["cell_type"] for cell in rec["context"][::-1]]+["code"]
        num_switches = -1
        for cell_type in cell_types:
            if cell_type != prev_cell_type:
                seq_lens[cell_type].append(1)
                num_switches += 1
            else:
                seq_lens[cell_type][-1] += 1
            prev_cell_type = cell_type
        switches.append(num_switches)
    # print(np.sum(seq_lens["code"]))
    # print(len(seq_lens["code"]))
    return {
        "avg. code seq len": round(np.mean(seq_lens["code"]), 3),
        "avg. markdown seq len": round(np.mean(seq_lens["markdown"]), 3),
        "avg. no. of transitions": round(np.mean(switches), 3),
        "max code seq len": np.max(seq_lens["code"]),
        "max markdown seq len": np.max(seq_lens["markdown"]),
        "max no. of transitions": np.max(switches),
    }
# main method.
if __name__ == "__main__":
    d = json.load(open(""))
    titles = cluster_titles(d)
    print(titles)