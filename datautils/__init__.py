# -*- coding: utf-8 -*-
# script to process 

# package for dataset processing

import os
import re
import json
import random
import pathlib
import linecache
from typing import *
from tqdm import tqdm
import nbformat as nbf
from collections import defaultdict

def camel_case_split(identifier, do_lower: bool=False):
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    if do_lower: return [m.group(0).lower() for m in matches]
    return [m.group(0) for m in matches]

def read_jsonl(path: str, use_tqdm: bool=True, 
               cutoff: Union[int, None]=None) -> List[dict]:
    data, i = [], 0
    with open(path, "r") as f:
        for line in tqdm(f, disable=not(use_tqdm)):
            i += 1
            rec = json.loads(line.strip())
            data.append(rec)
            if cutoff is not None and i == cutoff: break
            
    return data


def write_jsonl(data: List[dict], path: str, use_tqdm: bool=True):
    with open(path, "w") as f:
        for inst in tqdm(data, disable=not(use_tqdm)):
            f.write(json.dumps(inst)+"\n")
# not a good idea.
# def load_conala_data_for_code_sim(folder="./data/CoNaLa"):
#     import os
#     from collections import defaultdict
#     train_path = os.path.join(folder, "train.jsonl")
#     train_data = read_jsonl(train_path)[:100000] # load the top 100K most relevant NL-PL pairs.
#     # group together simiar code based on NL.
#     code_synsets = defaultdict(lambda:[])
#     for rec in train_data:
#         nl = rec["intent"]
#         pl = rec["snippet"]
#         code_synsets[nl].append(pl)
#     code_sim_pairs = []
#     for synset in code_synsets.values():
#         for i in range(len(synset)-1):
#             for j in range(i+1, len(synset)):
#                 code_sim_pairs.append((synset[i], synset[j]))

#     return {"train": code_sim_pairs}

class NBCompareBlock:
    def __init__(self, inst: dict):
        self.inst = inst
        self.cell_seq = []
        self.cell_content_to_cell_dict = {}
        for cell in self.inst["context"][::-1]:
            if cell["cell_type"] != "markdown":
                content = cell["code"]
            else: content = cell["nl_original"]
            self.cell_seq.append(content)
            self.cell_content_to_cell_dict[content] = cell 
        self.cell_seq.append(inst["code"])
        
    @classmethod
    def emptyBlock(cls):
        inst = {"code":"", "context":[]}
        obj = cls(inst)
        obj.cell_seq = []
        obj.cell_content_to_cell_dict = {}
        
        return obj

    def __eq__(self, other):
        return other.cell_seq == self.cell_seq

    def __add__(self, other):
        context = {}
        for content, cell in self.cell_content_to_cell_dict.items(): 
            context[content] = cell
        for content, cell in other.cell_content_to_cell_dict.items(): 
            context[content] = cell
        other.cell_seq = list(context.keys())
        other.cell_content_to_cell_dict = context
        other.inst["context"] = list(context.values())[::-1]

        return other

    def __lt__(self, other):
        if len(self.cell_seq) == 0:
            return True
        elif len(other.cell_seq) == 0:
            return False
        last_cell = self.cell_seq[-1]
        try: 
            index_of_last_cell_in_other = other.cell_seq.index(last_cell)
            # definite true
            if index_of_last_cell_in_other < len(other.cell_seq)-1:
                # last cell in this NB is not the last cell of the other NB.
                return True
            else: return not(len(self.cell_seq) < len(other.cell_seq))
        except ValueError:
            try:
                first_cell = self.cell_seq[-1]
                index_of_first_cell_in_other = other.cell_seq.index(first_cell)
                if index_of_first_cell_in_other > 0: return True
                else: return (len(self.cell_seq) < len(other.cell_seq))
            except ValueError:
                return len(self.cell_seq) < len(other.cell_seq) 

def write_dedup_train_data(path_to_ind, save_path: str="./data/juice-dataset/traindedup.jsonl"):
    assert not os.path.exists(save_path)
    open(save_path, "w")
    for uniq_path, nb_ids_and_lens in tqdm(path_to_ind.items()):
        insts = {id: NBCompareBlock(load_train_inst(id)) for id,_ in nb_ids_and_lens}    
        combined = sum(sorted(list(insts.values())), start=NBCompareBlock.emptyBlock())
        with open(save_path, "a") as f:
            f.write(json.dumps(combined.inst)+"\n")

def read_cell_seq(path: str, use_tqdm: bool=True): # block_size: int=1000, num_lines: int=1518104):
    cell_seqs = []
    cell_type_map = {
        "start": 0, "raw": 1, "code": 2,
        "markdown": 3, "heading": 4, "end": 5,
    }
    with open(path, "r") as f: 
        line_ctr = 0
        for line in tqdm(f, disable=not(use_tqdm)):
            line = line.strip()
            rec = json.loads(line)
            inst = [cell_type_map["start"]]
            for cell in rec["context"][::-1]:
                inst.append(cell_type_map[cell["cell_type"]])
            inst.append(cell_type_map["code"])
            inst.append(cell_type_map["end"])
            cell_seqs.append(inst)
            line_ctr += 1 
    # if line_ctr == 80000: break
    return cell_seqs

def read_cell_content_and_type_seq(
        path: str, use_tqdm: bool=True, 
        map_cell_types: bool=False,
        cutoff: int=80000
    ) -> List[Tuple[str, Union[str, int]]]: 
    # block_size: int=1000, num_lines: int=1518104):
    content_and_type_seq = []
    cell_type_map = {
        "start": 0, "raw": 1, "code": 2,
        "markdown": 3, "heading": 4, "end": 5,
    }
    with open(path, "r") as f: 
        line_ctr = 0
        for line in tqdm(f, disable=not(use_tqdm)):
            line = line.strip()
            rec = json.loads(line)
            # inst = [cell_type_map["start"]]
            inst = []
            for cell in rec["context"][::-1]:
                cell_type = cell["cell_type"]
                if cell_type == "markdown": 
                    content = cell["nl_original"]
                else: content = cell["code"]
                if map_cell_types:
                    cell_type = cell_type_map[cell_type]
                inst.append((content, cell_type))
            cell_type = "code"
            if map_cell_types:
                cell_type = cell_type_map[cell_type]
            inst.append((content, cell_type))
            content_and_type_seq.append(inst)
            # inst.append(cell_type_map["end"])
            # cell.append(inst)
            line_ctr += 1 
            if line_ctr == 80000: break
            
    return content_and_type_seq
    # num_blocks = (num_lines-1) // block_size + 1
    # all_idx = list(range(num_lines))
    # for i in range(num_blocks):
    #     idx = all_idx[i:i+1]
    #     for rec in linecache.getlines(path, idx):
def sample_from_trainset(path, idx: List[int]):
    import linecache
    from tqdm import tqdm
    """given a list of line numbers and a file, load the instances on those line numbers
    and return the resultant sampled set of the data"""
    sampled_data = {} # line number is the key and data JSON is the value
    lines = linecache.getlines(path, idx)
    for lineno, line in tqdm(zip(idx, lines)):
        sampled_data[lineno] = json.loads(line.strip())

    return sampled_data

def load_train_inst(ind: int=0, path: str="./data/juice-dataset/train.jsonl"):
    import linecache
    return json.loads(linecache.getline(path, ind+1))

def load_train_insts(idx: List[int]=[], path: str="./data/juice-dataset/train.jsonl"):
    insts = []
    for id in idx: insts.append(load_train_inst(id, path=path))

    return insts

def sample_train_inst_by_stratified_difficulty(path: str, inst_per_bucket: int=200):
    """Difficulty here refers to how much 
    - "in the wild" a Jupyter NB appears (ratio of NL cells to MD cells)
    - "in the wild" a Jupyter NB appears (how long/descriptive the MDs are)
    - 
    - how many libraries it requries (imports), how many function calls it contains, how long the context is,

    
    We creates buckets of difficulty of an instance from the train set based on
    the following notions of diffculty:

    1. 
    2. 
    """

def collapse_train_dups_by_metadata(path: str) -> Dict[str, List[Tuple[int, int]]]:
    """return a list of dictonaries
    keys: metadata paths
    values: list of tuples of (int, int)"""
    import linecache
    from tqdm import tqdm
    
    path_to_ind = defaultdict(lambda:[])
    for ind in tqdm(range(1518105)):
        line = linecache.getline(path, ind+1)
        rec = json.loads(line.strip())
        key = rec["metadata"]["path"]
        path_to_ind[key].append((ind, len(rec["context"])+1))
    path_to_ind = dict(path_to_ind)

    return path_to_ind

def collect_mds_for_kp(path_to_ind) -> Dict[str, List[dict]]:
    from tqdm import tqdm
    from collections import defaultdict

    uniq_mds = defaultdict(lambda:{"mentions": []})
    for key, id_and_len_list in tqdm(path_to_ind.items()):
        for id, _ in id_and_len_list:
            inst = load_train_inst(id)
            for md_id, cell in enumerate(inst["context"][::-1]):
                if cell["cell_type"] != "markdown": continue
                md = cell["nl_original"]
                uniq_mds[md]["mentions"].append({
                    "path": key, 
                    "md_id": md_id,
                    "id": id,
                })
    uniq_mds = dict(uniq_mds)

    return uniq_mds

def context_to_cell_seq(context: List[dict]) -> List[Tuple[str, int]]:
    """convert context attribute of JuICe data to list of cell content and type."""
    cell_seq = []
    for cell in context[::-1]:
        cell_type = cell['cell_type']
        content_key = "nl_original" if cell_type == "markdown" else "code"
        try: cell_seq.append((cell[content_key], cell_type))
        except KeyError:
            print(cell_type)
            print(cell.keys())
            raise KeyError

    return cell_seq

def write_instance_as_nb(inst: dict, path):
    """programmatically create jupyter notebook from JuICe dataset instance."""
    import nbformat as nbf
    nb = nbf.v4.new_notebook()
    nb["cells"] = []
    context = inst["context"][::-1]
    for cell in context:
        cell_type = cell["cell_type"]
        if cell_type == "markdown": content = cell["nl_original"]
        else: content = cell["code"]
        if cell_type == "markdown": nb["cells"].append(nbf.v4.new_markdown_cell(content))
        elif cell_type == "raw": nb["cells"].append(nbf.v4.new_raw_cell(content))
        elif cell_type == "heading": nb["cells"]. append(nbf.v4_new_heading_cell(content))
        else: nb["cells"].append(nbf.v4.new_code_cell(content)) # code cell.
    with open(path, "w") as f: nbf.write(nb, f)

def write_cells_as_nb(cells: List[Tuple[str, int]], path):
    """programmatically create jupyter notebook from cell sequence.
    Each element of the sequence is a (content, cell_type) tuple."""
    nb = nbf.v4.new_notebook()
    nb["cells"] = []
    for content, cell_type in cells:
        if cell_type == "markdown":
            nb["cells"].append(nbf.v4.new_markdown_cell(content))
        elif cell_type == "raw":
            nb["cells"].append(nbf.v4.new_raw_cell(content))
        else: # code cell.
            nb["cells"].append(nbf.v4.new_code_cell(content))
    with open(path, "w") as f:
        nbf.write(nb, f)

def load_key(key: str, path: str, use_tqdm: bool=True) -> List[str]:
    if os.path.exists(path):
        return json.load(open(path))
    data, i = [], 0
    base_name = pathlib.Path(path).name
    if base_name.startswith("train"):
        base_path = "train.jsonl"
    elif base_name.startswith("dev"):
        base_path = "dev.jsonl"
    elif base_name.startswith("test"):
        base_path = "test.jsonl"
    jsonl_path = os.path.join(pathlib.Path(path).parent, base_path)
    with open(jsonl_path, "r") as f:
        for line in tqdm(f, disable=not(use_tqdm)):
            i += 1
            rec = json.loads(line.strip())
            data.append(rec[key])
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    
    return data

def print_context(context: list) -> str:
    context_str = ""
    for cell in context:
        cell_type = cell['cell_type']
        if cell_type == "markdown":
            markdown = "\n".join(["#"+i for i in cell['nl_original'].split("\n")])
            context_str += f"#[{cell['distance_target']}]:\n{markdown}\n"
        elif cell_type == "code":
            context_str += f"#[{cell['distance_target']}]:\n{cell['code']}\n"
    
    return context_str

def load_data(name: str) -> dict:
    data = {}
    folder = os.path.dirname(os.path.abspath(__file__))
    folder = os.path.join(pathlib.Path(folder).parent, "data")
    if name == "juice":
        for split in ["train", "dev", "test"]:#["train", "dev", "test"]:
            path = os.path.join(folder, name+"-dataset", f"{split}.jsonl")
            data[split] = read_jsonl(path)

    return data

def get_value_counts(l: List[str]) -> Dict[str, int]:
    counts = defaultdict(lambda:0)
    for v in l: counts[v] += 1

    return dict(counts)

def load_plan_ops(path: str="./data/juice-dataset/seed_query_relabels.csv") -> List[str]:
    import pathlib
    import pandas as pd
    
    plan_ops = set()
    plan_ops_path = os.path.join(str(pathlib.Path(path).parent), "plan_ops.json")
    for rec in pd.read_csv(path).to_dict("records"):
        if str(rec["human"]) == "SKIP": continue
        name = str(rec["human"]) if str(rec["human"]).strip() != "nan" else str(rec["orig"])
        plan_ops.add(name)
    plan_ops = sorted(list(plan_ops))
    with open(plan_ops_path, "w") as f:
        json.dump(plan_ops, f, indent=4)

    return plan_ops

# def collapse_plan_operators(plan_ops: List[str]):
#     """
#     heuristic: longer plan operators
#     """
#     from tqdm import tqdm
#     from fuzzywuzzy import fuzz
#     from nltk.stem import PorterStemmer
#     # from collections import defaultdict

#     # plan_op2eqv = {k: i for i,k in enumerate(plan_ops)}
#     collapse_pairs = []
#     top_parents = {k: 0 for k in plan_ops}
#     ps = PorterStemmer()
#     for i in tqdm(range(len(plan_ops)-1)):
#         for j in range(i+1, len(plan_ops)):
#             pi = " ".join([ps.stem(w) for w in plan_ops[i].split()])
#             pj = " ".join([ps.stem(w) for w in plan_ops[j].split()])
#             if fuzz.token_set_ratio(pi, pj) >= 95:
#                 # (parent, child)
#                 if len(plan_ops[i]) <= len(plan_ops[j]):
#                     parent = plan_ops[i]
#                     child = plan_ops[j]
#                 else:
#                     parent = plan_ops[j]
#                     child = plan_ops[i]
#                 collapse_pairs.append((parent, child))
#                 try: del top_parents[child]
#                 except KeyError: pass
#                 # eqv_id = min(plan_op2eqv[plan_ops[i]], plan_op2eqv[plan_ops[j]])
#                 # plan_op2eqv[plan_ops[j]] = eqv_id
#                 # plan_op2eqv[plan_ops[i]] = eqv_id
#     top_parents = list(top_parents.keys())
#     # eqv2plan_op = defaultdict(lambda:[])
#     # for k,i in plan_op2eqv.items():
#         # eqv2plan_op[i].append(k)
#     # eqv2plan_op = dict(eqv2plan_op)
#     return collapse_pairs, top_parents
def score_pair_mask(v, thresh: float=0.8):
    import torch
    return ((torch.triu(v @ v.T)-torch.eye(len(v))) >= thresh).bool()

def semantic_pair_plan_operators(plan_ops: List[str]):
    import faiss
    import torch

    faiss_index_path = "./dense_indices/CoNaLa_CSN_CodeBERT_CodeSearch2_CosSim/codebert_plan_ops_cos_sim.index"
    plan_ops_index = faiss.read_index(faiss_index_path)
    plan_ops_vecs = plan_ops_index.reconstruct_n(0, len(plan_ops))
    plan_ops_vecs = torch.as_tensor(plan_ops_vecs)
    
    mask = score_pair_mask(plan_ops_vecs)
    semantic_pairs = []
    print(f"mask: {mask.sum()}")
    for i in tqdm(range(len(plan_ops)-1)):
        for j in range(i+1, len(plan_ops)):
            if mask[i][j].item(): 
                pi = plan_ops[i]
                pj = plan_ops[j]
                semantic_pairs.append((pi, pj))

    return semantic_pairs

def fuzzy_pair_plan_operators(plan_ops: List[str]):
    """
    heuristic: longer plan operators are children of shorter plan operators/longer plan operators are more specific than shorter plan operators
    """
    from tqdm import tqdm
    from fuzzywuzzy import fuzz
    from nltk.stem import PorterStemmer
    # from collections import defaultdict

    # plan_op2eqv = {k: i for i,k in enumerate(plan_ops)}
    fuzzy_pairs = []
    ps = PorterStemmer()
    for i in range(len(plan_ops)):
        fuzzy_pairs.append(("root", plan_ops[i]))
    for i in tqdm(range(len(plan_ops)-1)):
        for j in range(i+1, len(plan_ops)):
            pi = " ".join([ps.stem(w) for w in plan_ops[i].split()])
            pj = " ".join([ps.stem(w) for w in plan_ops[j].split()])
            if fuzz.token_set_ratio(pi, pj) >= 95:
                # (parent, child)
                if len(plan_ops[i]) <= len(plan_ops[j]):
                    parent = plan_ops[i]
                    child = plan_ops[j]
                else:
                    parent = plan_ops[j]
                    child = plan_ops[i]
                fuzzy_pairs.append((parent, child))

    return fuzzy_pairs

class OntologyNode:
    def __init__(self, name: str):
        self.parents = {}
        self.children = {}
        self.name = name

    def add_child(self, child):
        child.parents[self.name] = self
        self.children[child.name] = child

    def to_json(self):
        parents = list(self.parents.keys())
        children = list(self.children.keys())
        return {
            "parents": parents,
            "children": children,
            "name": self.name,
        }

    @classmethod
    def from_json(cls, node_json: Dict[str, List[str]],
                  ontology_map: Dict[str, Any]):
        node = cls(node_json["name"])
        for parent in node_json["parents"]:
            node.parents[parent] = ontology_map.get(
                parent, cls(parent)
            )
            # ontology_map[parent] = node.parents[parent]
        for child in node_json["children"]:
            node.children[child] = ontology_map.get(
                child, cls(child)
            )
            # ontology_map[child] = node.children[child]
        return node#, ontology_map

    def del_children(self, children: list):
        for child in children:
            try: del child.parents[self.name]
            except KeyError: pass
        children_names = [child.name for child in children]
        self.children = {child_name: child for child_name, child in self.children.items() if child_name not in children_names} 

    def del_child(self, child):
        try: del child.parents[self.name]
        except KeyError: pass
        try: del self.children[child.name]
        except KeyError: pass

def save_ontology(name_to_node: Dict[str, Any]):
    ontology_json = {}
    for name, node in name_to_node.items():
        ontology_json[name] = node.to_json()
    with open("./data/juice-dataset/plan_ops_ontology.json", "w") as f:
        json.dump(ontology_json, f, indent=4)

def load_ontology(path: str="./data/juice-dataset/plan_ops_ontology.json"):
    name_to_node = {}
    with open(path, "r") as f:
        ontology_json = json.load(f)
        for name, node_json in ontology_json.items():
            node = OntologyNode.from_json(node_json=node_json,
                                          ontology_map=name_to_node)
            name_to_node[name] = node

    return name_to_node

# def build_ontology(plan_ops: List[str], top_parents: List[str], pairs: List[Tuple[str, str]]):
#     ontology_root = OntologyNode("root")
#     name_to_node = {"root": ontology_root}
#     for name in plan_ops:
#         name_to_node[name] = OntologyNode(name) 
#     for pname in top_parents:
#         pnode = name_to_node[pname]
#         ontology_root.add_child(pnode)
#     for pname, cname in pairs:
#         pnode = name_to_node[pname]
#         cnode = name_to_node[cname]
#         pnode.add_child(cnode)

#     return ontology_root, name_to_node

def build_ontology(plan_ops: List[str], pairs: List[Tuple[str, str]]):
    ontology_root = OntologyNode("root")
    name_to_node = {"root": ontology_root}
    for name in plan_ops:
        name_to_node[name] = OntologyNode(name)
    for pname, cname in pairs:
        pnode = name_to_node[pname]
        cnode = name_to_node[cname]
        pnode.add_child(cnode)

    return ontology_root, name_to_node
        
def simplify_ontology(name_to_node):
    from tqdm import tqdm
    for name in tqdm(name_to_node):
        node = name_to_node[name]
        # orig_size = len(node.children)
        for child in node.children.values():
            grand_children = list(child.children.values())
            # print(f"{child.name}: {[c.name for c in grand_children]}")
            # print([child.name for child in children_to_delete])
            node.del_children(grand_children)
        # new_size = len(node.children)
        # print(f"{name}: {orig_size} -> {new_size}")
        # if name == "root": break
    return name_to_node

# main
if __name__ == "__main__":
    sampled_juice = {}
    idx = random.sample(range(1518104), k=1000)
    sampled_juice = sample_from_trainset("./data/juice-dataset/train.jsonl", idx)
    sampled_juice = {k: v for k,v in sampled_juice.items() if "pandas" in v['imports']}
    with open("./data/juice-dataset/sampled_juice_train.json", "w") as f:
        json.dump(sampled_juice, f, indent=4)
    for i, inst in tqdm(sampled_juice.items()):
        save_path = f"./data/juice-dataset/sampled_juice_nbs/{i}.ipynb"
        cell_seq = context_to_cell_seq(inst['context'])
        # nl_prompt = " ".join(inst['nl']).strip()
        # cell_seq.append((nl_prompt, "markdown"))
        cell_seq.append((inst['code'], "code"))
        write_cells_as_nb(cell_seq, save_path)