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

def read_cell_content_and_type_seq(path: str, use_tqdm: bool=True): 
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
            inst = [cell_type_map["start"]]
            for cell in rec["context"][::-1]:
                inst.append(cell_type_map[cell["cell_type"]])
            inst.append(cell_type_map["code"])
            inst.append(cell_type_map["end"])
            cell_seqs.append(inst)
            line_ctr += 1 
    # if line_ctr == 80000: break
    return cell_seqs
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