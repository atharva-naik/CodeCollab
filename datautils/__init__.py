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
from torch.utils.data import Dataset, DataLoader

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

# dataset class for CoNaLa code search.
class CoNaLaCodeSearchDataset(Dataset):
    """load CoNaLa data for code-search training."""
    def __init__(self, folder: str="./data/CoNaLa", 
                 split: str="train", tokenizer=None, **tok_args):
        self.split = split
        self.tok_args = tok_args
        self.tokenizer = tokenizer
        self.folder = folder
        if self.split == "train":
            self.data = read_jsonl(os.path.join(
                folder, "train.jsonl"
            ))
        else:
            self.data = json.load(open(
                os.path.join(
                    folder, f"{split}.json"
                )
            ))
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        q = self.data[i]["intent"] # query
        c = self.data[i]["snippet"] # document/code
        q_tok_dict = self.tokenizer(q, **self.tok_args)
        c_tok_dict = self.tokenizer(c, **self.tok_args)
        
        return [
            q_tok_dict["input_ids"][0], q_tok_dict["attention_mask"][0],
            c_tok_dict["input_ids"][0], c_tok_dict["attention_mask"][0],
        ]
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
    """given a list of line numbers and a file, load the instances on those line numbers
    and return the resultant sampled set of the data"""
    sampled_data = {} # line number is the key and data JSON is the value
    lines = linecache.getlines(path, idx)
    for lineno, line in zip(idx, lines):
        sampled_data[lineno] = json.loads(line.strip())

    return sampled_data

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
    nb = nbf.v4.new_notebook()
    nb["cells"] = []
    context = inst["context"][::-1]
    for cell in context:
        cell_type = cell["cell_type"]
        if cell_type == "markdown": content = cell["nl_original"]
        else: content = cell["code"]
        if cell_type == "markdown": nb["cells"].append(nbf.v4.new_markdown_cell(content))
        elif cell_type == "raw": nb["cells"].append(nbf.v4.new_raw_cell(content))
        elif cell_type == "heading": nb["cells"].append(nbf.v4_new_heading_cell(content))
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