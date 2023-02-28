import os
import json
import torch
from typing import *
from tqdm import tqdm
from transformers import AutoTokenizer
from datautils import read_jsonl, read_cell_seq
from torch.utils.data import Dataset, DataLoader

# cell sequence prediction dataloaders:
class SimpleCellSeqDataset(Dataset):
    """simplest possible cell sequence prediction dataset"""
    def __init__(self, path: str, 
                 padding_idx: int=6, 
                 max_seq_len: int=15) -> None:
        super(SimpleCellSeqDataset, self).__init__()
        self.data = read_cell_seq(path)
        self.max_seq_len = max_seq_len
        self.padding_idx = padding_idx
        # self.max_seq_len = max(len(cell_seq) for cell_seq in self.data)
        # self.max_seq_len = min(self.max_seq_len, 15) # at most 15 elements.
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i: int):
        # pad to max length.
        csq = self.data[i][:self.max_seq_len]
        pad_len = self.max_seq_len - len(csq)
        target = csq + [-100 for _ in range(pad_len)]
        padded_seq = csq + [self.padding_idx for _ in range(pad_len)]
        # mask = [1 for _ in range(len(csq))] + [0 for _ in range(pad_len)]
        return [torch.as_tensor(padded_seq), torch.as_tensor(target)]

class SimpleCellSeqDataLoader(DataLoader):
    def __init__(self, dataset, *args, **kwargs):
        super(SimpleCellSeqDataLoader, self).__init__(
            dataset, *args, **kwargs,
        )

class CodeBERTLSTMCellSeqDataset(Dataset):
    """next cell type prediction given previous 
    context of cells and cell types."""
    def __init__(self, path: str, 
                 padding_idx: int=6, 
                 max_seq_len: int=15) -> None:
        super(SimpleCellSeqDataset, self).__init__()
        self.data = read_cell_seq(path)
        self.max_seq_len = max_seq_len
        self.padding_idx = padding_idx
        # self.max_seq_len = max(len(cell_seq) for cell_seq in self.data)
        # self.max_seq_len = min(self.max_seq_len, 15) # at most 15 elements.
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i: int):
        # pad to max length.
        csq = self.data[i][:self.max_seq_len]
        pad_len = self.max_seq_len - len(csq)
        target = csq + [-100 for _ in range(pad_len)]
        padded_seq = csq + [self.padding_idx for _ in range(pad_len)]
        # mask = [1 for _ in range(len(csq))] + [0 for _ in range(pad_len)]
        return [torch.as_tensor(padded_seq), torch.as_tensor(target)]

# next cell type prediction task given the cell sequence and 
class InCoderCellSeqDataset(Dataset):
    def __init__(self, path: str, tok_path="facebook/incoder-1B", 
                 max_ctxt_len: int=5, tok_args: dict={
                    "padding": "max_length",
                    "return_tensors": "pt",                
                    "truncation": True,
                    "max_length": 300, 
                 }):
        super(InCoderCellSeqDataset, self).__init__()
        self.tok_args = tok_args
        self.tokenizer = AutoTokenizer.from_pretrained(tok_path)
        self.tokenizer.pad_token = "<pad>"
        self.tokenizer.padding_side = "left"
        self.data = []
        self.code_type_to_ind = {
            "code": 0, "markdown": 1,
            "heading": 2, "raw": 3, 
        }
        header = "<| file ext=.ipynb:python |>\n"
        with open(path) as f:
            line_ctr = 0
            pbar = tqdm(f)
            for line in pbar:
                line = line.strip()
                rec = json.loads(line)
                ctxt_cells = rec["context"][::-1]
                if len(ctxt_cells) == 0: continue # skip empty contexts.
                content = ctxt_cells[0].get("nl_original", 
                          ctxt_cells[0].get("code"))
                cell_type = ctxt_cells[0]["cell_type"]
                if cell_type == "markdown":
                    content = f"""<text>
{content}
</text>"""
                else: content = f"""<cell>
{content}
</cell>"""
                nb_context = [content]
                for cell in ctxt_cells[1:]:
                    cell_type = cell["cell_type"]
                    content = cell.get("nl_original", cell.get("code"))
                    self.data.append((
                        "\n".join(nb_context[-max_ctxt_len:]),
                        self.code_type_to_ind[cell_type],
                    ))
                    if cell_type == "markdown":
                        nb_context.append(f"""<text>
{content}
</text>""")
                    else: nb_context.append(f"""<cell>
{content}
</cell>""")
                self.data.append((
                    "\n".join(nb_context[-max_ctxt_len:]),
                    self.code_type_to_ind["code"],
                ))
                line_ctr += 1
                pbar.set_description(f"{len(self.data)} inst")
                if line_ctr == 100000: break
    
    def __len__(self): return len(self.data)

    def __getitem__(self, i: int):
        context = self.data[i][0]
        tok_dict = self.tokenizer(context, **self.tok_args)
        cell_type = torch.as_tensor(self.data[i][1])

        return [tok_dict["input_ids"][0], 
                tok_dict["attention_mask"][0], 
                cell_type]

# next cell type prediction task given the cell sequence and 
class InCoderInFillDataset(Dataset):
    def __init__(self, path: str, tok_path="facebook/incoder-1B", 
                 max_ctxt_len: int=5, tok_args: dict={
                    "padding": "max_length",
                    "return_tensors": "pt",                
                    "truncation": True,
                    "max_length": 300, 
                 }):
        super(InCoderInFillDataset, self).__init__()
        self.tok_args = tok_args
        self.tokenizer = AutoTokenizer.from_pretrained(tok_path)
        self.tokenizer.pad_token = "<pad>"
        self.tokenizer.padding_side = "left"
        self.data = []
        self.code_type_to_ind = {
            "code": 0, "markdown": 1,
            "heading": 2, "raw": 3, 
        }
        # header = "<| file ext=.ipynb:python |>\n"
        with open(path) as f:
            line_ctr = 0
            pbar = tqdm(f)
            for line in pbar:
                line = line.strip()
                rec = json.loads(line)
                context_chain = rec["context"][::-1]
                if len(context_chain) == 0: continue # skip empty contexts.
                current_context = [self.wrap_cell(context_chain[0])]
                for cell in context_chain[1:]:
                    cell_type = cell["cell_type"]
                    if cell_type == "markdown":
                        self.data.append([
                            "\n".join(current_context[-max_ctxt_len:])+"""\n<text>""",
                            cell['nl_original'], line_ctr,
                        ])
                    current_context.append(
                        self.wrap_cell(cell)
                    )
                line_ctr += 1
                pbar.set_description(f"{len(self.data)} inst")
                if line_ctr == 100000: break

    def wrap_cell(self, cell: dict):
        cell_type = cell["cell_type"]
        if cell_type == "markdown":
            return self.wrap_markdown(cell)
        else: return self.wrap_code_like(cell)

    def wrap_code_like(self, cell: dict):
        return f"""<cell>
{cell['code']}
</cell>"""

    def wrap_markdown(self, cell: dict):
        return f"""<text>
{cell['nl_original']}
</text>"""

    def __getitem__(self, i: int): return self.data[i]

    def __len__(self): return len(self.data)
    # def __getitem__(self, i: int):
    #     context = self.data[i][0]
    #     tok_dict = self.tokenizer(context, **self.tok_args)
    #     cell_type = torch.as_tensor(self.data[i][1])

    #     return [tok_dict["input_ids"][0], 
    #             tok_dict["attention_mask"][0], 
    #             cell_type]

# dataset class for CoNaLa code search.
class CoNaLaCodeBERTCodeSearchDataset(Dataset):
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