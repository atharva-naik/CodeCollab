#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import torch
import numpy as np
from typing import *
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoTokenizer
from tree_sitter import Language, Parser
from torch.utils.data import Dataset, DataLoader

from datautils import read_jsonl, read_cell_seq
from model.parser import DFG_python
from model.parser import (remove_comments_and_docstrings,
                              tree_to_token_index,
                              index_to_code_token,
                              tree_to_variable_index)
from datautils.code_cell_analysis import obfuscate_code

def graphcodebert_proc_code(code: str, parser, tokenizer, tok_args: dict):
    try: code = remove_comments_and_docstrings(code, 'python')
    except: pass
    try: 
        tree = parser[0].parse(bytes(code, 'utf8'))
        root_node = tree.root_node
        tokens_index = tree_to_token_index(root_node)
    except RecursionError:
        # skip very "deep" codes by replacing them with empty strings.
        tree = parser[0].parse(bytes('', 'utf8'))
        root_node = tree.root_node
        tokens_index = tree_to_token_index(root_node)
    code = code.split('\n')
    code_tokens = [index_to_code_token(x, code) for x in tokens_index]  
    index_to_code = {}
    for idx,(index,code) in enumerate(zip(tokens_index, code_tokens)):
        index_to_code[index] = (idx, code)
    try: DFG,_ = parser[1](root_node, index_to_code, {}) 
    except Exception as e: 
        # print("error in src.e_ret.datautils.CoNaLaGraphCodeBERTCodeSearchDataset.proc_code:", e)
        DFG = []
    DFG = sorted(DFG, key=lambda x:x[1])

    # except Exception as e: # Recursion depth exceeded error.
    #     # print("error in src.e_ret.datautils.CoNaLaGraphCodeBERTCodeSearchDataset.proc_code:", e)
    #     DFG = []
    #     print(code)
    #     print(f"\x1b[31;1mcode has length: {len(code)}\x1b[0m")
    #     print(f"\x1b[31;1mcode has tokens: {len(code.split())}\x1b[0m")

    indexs=set()
    for d in DFG:
        if len(d[-1]) != 0: indexs.add(d[1])
        for x in d[-1]: indexs.add(x)
    new_DFG=[]
    for d in DFG:
        if d[1] in indexs: new_DFG.append(d)
    dfg=new_DFG
    code_tokens=[tokenizer.tokenize('@ '+x)[1:] if idx!=0 else tokenizer.tokenize(x) for idx,x in enumerate(code_tokens)]
    ori2cur_pos={}
    ori2cur_pos[-1] = (0,0)
    for i in range(len(code_tokens)):
        ori2cur_pos[i] = (ori2cur_pos[i-1][1], ori2cur_pos[i-1][1] + len(code_tokens[i]))    
    code_tokens=[y for x in code_tokens for y in x]  
    #truncating
    code_tokens = code_tokens[:tok_args["code_length"] + tok_args["data_flow_length"] - 2 - min(len(dfg), tok_args["data_flow_length"])]
    code_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
    position_idx = [i+tokenizer.pad_token_id + 1 for i in range(len(code_tokens))]
    dfg=dfg[:tok_args["code_length"] + tok_args["data_flow_length"]
            -len(code_tokens)]
    code_tokens += [x[0] for x in dfg]
    position_idx += [0 for x in dfg]
    code_ids += [tokenizer.unk_token_id for x in dfg]
    padding_length = tok_args["code_length"] + tok_args["data_flow_length"] - len(code_ids)
    position_idx += [tokenizer.pad_token_id]*padding_length
    code_ids += [tokenizer.pad_token_id]*padding_length    
    # reindex
    reverse_index = {}
    for idx,x in enumerate(dfg):
        reverse_index[x[1]] = idx
    for idx,x in enumerate(dfg):
        dfg[idx] = x[:-1]+([reverse_index[i] for i in x[-1] if i in reverse_index],)    
    dfg_to_dfg = [x[-1] for x in dfg]
    dfg_to_code = [ori2cur_pos[x[1]] for x in dfg]
    length = len([tokenizer.cls_token])
    dfg_to_code = [(x[0] + length, x[1] + length) for x in dfg_to_code] 
    #calculate graph-guided masked function
    attn_mask = np.zeros((tok_args["code_length"] + tok_args["data_flow_length"],
                          tok_args["code_length"] + tok_args["data_flow_length"]), 
                          dtype=bool)
    #calculate begin index of node and max length of input
    node_index = sum([i > 1 for i in position_idx])
    max_length = sum([i != 1 for i in position_idx])
    #sequence can attend to sequence
    attn_mask[:node_index,:node_index] = True
    #special tokens attend to all tokens
    for idx,i in enumerate(code_ids):
        if i in [0,2]:
            attn_mask[idx,:max_length] = True
    #nodes attend to code tokens that are identified from
    for idx, (a,b) in enumerate(dfg_to_code):
        if a < node_index and b < node_index:
            attn_mask[idx + node_index,a:b] = True
            attn_mask[a:b,idx + node_index] = True
    #nodes attend to adjacent nodes 
    for idx,nodes in enumerate(dfg_to_dfg):
        for a in nodes:
            if a + node_index<len(position_idx):
                attn_mask[idx + node_index,a + node_index] = True  
    
    c_iids = torch.tensor(code_ids)
    c_attn = torch.tensor(attn_mask)
    c_pos_idx = torch.tensor(position_idx) 
    
    return c_iids, c_attn, c_pos_idx

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

# query datasets for CoNaLa:
class QueryCodeBERTDataset(Dataset):
    def __init__(self, queries: List[str], 
                 doc_ids: List[List[int]],
                 tokenizer, **tok_args):
        super(QueryCodeBERTDataset, self).__init__()
        self.tokenizer = tokenizer
        self.tok_args = tok_args
        self.data = queries
        print(type(queries), len(queries))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        q = self.data[i]
        # label = self.data
        # print(q, type(q))
        q_tok_dict = self.tokenizer(q, **self.tok_args)
        
        return [
            q_tok_dict["input_ids"][0], 
            q_tok_dict["attention_mask"][0],
        ]

class QueryGraphCodeBERTDataset(Dataset):
    def __init__(self, queries: List[str], doc_ids: List[List[int]],
                 tokenizer, parser, **tok_args):
        super(QueryGraphCodeBERTDataset, self).__init__()
        self.tokenizer = tokenizer
        self.tok_args = tok_args
        self.parser = parser
        self.data = queries
        print(type(queries), len(queries))

    def __len__(self):
        return len(self.data)

    def proc_text(self, q: str):
        nl_tokens=self.tokenizer.tokenize(q)[:self.tok_args["nl_length"]-2]
        nl_tokens =[self.tokenizer.cls_token]+nl_tokens+[self.tokenizer.sep_token]
        nl_ids =  self.tokenizer.convert_tokens_to_ids(nl_tokens)
        padding_length = self.tok_args["nl_length"] - len(nl_ids)
        nl_ids+=[self.tokenizer.pad_token_id]*padding_length

        return torch.tensor(nl_ids)

    def __getitem__(self, i):
        q = self.data[i]
        q_iids = self.proc_text(q)        

        return [q_iids]

class QueryUniXcoderDataset(Dataset):
    def __init__(self, queries: List[str], 
                 doc_ids: List[List[int]],
                 tokenizer, **tok_args):
        super(QueryUniXcoderDataset, self).__init__()
        self.tokenizer = tokenizer
        self.tok_args = tok_args
        self.data = queries
        print(type(queries), len(queries))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        q = self.data[i]

        return [self.tokenizer(q, **self.tok_args)["input_ids"][0]]

# docs datasets for CoNaLa
class DocsCodeBERTDataset(Dataset):
    def __init__(self, docs: List[str], 
                 tokenizer, **tok_args):
        super(DocsCodeBERTDataset, self).__init__()
        self.tokenizer = tokenizer
        self.tok_args = tok_args
        self.data = docs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        d = self.data[i]
        d_tok_dict = self.tokenizer(d, **self.tok_args)
        
        return [
            d_tok_dict["input_ids"][0], 
            d_tok_dict["attention_mask"][0],
        ]

class DocsGraphCodeBERTDataset(Dataset):
    def __init__(self, docs: List[str], 
                 tokenizer, parser, **tok_args):
        super(DocsGraphCodeBERTDataset, self).__init__()
        self.tokenizer = tokenizer
        self.tok_args = tok_args
        self.parser = parser
        self.data = docs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        c = self.data[i]
        c_iids, c_attn, c_pos_idx = graphcodebert_proc_code(code=c, parser=self.parser, 
                                                            tokenizer=self.tokenizer, 
                                                            tok_args=self.tok_args)

        return [c_iids, c_attn, c_pos_idx]

class DocsUniXcoderDataset(Dataset):
    def __init__(self, docs: List[str], 
                 tokenizer, **tok_args):
        super(DocsUniXcoderDataset, self).__init__()
        self.tokenizer = tokenizer
        self.tok_args = tok_args
        self.data = docs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        d = self.data[i]
        return [self.tokenizer(d, **self.tok_args)["input_ids"][0]]

# query datasets for inference:
class QueryCodeBERTInferenceDataset(Dataset):
    def __init__(self, queries: List[str], 
                 tokenizer, **tok_args):
        super(QueryCodeBERTInferenceDataset, self).__init__()
        self.tokenizer = tokenizer
        self.kwargs = tok_args
        self.data = queries

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        q = self.data[i]
        id = q['id']
        q_text = q['text']
        q_tok_dict = self.tokenizer(q_text, **self.kwargs)
        
        return [
            id,
            q_tok_dict["input_ids"].squeeze(),
            q_tok_dict["attention_mask"].squeeze(),
        ]

class QueryGraphCodeBERTInferenceDataset(Dataset):
    def __init__(self, queries: List[str],
                 tokenizer, **tok_args):
        super(QueryGraphCodeBERTInferenceDataset, self).__init__()
        self.tokenizer = tokenizer
        self.tok_args = tok_args
        LANGUAGE = Language('./model/parser/py_parser.so', 'python')
        PARSER = Parser()
        PARSER.set_language(LANGUAGE)
        self.parser = [PARSER, DFG_python]
        self.data = queries
        print(type(queries), len(queries))

    def __len__(self):
        return len(self.data)

    def proc_text(self, q: str):
        nl_tokens=self.tokenizer.tokenize(q)[:self.tok_args["nl_length"]-2]
        nl_tokens =[self.tokenizer.cls_token]+nl_tokens+[self.tokenizer.sep_token]
        nl_ids =  self.tokenizer.convert_tokens_to_ids(nl_tokens)
        padding_length = self.tok_args["nl_length"] - len(nl_ids)
        nl_ids+=[self.tokenizer.pad_token_id]*padding_length

        return torch.tensor(nl_ids)

    def __getitem__(self, i):
        q = self.data[i]
        q_id = q['id']
        q_text = q['text']
        q_iids = self.proc_text(q_text)

        return [q_id, q_iids]

class QueryUniXcoderInferenceDataset(Dataset):
    def __init__(self, queries: List[str], 
                 tokenizer, **tok_args):
        super(QueryUniXcoderInferenceDataset, self).__init__()
        self.tokenizer = tokenizer
        self.tok_args = tok_args
        self.data = queries
        print(type(queries), len(queries))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        q = self.data[i]
        q_id = q['id']
        q_text = q['text']

        return [q_id, self.tokenizer(q_text, **self.tok_args)["input_ids"][0]]

# docs datasets for inference
class DocsCodeBERTInferenceDataset(Dataset):
    def __init__(self, docs: List[str], 
                 tokenizer, **tok_args):
        super(DocsCodeBERTInferenceDataset, self).__init__()
        self.tokenizer = tokenizer
        self.tok_args = tok_args
        self.data = docs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        d = self.data[i]
        d_id = d['id']
        d_text = d['text']
        d_tok_dict = self.tokenizer(d_text, **self.tok_args)
        
        return [
            d_id,
            d_tok_dict["input_ids"].squeeze(), 
            d_tok_dict["attention_mask"].squeeze(),
        ]

class DocsGraphCodeBERTInferenceDataset(Dataset):
    def __init__(self, docs: List[str], 
                 tokenizer, **tok_args):
        super(DocsGraphCodeBERTInferenceDataset, self).__init__()
        self.tokenizer = tokenizer
        self.tok_args = tok_args
        
        LANGUAGE = Language('./model/parser/py_parser.so', 'python')
        PARSER = Parser()
        PARSER.set_language(LANGUAGE)
        self.parser = [PARSER, DFG_python]

        self.data = docs
    def get_example(self, i):
        return self.data[i]
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        c = self.data[i]
        # c_id = c['id']
        c_text = c['text']
        c_iids, c_attn, c_pos_idx = graphcodebert_proc_code(code=c_text, parser=self.parser, 
                                                            tokenizer=self.tokenizer, 
                                                            tok_args=self.tok_args)

        return [c_iids, c_attn, c_pos_idx]

class DocsUniXcoderInferenceDataset(Dataset):
    def __init__(self, docs: List[str], 
                 tokenizer, **tok_args):
        super(DocsUniXcoderInferenceDataset, self).__init__()
        self.tokenizer = tokenizer
        self.tok_args = tok_args
        self.data = docs
    def get_example(self, i):
        return self.data[i]
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i, include_ids: bool=False):
        d = self.data[i]
        d_id = d['id']
        d_text = d['text']
        if include_ids:
            return [d_id, self.tokenizer(d_text, **self.tok_args)["input_ids"][0]]
        else:
            return [self.tokenizer(d_text, **self.tok_args)["input_ids"][0]]

# inference classes
class JuICeKBNNCodeBERTCodeSearchDataset(Dataset):
    """load JuICe Code KB data for NN search using CodeBERT dense representations"""
    def __init__(self, folder: str="./JuICe_train_code_KB.json", queries=None, 
                 obf_code: bool=False, tokenizer=None, **tok_args):
        super(JuICeKBNNCodeBERTCodeSearchDataset, self).__init__()
        self.tok_args = tok_args
        self.tokenizer = tokenizer
        self.folder = folder
        self.obf_code = obf_code
        folder_stem, ext = os.path.splitext(folder)
        if obf_code:
            folder_ = folder_stem + "_obf" + ext
            if not os.path.exists(folder_):
                codes = list(json.load(open(folder)).keys())
                for ind in tqdm(range(len(codes))):
                    try: codes[ind] = obfuscate_code(codes[ind])
                    except RecursionError: pass
                with open(folder_, "w") as f:
                    json.dump(codes, f, indent=4)
            folder = folder_
        if queries is not None:
            self.docs = queries
        else:
            self.data = json.load(open(folder))
            self.docs = []
            if isinstance(self.data, dict):
                for i, code in enumerate(self.data.keys()):
                    self.docs.append(code)
            elif isinstance(self.data, list):
                for i, code in enumerate(self.data):
                    self.docs.append(code)

    def get_docs_loader(self, batch_size: int=100):
        dset = DocsCodeBERTDataset(self.docs, self.tokenizer, **self.tok_args)
        d_loader = DataLoader(dset, batch_size=batch_size, shuffle=False)
        
        return d_loader

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

class JuICeKBNNGraphCodeBERTCodeSearchDataset(Dataset):
    """load JuICe Code KB data for NN search using GraphCodeBERT dense representations"""
    def __init__(self, folder: str="./JuICe_train_code_KB.json", queries=None, 
                 obf_code: bool=False, tokenizer=None, skip: int=0, **tok_args):
        super(JuICeKBNNGraphCodeBERTCodeSearchDataset, self).__init__()
        self.tok_args = tok_args
        self.tokenizer = tokenizer
        self.folder = folder
        self.obf_code = obf_code
        LANGUAGE = Language('./model/parser/py_parser.so', 'python')
        PARSER = Parser()
        PARSER.set_language(LANGUAGE)
        self.parser = [PARSER, DFG_python]
        folder_stem, ext = os.path.splitext(folder)
        if obf_code:
            folder_ = folder_stem + "_obf" + ext
            if not os.path.exists(folder_):
                codes = list(json.load(open(folder)).keys())
                for ind in tqdm(range(len(codes))):
                    try: codes[ind] = obfuscate_code(codes[ind])
                    except RecursionError: pass
                with open(folder_, "w") as f:
                    json.dump(codes, f, indent=4)
            folder = folder_
        if queries is not None:
            self.docs = queries[:skip]
        else:
            self.data = json.load(open(folder))
            self.docs = []
            if isinstance(self.data, dict):
                for i, code in enumerate(self.data.keys()):
                    self.docs.append(code)
            elif isinstance(self.data, list):
                for i, code in enumerate(self.data):
                    self.docs.append(code)
        self.docs = self.docs[skip:]

    def get_docs_loader(self, batch_size: int=100):
        dset = DocsGraphCodeBERTDataset(
            self.docs, self.tokenizer, 
            self.parser, **self.tok_args
        )
        d_loader = DataLoader(dset, batch_size=batch_size, shuffle=False)
        
        return d_loader

class JuICeKBNNUniXcoderCodeSearchDataset(Dataset):
    """load JuICe Code KB data for NN search using GraphCodeBERT dense representations"""
    def __init__(self, folder: str="./JuICe_train_code_KB.json", queries=None, 
                 obf_code: bool=False, tokenizer=None, **tok_args):
        super(JuICeKBNNUniXcoderCodeSearchDataset, self).__init__()
        self.tok_args = tok_args
        self.tokenizer = tokenizer
        self.folder = folder
        self.obf_code = obf_code
        folder_stem, ext = os.path.splitext(folder)
        if obf_code:
            folder_ = folder_stem + "_obf" + ext
            if not os.path.exists(folder_):
                codes = list(json.load(open(folder)).keys())
                for ind in tqdm(range(len(codes))):
                    try: codes[ind] = obfuscate_code(codes[ind])
                    except RecursionError: pass
                with open(folder_, "w") as f:
                    json.dump(codes, f, indent=4)
            folder = folder_
        if queries is not None:
            self.docs = queries
        else:
            self.data = json.load(open(folder))
            self.docs = []
            if isinstance(self.data, dict):
                for i, code in enumerate(self.data.keys()):
                    self.docs.append(code)
            elif isinstance(self.data, list):
                for i, code in enumerate(self.data):
                    self.docs.append(code)

    def get_docs_loader(self, batch_size: int=100):
        dset = DocsUniXcoderDataset(self.docs, self.tokenizer, **self.tok_args)
        d_loader = DataLoader(dset, batch_size=batch_size, shuffle=False)
        
        return d_loader

# training classes
class CodeSearchNetCodeBERTCodeSearchDataset(Dataset):
    """load CodeSearchNet data for code-search training."""
    def __init__(self, folder: str="./data/CoNaLa", split: str="train", tokenizer=None, 
                 obf_code: bool=False, csn_folder="./data/CodeSearchNet/", 
                 filt_conala_top_k: bool=True, filt_k: int=100000, **tok_args):
        super(CodeSearchNetCodeBERTCodeSearchDataset, self).__init__()
        self.split = split
        print(f"obf_code: {obf_code}")
        self.tok_args = tok_args
        self.tokenizer = tokenizer
        self.folder = folder
        self.filt_k = filt_k
        self.filt_conala_top_k = filt_conala_top_k
        if self.split == "train":
            self.data = read_jsonl(os.path.join(
                folder, "train.jsonl"
            ))
            # filter CoNaLa instances based on decreasing order of relevance
            # CoNaLa data is already organized in decreasing order of relevance.
            if filt_conala_top_k:
                self.data = self.data[:self.filt_k]
            if obf_code: desc = "obfuscating CoNaLa"
            else: desc = "loading CoNaLa"          
            for i in tqdm(range(len(self.data)), desc=desc): 
                # remove comments and docstrings.
                self.data[i]["snippet"] = remove_comments_and_docstrings(self.data[i]["snippet"])
                if obf_code: self.data[i]["snippet"] = obfuscate_code(self.data[i]["snippet"])
            if csn_folder is not None:
                print(f"\x1b[32;1madding CodeSearchNet data\x1b[0m")
                all_csn_data = []
                for idx in range(5):
                    csn_data = read_jsonl(os.path.join(csn_folder, "train", f"python_train_{idx}.jsonl"))
                    if obf_code: desc = "obfuscating CSN"
                    else: desc = "loading CSN"
                    for rec in tqdm(csn_data, desc=desc):
                        code = remove_comments_and_docstrings(rec["code"])
                        if obf_code: code = obfuscate_code(code)
                        all_csn_data.append({
                            "snippet": code,
                            "intent": rec["docstring"]
                        })
                print(f"\x1b[32;1madded CodeSearchNet data\x1b[0m")
                self.data += all_csn_data
        else:
            self.data = json.load(open(
                os.path.join(
                    folder, f"{split}.json"
                )
            ))
            for i in tqdm(range(len(self.data)), desc="obfuscating code"): 
                # remove comments and docstrings.
                self.data[i]["snippet"] = remove_comments_and_docstrings(self.data[i]["snippet"])
                if obf_code: self.data[i]["snippet"] = obfuscate_code(self.data[i]["snippet"])
            self.queries = {}
            self.doc_ids = defaultdict(lambda:[])
            self.docs = {}
            d_ctr = 0
            q_ctr = 0
            for i, rec in enumerate(self.data):
                q = rec['intent']
                d = rec['snippet']
                if q not in self.queries: 
                    self.queries[q] = q_ctr
                    q_ctr += 1
                if d not in self.docs: 
                    self.docs[d] = d_ctr
                    d_ctr += 1
            for i, rec in enumerate(self.data):
                q = rec['intent']
                d = rec['snippet']
                self.doc_ids[q].append(self.docs[d])
            self.queries = list(self.queries.keys())
            self.docs = list(self.docs.keys())
            self.doc_ids = list(self.doc_ids.values())
            # print(self.doc_ids[-5:])
    def get_doc_ids_mask(self):
        mask = np.zeros((len(self.queries), len(self.docs)))
        for i, ids in enumerate(self.doc_ids):
            for j in ids: mask[i][j] = 1
        return mask

    def get_query_loader(self, batch_size: int=100):
        qset = QueryCodeBERTDataset(self.queries, self.doc_ids, 
                            self.tokenizer, **self.tok_args)
        q_loader = DataLoader(qset, batch_size=batch_size, shuffle=False)
        return q_loader

    def get_docs_loader(self, batch_size: int=100):
        dset = DocsCodeBERTDataset(self.docs, self.tokenizer, **self.tok_args)
        d_loader = DataLoader(dset, batch_size=batch_size, shuffle=False)
        return d_loader

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

class CoNaLaGraphCodeBERTCodeSearchDataset(Dataset):
    """load CoNaLa data for code-search training."""
    def __init__(self, folder: str="./data/CoNaLa", 
                 split: str="train", tokenizer=None, **tok_args):
        super(CoNaLaGraphCodeBERTCodeSearchDataset, self).__init__()
        self.split = split
        self.tok_args = tok_args
        self.tokenizer = tokenizer
        LANGUAGE = Language('./model/parser/py_parser.so', 'python')
        PARSER = Parser()
        PARSER.set_language(LANGUAGE)
        self.parser = [PARSER, DFG_python]
        self.folder = folder
        if self.split == "train":
            self.data = read_jsonl(os.path.join(
                folder, "train.jsonl"
            ))
        else:
            self.data = read_jsonl(os.path.join(
                    folder, f"{split}.jsonl"
            ))
            self.queries = {}
            self.doc_ids = defaultdict(lambda:[])
            self.docs = {}
            d_ctr = 0
            q_ctr = 0
            for i, rec in enumerate(self.data):
                q = rec['intent']
                d = rec['snippet']
                if q not in self.queries: 
                    self.queries[q] = q_ctr
                    q_ctr += 1
                if d not in self.docs: 
                    self.docs[d] = d_ctr
                    d_ctr += 1
            for i, rec in enumerate(self.data):
                q = rec['intent']
                d = rec['snippet']
                self.doc_ids[q].append(self.docs[d])
            self.queries = list(self.queries.keys())
            self.docs = list(self.docs.keys())
            self.doc_ids = list(self.doc_ids.values())
            # print(self.doc_ids[-5:])
            
    def get_doc_ids_mask(self):
        mask = np.zeros((len(self.queries), len(self.docs)))
        for i, ids in enumerate(self.doc_ids):
            for j in ids: mask[i][j] = 1
        return mask

    def get_query_loader(self, batch_size: int=100):
        qset = QueryGraphCodeBERTDataset(
            self.queries, self.doc_ids, 
            self.tokenizer, self.parser, 
            **self.tok_args
        )
        q_loader = DataLoader(qset, batch_size=batch_size, shuffle=False)
        return q_loader

    def get_docs_loader(self, batch_size: int=100):
        dset = DocsGraphCodeBERTDataset(
            self.docs, self.tokenizer, 
            self.parser, **self.tok_args
        )
        d_loader = DataLoader(dset, batch_size=batch_size, shuffle=False)
        return d_loader

    def __len__(self):
        return len(self.data)

    def proc_text(self, q: str):
        nl_tokens=self.tokenizer.tokenize(q)[:self.tok_args["nl_length"]-2]
        nl_tokens =[self.tokenizer.cls_token]+nl_tokens+[self.tokenizer.sep_token]
        nl_ids =  self.tokenizer.convert_tokens_to_ids(nl_tokens)
        padding_length = self.tok_args["nl_length"] - len(nl_ids)
        nl_ids+=[self.tokenizer.pad_token_id]*padding_length

        return torch.tensor(nl_ids)

    def __getitem__(self, i):
        q = self.data[i]["intent"] # query
        c = self.data[i]["snippet"] # document/code
        q_iids = self.proc_text(q)     
        c_iids, c_attn, c_pos_idx = graphcodebert_proc_code(code=c, parser=self.parser, 
                                                            tokenizer=self.tokenizer, 
                                                            tok_args=self.tok_args)

        return [
            q_iids, c_iids, 
            c_attn, c_pos_idx,
        ]

class CodeSearchNetGraphCodeBERTCodeSearchDataset(Dataset):
    """load CoNaLa+CodeSearchNet data for code-search training."""
    def __init__(self, folder: str="./data/CoNaLa", split: str="train", tokenizer=None, 
                 obf_code: bool=False, csn_folder="./data/CodeSearchNet/", 
                 filt_conala_top_k: bool=True, filt_k: int=100000, **tok_args):
        super(CodeSearchNetGraphCodeBERTCodeSearchDataset, self).__init__()
        self.split = split
        LANGUAGE = Language('./model/parser/py_parser.so', 'python')
        PARSER = Parser()
        PARSER.set_language(LANGUAGE)
        self.parser = [PARSER, DFG_python]
        print(f"obf_code: {obf_code}")
        self.tok_args = tok_args
        self.tokenizer = tokenizer
        self.folder = folder
        self.filt_k = filt_k
        self.filt_conala_top_k = filt_conala_top_k
        if self.split == "train":
            self.data = read_jsonl(os.path.join(
                folder, "train.jsonl"
            ))
            # filter CoNaLa instances based on decreasing order of relevance
            # CoNaLa data is already organized in decreasing order of relevance.
            if filt_conala_top_k:
                self.data = self.data[:self.filt_k]
            if obf_code: desc = "obfuscating CoNaLa"
            else: desc = "loading CoNaLa"          
            for i in tqdm(range(len(self.data)), desc=desc): 
                # remove comments and docstrings.
                self.data[i]["snippet"] = remove_comments_and_docstrings(self.data[i]["snippet"])
                if obf_code: self.data[i]["snippet"] = obfuscate_code(self.data[i]["snippet"])
            if csn_folder is not None:
                print(f"\x1b[32;1madding CodeSearchNet data\x1b[0m")
                all_csn_data = []
                for idx in range(5):
                    csn_data = read_jsonl(os.path.join(csn_folder, "train", f"python_train_{idx}.jsonl"))
                    if obf_code: desc = "obfuscating CSN"
                    else: desc = "loading CSN"
                    for rec in tqdm(csn_data, desc=desc):
                        code = remove_comments_and_docstrings(rec["code"])
                        if obf_code: code = obfuscate_code(code)
                        all_csn_data.append({
                            "snippet": code,
                            "intent": rec["docstring"]
                        })
                print(f"\x1b[32;1madded CodeSearchNet data\x1b[0m")
                self.data += all_csn_data
        else:
            self.data = json.load(open(
                os.path.join(
                    folder, f"{split}.json"
                )
            ))
            for i in tqdm(range(len(self.data)), desc="obfuscating code"): 
                # remove comments and docstrings.
                self.data[i]["snippet"] = remove_comments_and_docstrings(self.data[i]["snippet"])
                if obf_code: self.data[i]["snippet"] = obfuscate_code(self.data[i]["snippet"])
            self.queries = {}
            self.doc_ids = defaultdict(lambda:[])
            self.docs = {}
            d_ctr = 0
            q_ctr = 0
            for i, rec in enumerate(self.data):
                q = rec['intent']
                d = rec['snippet']
                if q not in self.queries: 
                    self.queries[q] = q_ctr
                    q_ctr += 1
                if d not in self.docs: 
                    self.docs[d] = d_ctr
                    d_ctr += 1
            for i, rec in enumerate(self.data):
                q = rec['intent']
                d = rec['snippet']
                self.doc_ids[q].append(self.docs[d])
            self.queries = list(self.queries.keys())
            self.docs = list(self.docs.keys())
            self.doc_ids = list(self.doc_ids.values())
            
    def get_doc_ids_mask(self):
        mask = np.zeros((len(self.queries), len(self.docs)))
        for i, ids in enumerate(self.doc_ids):
            for j in ids: mask[i][j] = 1
        return mask

    def get_query_loader(self, batch_size: int=100):
        qset = QueryGraphCodeBERTDataset(
            self.queries, self.doc_ids, 
            self.tokenizer, self.parser, 
            **self.tok_args
        )
        q_loader = DataLoader(qset, batch_size=batch_size, shuffle=False)
        return q_loader

    def get_docs_loader(self, batch_size: int=100):
        dset = DocsGraphCodeBERTDataset(
            self.docs, self.tokenizer, 
            self.parser, **self.tok_args
        )
        d_loader = DataLoader(dset, batch_size=batch_size, shuffle=False)
        return d_loader

    def __len__(self):
        return len(self.data)

    def proc_text(self, q: str):
        nl_tokens=self.tokenizer.tokenize(q)[:self.tok_args["nl_length"]-2]
        nl_tokens =[self.tokenizer.cls_token]+nl_tokens+[self.tokenizer.sep_token]
        nl_ids =  self.tokenizer.convert_tokens_to_ids(nl_tokens)
        padding_length = self.tok_args["nl_length"] - len(nl_ids)
        nl_ids+=[self.tokenizer.pad_token_id]*padding_length

        return torch.tensor(nl_ids)

    def __getitem__(self, i):
        q = self.data[i]["intent"] # query
        c = self.data[i]["snippet"] # document/code
        q_iids = self.proc_text(q)     
        c_iids, c_attn, c_pos_idx = graphcodebert_proc_code(code=c, parser=self.parser, 
                                                            tokenizer=self.tokenizer, 
                                                            tok_args=self.tok_args)

        return [
            q_iids, c_iids, 
            c_attn, c_pos_idx,
        ]

class CoNaLaUniXcoderCodeSearchDataset(Dataset):
    """load CoNaLa data for code-search training."""
    def __init__(self, folder: str="./data/CoNaLa", 
                 split: str="train", tokenizer=None, **tok_args):
        super(CoNaLaUniXcoderCodeSearchDataset, self).__init__()
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
            self.queries = {}
            self.doc_ids = defaultdict(lambda:[])
            self.docs = {}
            d_ctr = 0
            q_ctr = 0
            for i, rec in enumerate(self.data):
                q = rec['intent']
                d = rec['snippet']
                if q not in self.queries: 
                    self.queries[q] = q_ctr
                    q_ctr += 1
                if d not in self.docs: 
                    self.docs[d] = d_ctr
                    d_ctr += 1
            for i, rec in enumerate(self.data):
                q = rec['intent']
                d = rec['snippet']
                self.doc_ids[q].append(self.docs[d])
            self.queries = list(self.queries.keys())
            self.docs = list(self.docs.keys())
            self.doc_ids = list(self.doc_ids.values())
            # print(self.doc_ids[-5:])
    def get_doc_ids_mask(self):
        mask = np.zeros((len(self.queries), len(self.docs)))
        for i, ids in enumerate(self.doc_ids):
            for j in ids: mask[i][j] = 1
        return mask

    def get_query_loader(self, batch_size: int=100):
        qset = QueryUniXcoderDataset(
            self.queries, self.doc_ids, 
            self.tokenizer, **self.tok_args,
        )
        q_loader = DataLoader(qset, batch_size=batch_size, shuffle=False)
        return q_loader

    def get_docs_loader(self, batch_size: int=100):
        dset = DocsUniXcoderDataset(
            self.docs, self.tokenizer, 
            **self.tok_args
        )
        d_loader = DataLoader(dset, batch_size=batch_size, shuffle=False)
        return d_loader

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        q = self.data[i]["intent"] # query
        c = self.data[i]["snippet"] # document/code
        q_iids = self.tokenizer(q, **self.tok_args)["input_ids"][0]
        c_iids = self.tokenizer(c, **self.tok_args)["input_ids"][0]
        
        return [q_iids, c_iids]

class CodeSearchNetUniXcoderCodeSearchDataset(Dataset):
    """load CoNaLa+CodeSearchNet data for code-search training."""
    def __init__(self, folder: str="./data/CoNaLa", split: str="train", tokenizer=None, 
                 obf_code: bool=False, csn_folder="./data/CodeSearchNet/", 
                 filt_conala_top_k: bool=True, filt_k: int=100000, **tok_args):
        super(CodeSearchNetUniXcoderCodeSearchDataset, self).__init__()
        self.split = split
        print(f"obf_code: {obf_code}")
        self.tok_args = tok_args
        self.tokenizer = tokenizer
        self.folder = folder
        self.filt_k = filt_k
        self.filt_conala_top_k = filt_conala_top_k
        if self.split == "train":
            self.data = read_jsonl(os.path.join(
                folder, "train.jsonl"
            ))
            # filter CoNaLa instances based on decreasing order of relevance
            # CoNaLa data is already organized in decreasing order of relevance.
            if filt_conala_top_k:
                self.data = self.data[:self.filt_k]
            if obf_code: desc = "obfuscating CoNaLa"
            else: desc = "loading CoNaLa"          
            for i in tqdm(range(len(self.data)), desc=desc): 
                # remove comments and docstrings.
                self.data[i]["snippet"] = remove_comments_and_docstrings(self.data[i]["snippet"])
                if obf_code: self.data[i]["snippet"] = obfuscate_code(self.data[i]["snippet"])
            if csn_folder is not None:
                print(f"\x1b[32;1madding CodeSearchNet data\x1b[0m")
                all_csn_data = []
                for idx in range(5):
                    csn_data = read_jsonl(os.path.join(csn_folder, "train", f"python_train_{idx}.jsonl"))
                    if obf_code: desc = "obfuscating CSN"
                    else: desc = "loading CSN"
                    for rec in tqdm(csn_data, desc=desc):
                        code = remove_comments_and_docstrings(rec["code"])
                        if obf_code: code = obfuscate_code(code)
                        all_csn_data.append({
                            "snippet": code,
                            "intent": rec["docstring"]
                        })
                print(f"\x1b[32;1madded CodeSearchNet data\x1b[0m")
                self.data += all_csn_data
        else:
            self.data = json.load(open(
                os.path.join(
                    folder, f"{split}.json"
                )
            ))
            for i in tqdm(range(len(self.data)), desc="obfuscating code"): 
                # remove comments and docstrings.
                self.data[i]["snippet"] = remove_comments_and_docstrings(self.data[i]["snippet"])
                if obf_code: self.data[i]["snippet"] = obfuscate_code(self.data[i]["snippet"])
            self.queries = {}
            self.doc_ids = defaultdict(lambda:[])
            self.docs = {}
            d_ctr = 0
            q_ctr = 0
            for i, rec in enumerate(self.data):
                q = rec['intent']
                d = rec['snippet']
                if q not in self.queries: 
                    self.queries[q] = q_ctr
                    q_ctr += 1
                if d not in self.docs: 
                    self.docs[d] = d_ctr
                    d_ctr += 1
            for i, rec in enumerate(self.data):
                q = rec['intent']
                d = rec['snippet']
                self.doc_ids[q].append(self.docs[d])
            self.queries = list(self.queries.keys())
            self.docs = list(self.docs.keys())
            self.doc_ids = list(self.doc_ids.values())
            # print(self.doc_ids[-5:])
    def get_doc_ids_mask(self):
        mask = np.zeros((len(self.queries), len(self.docs)))
        for i, ids in enumerate(self.doc_ids):
            for j in ids: mask[i][j] = 1
        return mask

    def get_query_loader(self, batch_size: int=100):
        qset = QueryUniXcoderDataset(
            self.queries, self.doc_ids, 
            self.tokenizer, **self.tok_args,
        )
        q_loader = DataLoader(qset, batch_size=batch_size, shuffle=False)
        return q_loader

    def get_docs_loader(self, batch_size: int=100):
        dset = DocsUniXcoderDataset(
            self.docs, self.tokenizer, 
            **self.tok_args
        )
        d_loader = DataLoader(dset, batch_size=batch_size, shuffle=False)
        return d_loader

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        q = self.data[i]["intent"] # query
        c = self.data[i]["snippet"] # document/code
        q_iids = self.tokenizer(q, **self.tok_args)["input_ids"][0]
        c_iids = self.tokenizer(c, **self.tok_args)["input_ids"][0]
        
        return [q_iids, c_iids]