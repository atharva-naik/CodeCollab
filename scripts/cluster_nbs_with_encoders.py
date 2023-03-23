#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import *
from datautils.markdown_cell_analysis import process_markdown

# use text encoder from CoNaLa to represent markdown 
# use code encoder from CoNaLa to represent code

def get_cell_seq(inst: dict) -> List[Tuple[str, str]]:
    seq = []
    for cell in inst["context"][::-1]:
        cell_type = cell['cell_type']
        if cell_type == "markdown":
            seq.append((process_markdown(cell["nl_original"]), cell_type))
        elif cell_type == "code":
            seq.append((cell['code'], cell_type))
    seq.append((inst['code'], "code"))
    
    return seq
    
def find_nearest_neighbors(data: List[dict], bi_encoder, tokenizer):
    for inst in data:
        inst_cell_seq = get_cell_seq(inst)
        for content, dtype in inst_cell_seq:
            tokenizer()
            bi_encoder.encode()

