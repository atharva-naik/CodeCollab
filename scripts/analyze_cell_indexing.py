import json
import tokenize
from typing import *
from tqdm import tqdm
from collections import defaultdict
from datautils.code_cell_analysis import process_nb_cell

def get_tokens(code) -> List[str]:
    tokens = []
    for token in tokenize.generate_tokens(iter(code.splitlines()).__next__):
        token = token[1]
        if token.strip() == "": continue
        tokens.append(token)
    
    return tokens

# main
if __name__ == "__main__":
    code_KB = json.load(open("./JuICe_train_code_KB.json"))
    codes = list(code_KB.keys())
    code_len_dist = defaultdict(lambda:0)
    for code in tqdm(codes):
        code = process_nb_cell(code)
        try: code_len_dist[len(get_tokens(code))] += 1
        except (tokenize.TokenError, IndentationError): continue
    code_len_dist = {k:v for k,v in sorted(code_len_dist.items(), key=lambda x: x[1], reverse=True)}
    print(code_len_dist)