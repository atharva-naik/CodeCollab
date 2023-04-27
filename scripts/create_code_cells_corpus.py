# create a corpus.jsonl file for the dataset to create inverted index for JuICe code KG (code cells to path mapping).
import os
import json
from tqdm import tqdm
from datautils import read_jsonl

# main
if __name__ == "__main__":
    code_cell_to_code_bow = json.load(open("./JuICe_train_code2words_new.json"))
    code_KG = json.load(open("./JuICe_train_code_KB.json"))
    corpus_dump_path = "./JuICe_train_codebm25_indices/uniq_vars_all_funcs/corpus.jsonl"
    assert corpus_dump_path.endswith(".jsonl"), "incorrect filename"
    with open(corpus_dump_path, "w") as f:
        for i, code in tqdm(enumerate(code_KG)):
            f.write(json.dumps({
                "id": i,
                "contents": code_cell_to_code_bow[code],
            })+"\n")