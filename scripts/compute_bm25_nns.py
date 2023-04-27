# compute nearest neighbors per code in the KB 
# using BM25 Lucene Searcher implementation of pyserini
import os
import json
from tqdm import tqdm
from pyserini.search.lucene import LuceneSearcher

code_queries = list(json.load(open("./JuICe_train_code_KB.json")).keys())
code2words = json.load(open("./JuICe_train_code2words.json"))
searcher = LuceneSearcher('./JuICe_train_code_cells_index')
unicode_exceptions_ctr = 0

write_path = "./JuICe_train_code_knn.jsonl"
assert not os.path.exists(write_path), "Aborting to avoid overwrite issues!"
open(write_path, "w")

for docid, q in tqdm(enumerate(code_queries),
                     total=len(code_queries)):
    # if docid < 749199: continue
    # try: 
    # limit clauses to 1024
    query = " ".join(code2words[q].split()[:1024])
    hits = searcher.search(query)
    knn_ids = {
        "docid": docid,
        "matched": [hit.docid for hit in hits if hit.docid != docid]
    }
    # except Exception as e:
        # print(len(code2words[q].split()))
    with open(write_path, "a") as f:
        f.write(json.dumps(knn_ids)+"\n")
print(f"{unicode_exceptions_ctr} unicode errors!")