# code to encode and represent 
import os
import json
import torch
from typing import *
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

def test_sbert_encoder(candidate_phrases: List[List[str]], all_node_names: List[str], k: int=1):
    sbert = SentenceTransformer("multi-qa-mpnet-base-dot-v1")
    # use cuda if possible.
    if torch.cuda.is_available(): sbert.cuda()
    # encode node names.
    node_enc = sbert.encode(all_node_names, show_progress_bar=True, batch_size=128, device="cuda:0", convert_to_tensor=True)
    entity_preds = []
    for inst_phrases in tqdm(candidate_phrases):
        phrase_enc = sbert.encode(inst_phrases, device="cuda:0", convert_to_tensor=True)
        node_ids = torch.topk(util.cos_sim(phrase_enc, node_enc), axis=-1, k=1).indices
        cand_preds = []
        for i, cand_pred in enumerate(node_ids):
            if k == 1:
                cand_pred = all_node_names[node_ids[i][0].item()]
                cand_preds.append(cand_pred)
        entity_preds.append(cand_preds)

    return entity_preds

# main
if __name__ == "__main__":
    # load all the nodes in the KB.
    all_nodes = json.load(open("./data/DS_KB/all_nodes.json"))
    # encode the nodes in the KB.
    all_node_names = list(all_nodes.keys())
    print(test_sbert_encoder(["GPT-4.5 LLM"], all_node_names))