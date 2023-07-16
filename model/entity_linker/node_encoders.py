# code to encode and represent 
import os
import json
import torch
from typing import *
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

def test_sbert_encoder(candidate_phrases: List[List[str]], 
                       all_node_names: List[str], k: int=1,
                       unlinkable_threshold: float=0):
    sbert = SentenceTransformer("multi-qa-mpnet-base-dot-v1")
    # use cuda if possible.
    if torch.cuda.is_available(): sbert.cuda()
    # encode node names.
    node_enc = sbert.encode(all_node_names, show_progress_bar=True, batch_size=128, device="cuda:0", convert_to_tensor=True)
    entity_preds = []
    for inst_phrases in tqdm(candidate_phrases):
        phrase_enc = sbert.encode(inst_phrases, device="cuda:0", convert_to_tensor=True)
        linkage_results = torch.topk(util.cos_sim(phrase_enc, node_enc), axis=-1, k=1)
        linkage_scores = linkage_results.values
        linkage_indices = linkage_results.indices
        cand_preds = []
        for i, cand_pred in enumerate(linkage_indices):
            if k == 1:
                cand_pred = all_node_names[linkage_indices[i][0].item()]
                if linkage_scores[i][0].item() < unlinkable_threshold: continue
                cand_preds.append((inst_phrases[i], linkage_scores[i][0].item(), cand_pred))
        entity_preds.append(cand_preds)

    return entity_preds

# main
if __name__ == "__main__":
    # load all the nodes in the KB.
    all_nodes = json.load(open("./data/DS_KB/all_nodes.json"))
    # encode the nodes in the KB.
    all_node_names = list(all_nodes.keys())

    from datautils import read_jsonl
    from model.entity_linker.candidate_phrase_extractor import MarkdownPhraseExtractor

    md_ext = MarkdownPhraseExtractor()
    nbs = read_jsonl("./data/juice-dataset/dev.jsonl")
    for cell in nbs[0]["context"]:
        if cell["cell_type"] == "markdown": break
    phrases = md_ext(cell["nl_original"], k=0)
    print(cell["nl_original"])
    print(json.dumps(test_sbert_encoder([phrases], all_node_names), indent=4))