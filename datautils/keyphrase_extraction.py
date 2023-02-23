import json
import numpy as np
from typing import *
from tqdm import tqdm
from transformers import (
    TokenClassificationPipeline,
    AutoModelForTokenClassification,
    AutoTokenizer,
)
import matplotlib.pyplot as plt
from collections import defaultdict
from transformers.pipelines import AggregationStrategy
from datautils.markdown_cell_analysis import process_markdown, get_title_hierarchy_and_stripped_title

# Define keyphrase extraction pipeline
class KeyphraseExtractionPipeline(TokenClassificationPipeline):
    def __init__(self, model, *args, **kwargs):
        super().__init__(
            model=AutoModelForTokenClassification.from_pretrained(model),
            tokenizer=AutoTokenizer.from_pretrained(model),
            *args,
            **kwargs
        )

    def postprocess(self, model_outputs):
        results = super().postprocess(
            model_outputs=model_outputs,
            aggregation_strategy=AggregationStrategy.SIMPLE,
        )
        return np.unique([result.get("word").strip() for result in results])

def load_keyphrases(path: str) -> Dict[str, List[dict]]:
    nb_wise_kps = defaultdict(lambda:[])
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            rec = json.loads(line)
            nb_wise_kps[rec["nb_id"]].append(rec)

    return nb_wise_kps

def load_keyphrases_with_title_hier(path: str):
    nb_wise_kps = defaultdict(lambda:[])
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            rec = json.loads(line)
            rec["hier"], _ = get_title_hierarchy_and_stripped_title(rec["markdown"])
            nb_wise_kps[rec["nb_id"]].append(rec)
    for nb_id, cells in nb_wise_kps.items():
        hiers_found = sorted(set(
            i['hier'] for i in cells
        ))
        for i in range(len(cells)):
            cells[i]['rank'] = hiers_found.index(cells[i]['hier'])

    return nb_wise_kps

def accum_keyphrases(nb_kps_with_hier: Dict[str, Any]) -> Dict[int, Dict[str, int]]:
    accum_kps = defaultdict(lambda: defaultdict(lambda:0))
    for nb in nb_kps_with_hier.values():
        for cell in nb:
            for kp in cell["keyphrases"]:
                accum_kps[cell["rank"]][kp] += 1
    for rank, kp_dist in accum_kps.items():
        accum_kps[rank] = {k:v for k,v in sorted(kp_dist.items(), reverse=True, key=lambda x:x[1])}

    return accum_kps

def accum_keyphrases_v2(nb_kps_with_hier: Dict[str, Any], 
                        rel_rank: bool=True, trunc_rank: int=5) -> Dict[int, Dict[str, int]]:
    accum_kps = defaultdict(lambda: defaultdict(lambda:0))
    very_common_phrases = None
    key = "rank" if rel_rank else "hier"
    for nb in nb_kps_with_hier.values():
        for cell in nb:
            for kp in cell["keyphrases"]:
                accum_kps[cell[key]][kp] += 1
    for rank, kp_dist in accum_kps.items():
        if rank > trunc_rank: continue
        rank_kps = set(k for k in kp_dist)
        if very_common_phrases is None:
            very_common_phrases = rank_kps
        else:
            very_common_phrases = very_common_phrases.intersection(rank_kps)
    print(f"{len(very_common_phrases)} very common keyphrases (upto rank {trunc_rank})")
    for key, kp_dist in accum_kps.items():
        accum_kps[key] = {k:v for k,v in sorted(kp_dist.items(), reverse=True, key=lambda x:x[1]) if k not in very_common_phrases}

    return accum_kps

def extract_keyphrases_for_val_data(data: List[dict], extractor, path: str) -> List[dict]:
    extracted_keyphrases = []
    pbar = tqdm(data)
    open(path, "w")
    nb_id = 0
    for rec in pbar:
        ctxt = rec["context"][::-1]
        inst_extracted = []
        for i, cell in enumerate(ctxt):
            pbar.set_description(f"{i}/{len(ctxt)}")
            if cell["cell_type"] == "markdown":
                proc_md_cell = process_markdown(cell["nl_original"])
                keyphrases = extractor(proc_md_cell.replace("\n", " "))
                inst = {
                    "nb_id": nb_id,
                    "md_id": i, 
                    "markdown": cell["nl_original"],
                    "keyphrases": list(set(kp.lower() for kp in keyphrases.tolist())),
                } # print(keyphrases)
                with open(path, "a") as f: f.write(json.dumps(inst)+"\n")
                inst_extracted.append(inst)
        extracted_keyphrases.append(inst_extracted)
        nb_id += 1

    return extracted_keyphrases

def plot_uniq_kps_per_hier(rank_wise_kp_dist: Dict[int, Dict[str, int]],
                           path: str="./plots/unique_keyphrases_vs_hier.png"):
    x = range(len(rank_wise_kp_dist))
    y = [len(rank_wise_kp_dist[i]) for i in x]
    plt.clf()
    plt.title("Number of unique key phrases at each level of hierarchy")
    plt.ylabel("No. of unique keyphrases")
    plt.xlabel("Relative hierarchy level")
    bar = plt.bar(x, y, color="red")
    for rect in bar:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2.0, height, 
                 f'{height:.0f}', ha='center', va='bottom')
    # plt.xticks(x, labels=list(dist.keys())[:topk], rotation=90)
    plt.tight_layout()
    plt.savefig(path)

def plot_total_kps_per_hier(rank_wise_kp_dist: Dict[int, Dict[str, int]],
                            path: str="./plots/total_keyphrases_vs_hier.png"):
    x = range(len(rank_wise_kp_dist))
    y = [sum(rank_wise_kp_dist[i].values()) for i in x]
    plt.clf()
    plt.title("Total key phrases at each level of hierarchy")
    plt.ylabel("Total keyphrases")
    plt.xlabel("Relative hierarchy level")
    bar = plt.bar(x, y, color="green")
    for rect in bar:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2.0, height, 
                 f'{height:.0f}', ha='center', va='bottom')
    # plt.xticks(x, labels=list(dist.keys())[:topk], rotation=90)
    plt.tight_layout()
    plt.savefig(path)

# main
if __name__ == "__main__":
    model_name = "ml6team/keyphrase-extraction-kbir-inspec"
    extractor = KeyphraseExtractionPipeline(model=model_name)
#     text = """
# Keyphrase extraction is a technique in text analysis where you extract the
# important keyphrases from a document. Thanks to these keyphrases humans can
# understand the content of a text very quickly and easily without reading it
# completely. Keyphrase extraction was first done primarily by human annotators,
# who read the text in detail and then wrote down the most important keyphrases.
# The disadvantage is that if you work with a lot of documents, this process
# can take a lot of time. 

# Here is where Artificial Intelligence comes in. Currently, classical machine
# learning methods, that use statistical and linguistic features, are widely used
# for the extraction process. Now with deep learning, it is possible to capture
# the semantic meaning of a text even better than these classical methods.
# Classical methods look at the frequency, occurrence and order of words
# in the text, whereas these neural approaches can capture long-term
# semantic dependencies and context of words in a text.
# """
    # keyphrases = extractor(text.replace("\n", " "))
    # print(keyphrases)
    from datautils import read_jsonl
    val_data = read_jsonl("./data/juice-dataset/dev.jsonl")
    val_keyphrases = extract_keyphrases_for_val_data(
        val_data, extractor=extractor, 
        path="./analysis/dev_keyphrases.jsonl",
    )