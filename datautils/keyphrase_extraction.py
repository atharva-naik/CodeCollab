import json
import yake
import spacy
import torch
import textwrap
import fuzzywuzzy
import numpy as np
from typing import *
from tqdm import tqdm
from transformers import (
    TokenClassificationPipeline,
    AutoModelForTokenClassification,
    AutoTokenizer,
)
from fuzzywuzzy import fuzz
import matplotlib.pyplot as plt
from datautils import read_jsonl
from collections import defaultdict
from transformers.pipelines import AggregationStrategy
from datautils.markdown_cell_analysis import process_markdown, get_title_hierarchy_and_stripped_title, extract_notebook_hierarchy

import warnings
with warnings.catch_warnings():
    warnings.simplefilter(action='ignore', category=FutureWarning)
    
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

class EnsembleKeyPhraseExtractor:
    """ensembled keyphrase extractor between YAKE and KBIR."""
    def __init__(self, cache_path: str="./analysis/dev_keyphrases.jsonl",
                 model_name: str="ml6team/keyphrase-extraction-kbir-inspec",
                 yake_strategy: str="topk", yake_k: int=3, **yake_args):
        self.cache_path = cache_path
        self.cached_kps = {rec["markdown"]: rec["keyphrases"] for rec in read_jsonl(cache_path)}
        self.model_name = model_name
        # self.kbir_pipeline = KeyphraseExtractionPipeline(model=model_name)
        # if torch.cuda.is_available():
        self.kbir_pipeline = KeyphraseExtractionPipeline(model=model_name, device=0)
        # else: self.kbir_pipeline = KeyphraseExtractionPipeline(model=model_name)
        self.yake_pipeline = yake.KeywordExtractor(**yake_args)
        self.yake_strategy = yake_strategy
        self.yake_k = yake_k

    def _get_non_redundant_yake_keyphrases(self, markdown: str) -> List[str]:
        proc_markdown = process_markdown(markdown)
        # the returned keywords are ordered in order of decreasing relevance.
        keyphrases = self.yake_pipeline.extract_keywords(proc_markdown)
        # the lower score, the more relevant it is.
        non_redundant_kps = []
        words_seen_till_now = set()
        for phrase, score in keyphrases:
            words = set(phrase.strip().split())
            if len(words.difference(words_seen_till_now)) > 0:
                non_redundant_kps.append(phrase.lower())
            for word in words: words_seen_till_now.add(word.lower())

        return non_redundant_kps

    def _get_topk_yake_keyphrases(self, markdown: str) -> List[str]:
        """get topk yake keyphrases (after lowering the text)"""
        proc_markdown = process_markdown(markdown)
        # the returned keywords are ordered in order of decreasing relevance.
        keyphrases = self.yake_pipeline.extract_keywords(proc_markdown)
        kps = set()
        for phrase, score in keyphrases:
            kps.add(phrase.lower())
            if len(kps) == self.yake_k:
                return sorted(list(kps))
        
        return sorted(list(kps))

    def _get_yake_keyphrases(self, markdown: str) -> List[str]:
        if self.yake_strategy == "non_red": # get non redundant keyphrases.
            return self._get_non_redundant_yake_keyphrases(markdown)
        elif self.yake_strategy == "topk":
            return self._get_topk_yake_keyphrases(markdown)

    def _get_from_cache(self, markdown: str):
        unproc_md_kps = self.cached_kps.get(markdown)
        if unproc_md_kps is None:
            proc_md = process_markdown(markdown)
            proc_md_kps = self.cached_kps.get(proc_md)
            return proc_md_kps

        return unproc_md_kps

    def _generate_keyphrases(self, markdown: str) -> List[str]:
        """generate keyphrases for markdown"""
        kbir_kps = self.kbir_pipeline(process_markdown(markdown))
        if kbir_kps == []:
            return self._get_yake_keyphrases(markdown)
        return kbir_kps

    def __call__(self, markdown: str):
        cached_kps = self._get_from_cache(markdown)
        if cached_kps is None:
            return self._generate_keyphrases(markdown)
        elif cached_kps == []:
            return self._get_yake_keyphrases(markdown)
        return cached_kps

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

def write_train_kps(md_path: str="./data/juice-dataset/train_uniq_mds.jsonl", save_path: str="./data/juice-dataset/train_uniq_mds_with_kps.jsonl"):
    # avoid overwrite
    import os
    import json
    from tqdm import tqdm

    assert not os.path.exists(save_path)
    ext = EnsembleKeyPhraseExtractor()

    # reset/create file
    open(save_path, "w")
    with open(md_path) as f:
        for line in tqdm(f):
            rec = json.loads(line.strip())
            rec["kps"] = list(set(kp.lower() for kp in ext(rec["md"])))
            with open(save_path, "a") as g:
                g.write(json.dumps(rec)+"\n")

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

def analyze_markdown_using_kps(nlp, markdown: str, kps: List[str]) -> List[str]:
    proc_md = process_markdown(markdown)
    kps = [kp.lower() for kp in kps]
    sents = list(nlp(proc_md).sents)
    insts = []
    for i, sent in enumerate(sents):
        sent = sent.as_doc()
        # print(f"sent-{i}: {sent}\n")
        root_to_nc = {nc.root.text: nc.text for nc in sent.noun_chunks}
        for nc in sent.noun_chunks:
            not_matched_with_kp = True
            for kp in kps:
                if fuzz.partial_ratio(nc.text.lower(), kp) >= 90:
                    not_matched_with_kp = False
                    break
            if not_matched_with_kp: continue
            # print(f"\tkp: {kp}({nc.text})")
            func_call = " ".join([a.text.lower() if a.pos_ != "NOUN" else " ".join(root_to_nc[a.text].lower().split()) for a in list(nc.root.ancestors)[::-1]])
            # insts.append("\x1b[34m"+func_call+f"\x1b[0m\x1b[1m(\x1b[0m\x1b[32m{kp}\x1b[0m[\"{nc.text}\"]\x1b[1m)\x1b[0m")
            # insts.append(func_call+f"({kp}[\"{nc.text}\"])")
            insts.append(func_call+f" {kp}(\"{nc.text}\")")
            # for a in nc.root.ancestors:
            #     a_text = a.text if a.pos_ != "NOUN" else root_to_nc[a.text]
            #     print(f"\t\t{a.pos_}: {a_text}")
    return insts

def print_plan(node: dict, ext: EnsembleKeyPhraseExtractor, path_from_root: str, nlp):
    value = node['value']
    print("")
    print(path_from_root+f"{value['id']}")
    if value['type'] == "root":
        print("ROOT(0):")
    elif value["type"] == "markdown":
        content = value["content"]
        kps = ext(content)
        insts = analyze_markdown_using_kps(nlp, content, kps)
        print("\n".join(insts))
    elif value["type"] == "code":
        print(value['content'])
    print("")
    for child in node['children']:
        print_plan(child, ext, path_from_root+f"{node['value']['id']}->", nlp)

def accum_keyphrases_v3(nb_kps_with_hier: Dict[str, Any], 
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
    # for key, kp_dist in accum_kps.items():
    #     accum_kps[key] = {k:v for k,v in sorted(kp_dist.items(), reverse=True, key=lambda x:x[1]) if k not in very_common_phrases}
    pair_counts = defaultdict(lambda: defaultdict(lambda:0))
    for nb in nb_kps_with_hier.values():
        for i in range(len(nb)-1):
            for j in range(i+1, len(nb)):
                kps1 = nb[i]["keyphrases"]
                kps2 = nb[j]["keyphrases"]
                for kp1 in kps1:
                    for kp2 in kps2:
                        if kp1 in very_common_phrases: continue
                        if kp2 in very_common_phrases: continue
                        if nb[i][key] < nb[j][key]:
                            pair_counts[kp1][kp2] += 1
    for parent in pair_counts:
        pair_counts[parent] = {
            k: v for k,v in sorted(
                pair_counts[parent].items(), 
                reverse=True, key=lambda x: x[1]
            )}
    pair_counts = {k: v for k,v in sorted(pair_counts.items(), reverse=True, key=lambda x: sum(x[1].values()))}

    return pair_counts

def plot_child_dist(parent: str, pair_counts: Dict[str, int], topk: int=8, color: str="red"):
    dist = pair_counts[parent]
    plt.clf()
    plt.title(f"Most frequent keyphrase children of {parent}")
    xlabels = ["\n".join(textwrap.wrap(k, width=12)) for k in list(dist.keys())[:topk]]
    y = list(dist.values())[:topk]
    x = range(1, topk+1)
    bar = plt.bar(x, y, color=color)
    for rect in bar:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2.0, height, 
                 f'{height:.0f}', ha='center', va='bottom')
    plt.xticks(x, labels=xlabels, rotation=90)
    plt.tight_layout()
    plt.savefig(f"./plots/{parent}_child_counts.png")

def present_plan_for_inst(inst: dict, ext: EnsembleKeyPhraseExtractor, nlp):
    # id_to_kps = {rec["md_id"]: rec[""] for rec in kps_dict}
    root, nodes = extract_notebook_hierarchy(inst)
    root = root.serialize()
    print_plan(root, ext, "", nlp)

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