import os
import re
import spacy
import numpy as np
from typing import *
from tqdm import tqdm
from fuzzywuzzy import fuzz
from datautils import read_jsonl
from datautils.plan_graph_extraction import get_verb_phrases
from datautils.code_cell_analysis import ast_parse, get_uniq_vars_and_funcs

def process_markdown(markdown: str):
    """strip formatting and syntax cues from markdown."""
    markdown = markdown.replace("#", " ").replace("`", " ").replace("*", " ")
    markdown = " ".join(markdown.split())
    
    return markdown

def get_code_entity_mentions(markdown: str, inst: dict):
    imports = inst["imports"]
    func_names = set() 
    var_names = set()
    for cell in inst["context"]:
        if cell["cell_type"] == "code":
            try: root = ast_parse(cell["code"])
            except SyntaxError: continue
            if root is None: continue
            ret = get_uniq_vars_and_funcs(
                cell_code_ast_root=root, 
                imported_module_names=imports
            )
            for var in ret["vars"]: var_names.add(var)
            for func in ret["func"]: func_names.add(func)

    var_names = sorted(list(var_names))
    func_names = sorted(list(func_names))
    proc_markdown = process_markdown(markdown)
    entities = {"vars": set(), "func": set()}
    for term in proc_markdown.split():
        term = term.strip().split("(")[0].strip()
        for var in var_names:
            if term == var: entities["vars"].add(var)
        for func in func_names:
            if term == func: entities["func"].add(func)

    return entities 

def get_embedded_code_fragments(markdown: str):
    """given unprocessed markdown (which has intact syntax cues)"""
    return [ent.strip() for ent in re.findall('`([^`]*)`', markdown) if ent.strip() != ""]

def get_keyphrases(markdown: str, extractor=None):
    if extractor is None: extractor = yake.KeywordExtractor()
    proc_markdown = process_markdown(markdown)
    # the returned keywords are ordered in order of decreasing relevance.
    keyphrases = extractor.extract_keywords(proc_markdown)
    # the lower score, the more relevant it is.
    non_redundant_kps = []
    words_seen_till_now = set()
    for phrase, score in keyphrases:
        words = set(phrase.strip().split())
        if len(words.difference(words_seen_till_now)) > 0:
            non_redundant_kps.append((phrase, score))
        for word in words: words_seen_till_now.add(word)

    return non_redundant_kps

def analyze_code_in_markdown(infill, val_insts):
    frag_overlap, frag_tot = 0, 0
    vars_overlap, vars_tot = 0, 0
    func_overlap, func_tot = 0, 0
    
    for rec in tqdm(infill):
        id = rec["id"]
        true = rec["true"]
        pred = rec["pred"]
        inst = val_insts[id]
        
        true_code_frags = set(get_embedded_code_fragments(true))
        pred_code_frags = set(get_embedded_code_fragments(pred))
        true_ents = get_code_entity_mentions(true, inst)
        pred_ents = get_code_entity_mentions(pred, inst)
        
        vars_overlap += len(true_ents['vars'].intersection(pred_ents['vars']))
        func_overlap += len(true_ents['vars'].intersection(pred_ents['vars']))
        frag_overlap += len(true_code_frags.intersection(pred_code_frags))

        frag_tot += len(true_code_frags)
        vars_tot += len(true_ents['vars'])
        func_tot += len(true_ents['func'])

    print(f"code fragment overlap: {(100*frag_overlap/frag_tot):.2f}%")
    print(f"func entities overlap: {(100*func_overlap/func_tot):.2f}%")        
    print(f"vars entities overlap: {(100*vars_overlap/vars_tot):.2f}%")

def analyze_keyphrases_in_markdown(infill, extractor):
    tot, overlap = 0, 0
    for rec in tqdm(infill):
        true_kp = get_keyphrases(rec['true'], extractor)
        pred_kp = get_keyphrases(rec['pred'], extractor)
        # check for fuzzy string match using fuzzywuzzy.
        for kp_t in true_kp:
            align_scores = [fuzz.token_sort_ratio(kp_t, kp_p) for kp_p in pred_kp]
            if len(align_scores) == 0:
                tot += 100
                overlap += 0
                continue
            i = np.argmax(align_scores) # pkp = pred_kp[i]
            tot += 100
            overlap += align_scores[i]
    print(f"keyphrase fuzzy overlap: {(100*overlap/tot):.2f}%")

def analyze_vp_np_overlap(infill):
    nlp = spacy.load("en_core_web_lg")
    np_tot, np_overlap = 0, 0
    vp_tot, vp_overlap = 0, 0
    for rec in tqdm(infill):
        true = process_markdown(rec['true'])
        pred = process_markdown(rec['pred'])
        true_nc = list(nlp(true).noun_chunks)
        pred_nc = list(nlp(pred).noun_chunks)
        true_vp = get_verb_phrases(true, nlp)
        pred_vp = get_verb_phrases(pred, nlp)
        for vp_t in true_vp:
            align_scores = [fuzz.token_sort_ratio(vp_t, vp_p) for vp_p in pred_vp]
            if len(align_scores) == 0:
                vp_tot += 100
                vp_overlap += 0
                continue
            i = np.argmax(align_scores)
            vp_tot += 100
            vp_overlap += align_scores[i]            
        for nc_t in true_nc:
            align_scores = [fuzz.token_sort_ratio(nc_t, nc_p) for nc_p in pred_nc]
            if len(align_scores) == 0:
                np_tot += 100
                np_overlap += 0
                continue
            i = np.argmax(align_scores)
            np_tot += 100
            np_overlap += align_scores[i]
    print(f"noun chunks fuzzy overlap: {(100*np_overlap/np_tot):.2f}%")
    print(f"verb chunks fuzzy overlap: {(100*vp_overlap/vp_tot):.2f}%")

# main
if __name__ == "__main__":
    # compare code fragments.
    infill = read_jsonl("./analysis/incoder_markdown_infill_val.jsonl")
    val_insts = read_jsonl("./data/juice-dataset/dev.jsonl")
    ANALYZE_CODE_IN_MARKDOWN = False # mixed: instruction+whole code level
    ANALYZE_KEYPHRASES = False  # local: instruction level.
    use_keyphrase_extractor = ["yake", "hf"][0] # hf is a huggingface based transformer model for keyphrase extraction.
    ANALYZE_VERB_AND_NOUN_PHRASES = True # local: instruction level.
    ANALYZE_INSTRUCTION_TOPICS = False # global: topic model is fit on all the markdown comments.s
    if ANALYZE_CODE_IN_MARKDOWN:
        # analyze the 
        # 1) embedded code and 
        # 2) code fragments 
        # in markdown.
        analyze_code_in_markdown(infill, val_insts)
    if ANALYZE_KEYPHRASES:
        if use_keyphrase_extractor == "yake":
            import yake
            extractor = yake.KeywordExtractor()
        elif use_keyphrase_extractor == "hf":
            pass
        analyze_keyphrases_in_markdown(infill, extractor)
    if ANALYZE_VERB_AND_NOUN_PHRASES:
        analyze_vp_np_overlap(infill)