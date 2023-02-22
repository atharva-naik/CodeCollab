import os
import re
import nltk
import spacy
import gensim
import numpy as np
from typing import *
from tqdm import tqdm
from fuzzywuzzy import fuzz
from datautils import read_jsonl
import gensim.corpora as corpora
from nltk.corpus import stopwords
from collections import defaultdict
from gensim.utils import simple_preprocess
from datautils.plan_graph_extraction import get_verb_phrases
from datautils.markdown_cell_analysis import process_markdown
from datautils.code_cell_analysis import ast_parse, get_uniq_vars_and_funcs

# stop words for English.
nltk.download('stopwords')
stop_words = stopwords.words('english')
# stop_words.extend([
#     'data', 'question', 'assignment', 
#     'use', 'using', 'function', 'model',
# ])
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
            non_redundant_kps.append(phrase)
        for word in words: words_seen_till_now.add(word.lower())

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

def analyze_keyphrases_in_markdown(infill, extractor, name="yake"):
    tot, overlap = 0, 0
    for rec in tqdm(infill):
        if name == "yake":
            true_kp = get_keyphrases(rec['true'], extractor)
            pred_kp = get_keyphrases(rec['pred'], extractor)
        elif name == "hf":
            true_kp = list(set(kp.lower() for kp in extractor(rec['true'])))
            pred_kp = list(set(kp.lower() for kp in extractor(rec['pred'])))
        # check for fuzzy string match using fuzzywuzzy.
        for kp_t in true_kp:
            align_scores = [fuzz.token_set_ratio(kp_t, kp_p) for kp_p in pred_kp]
            if len(align_scores) == 0:
                tot += 100
                overlap += 0
                continue
            i = np.argmax(align_scores) # pkp = pred_kp[i]
            tot += 100 # print(kp_t, "||", pred_kp[i])
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
            align_scores = [fuzz.token_set_ratio(vp_t, vp_p) for vp_p in pred_vp]
            if len(align_scores) == 0:
                vp_tot += 100
                vp_overlap += 0
                continue
            i = np.argmax(align_scores)
            vp_tot += 100
            vp_overlap += align_scores[i]            
        for nc_t in true_nc:
            align_scores = [fuzz.token_set_ratio(nc_t, nc_p) for nc_p in pred_nc]
            if len(align_scores) == 0:
                np_tot += 100
                np_overlap += 0
                continue
            i = np.argmax(align_scores)
            np_tot += 100
            np_overlap += align_scores[i]
    print(f"noun chunks fuzzy overlap: {(100*np_overlap/np_tot):.2f}%")
    print(f"verb chunks fuzzy overlap: {(100*vp_overlap/vp_tot):.2f}%")

def sent_to_words(sentences):
    for sentence in sentences:
        # deacc=True removes punctuations
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

def remove_stopwords(texts):
    global stop_words
    return [[word for word in simple_preprocess(str(doc)) 
             if word not in stop_words] for doc in texts]

def get_word_dist(infill) -> Tuple[Dict[str, int], Dict[str, int]]:
    true_dist = defaultdict(lambda:0)
    pred_dist = defaultdict(lambda:0)
    true_sents = [i["true"] for i in infill]
    pred_sents = [i["pred"] for i in infill]
    for words in tqdm(remove_stopwords(list(sent_to_words(true_sents)))): 
        for w in words: true_dist[w] += 1
    for words in tqdm(remove_stopwords(list(sent_to_words(pred_sents)))): 
        for w in words: pred_dist[w] += 1
    true_dist = {k: v for k,v in sorted(true_dist.items(), key=lambda x: x[1], reverse=True)}
    pred_dist = {k: v for k,v in sorted(pred_dist.items(), key=lambda x: x[1], reverse=True)}

    return true_dist, pred_dist

def plot_word_dist(dist: Dict[str, int], path: str, 
                   topk: int=10, color: str="green"):
    import matplotlib.pyplot as plt
    plt.clf()
    x, y, i = [], [], 1
    tot = sum(dist.values())
    labels = []
    for k,v in list(dist.items())[:topk]:
        x.append(i)
        y.append(100*v/tot)
        labels.append(k)
        i += 1
    bar = plt.bar(x, y, color=color)
    for rect in bar:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2, height, 
                 f'{height:.1f}', ha='center', va='bottom')
    plt.xlabel("words")
    plt.ylabel("counts")
    plt.xticks(x, labels=labels, rotation="45")
    plt.title("Word frequency distribution")
    plt.tight_layout()
    plt.savefig(path)

def build_lda_topic_model(insts):
    data = []
    for inst in tqdm(insts):
        for cell in inst["context"]:
            if cell["cell_type"] != "markdown": continue
            proc_markdown = process_markdown(cell["nl_original"])
            data.append(proc_markdown)
    data_words = list(sent_to_words(data))
    data_words = remove_stopwords(data_words)
    id2word = corpora.Dictionary(data_words) 
    corpus = [id2word.doc2bow(text) for text in data_words]
    lda_model = gensim.models.LdaMulticore( 
        corpus=corpus, num_topics=10,
        id2word=id2word,
    )

    return lda_model

def load_as_notebooks(pred_path: str):#, data_path: str):
    # data = read_jsonl(data_path)
    pred = read_jsonl(pred_path)
    notebooks = defaultdict(lambda:[])
    for rec in pred:
        notebooks[rec["id"]].append({
            "true": rec["true"],
            "pred": rec["pred"],
        })

    return notebooks

def count_nbs_with_emb_code(pred_path: str) -> int:
    nbs = load_as_notebooks(pred_path)
    ctr_t = 0
    ctr_p = 0
    for nb in tqdm(nbs.values()):
        found_emb_code = False
        for step in nb:
            if len(get_embedded_code_fragments(step['true'])) > 0:
                found_emb_code = True
                break
        if found_emb_code: ctr_t += 1
        found_emb_code = False
        for step in nb:
            if len(get_embedded_code_fragments(step['pred'])) > 0:
                found_emb_code = True
                break
        if found_emb_code: ctr_p += 1

    return {"true": ctr_t, "pred": ctr_p, "total": len(nbs)}

def print_lda_topics(model):
    topics = model.print_topics()
    for i, topic in enumerate(topics):
        topic_words = [
            i.split("*")[1].strip().replace('"','') for i in topic[1].split(" + ")
        ]
        topic_str = ", ".join(topic_words)
        print(f"T{i+1}", topic_str)
    # for topic in topics:

def analyze_topic_match(infill, insts):
    lda_model = build_lda_topic_model(insts)

# main
if __name__ == "__main__":
    # compare code fragments.
    infill = read_jsonl("./analysis/incoder_markdown_infill_val.jsonl")
    val_insts = read_jsonl("./data/juice-dataset/dev.jsonl")
    ANALYZE_CODE_IN_MARKDOWN = False # mixed: instruction+whole code level
    ANALYZE_KEYPHRASES = True  # local: instruction level.
    use_keyphrase_extractor = ["yake", "hf"][1] # hf is a huggingface based transformer model for keyphrase extraction.
    ANALYZE_VERB_AND_NOUN_PHRASES = False # local: instruction level.
    ANALYZE_INSTRUCTION_TOPICS = False # global: topic model is fit on all the markdown comments.
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
            from datautils.keyphrase_extraction import KeyphraseExtractionPipeline
            model_name = "ml6team/keyphrase-extraction-kbir-inspec"
            extractor = KeyphraseExtractionPipeline(
                model=model_name#, device="cuda",
            )
        analyze_keyphrases_in_markdown(
            infill, extractor, 
            name=use_keyphrase_extractor,
        )
    if ANALYZE_VERB_AND_NOUN_PHRASES:
        analyze_vp_np_overlap(infill)
    if ANALYZE_INSTRUCTION_TOPICS:
        analyze_topic_match(infill, val_insts)