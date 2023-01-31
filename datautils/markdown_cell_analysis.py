import json
import nltk
import gensim
import pprint
import numpy as np
from typing import *
import gensim.corpora as corpora
from nltk.corpus import stopwords
from collections import defaultdict
from gensim.utils import simple_preprocess
from datautils.plan_graph_extraction import get_noun_phrases, get_verb_phrases

# download stopwords
nltk.download('stopwords')

# import copy
def get_title_hierarchy_and_stripped_title(title: str):
    ctr = 0
    for char in title:
        if char == "#": ctr += 1
    return ctr, title[ctr:].strip()

def extract_title_phrases(data: List[dict], model, path: str):
    titles = defaultdict(lambda:[])
    for rec in data.values():
        for cell in rec["context"]:
            if cell["cell_type"] != "markdown": continue
            for line in cell["nl_original"].split("\n"):
                line = line.strip()
                if not(line.startswith("#")): continue
                level, stripped_title = get_title_hierarchy_and_stripped_title(line)
                verb_phrases = get_verb_phrases(stripped_title, model)
                noun_phrases = get_noun_phrases(stripped_title, model)
                titles[level].append({
                    "verbs": verb_phrases,
                    "nouns": noun_phrases,
                    "title": stripped_title,
                })
    with open(path, "w") as f:
        json.dump(titles, f, indent=4)

def sent_to_words(sentences):    
    for sentence in sentences:        
        # deacc=True removes punctuations        
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

stop_words = stopwords.words('english')
def remove_stopwords(texts):
    global stop_words
    return [[word for word in simple_preprocess(str(doc)) 
             if word not in stop_words] for doc in texts]

def get_lda_topics(data: List[str], num_topics: int=10):
    data_words = remove_stopwords(list(sent_to_words(data)))
    id2word = corpora.Dictionary(data_words)
    texts = data_words
    corpus = []
    for text in texts:
        corpus.append(id2word.doc2bow(text))
    lda_model = gensim.models.LdaMulticore(
        corpus=corpus, 
        id2word=id2word,
        num_topics=num_topics,
    )
    pprint.pprint(lda_model.print_topics())

def cluster_titles(data: List[dict], model, path: str):
    titles = defaultdict(lambda:[])
    for rec in data.values():
        for cell in rec["context"]:
            if cell["cell_type"] != "markdown": continue
            for line in cell["nl_original"].split("\n"):
                line = line.strip()
                if not(line.startswith("#")): continue
                level, stripped_title = get_title_hierarchy_and_stripped_title(line)
                titles[level].append(stripped_title)
    with open(path, "w") as f:
        json.dump(titles, f, indent=4)

def compute_interleave_stats(data: List[dict]):
    seq_lens = {"code": [], "markdown": [], "raw": []}
    switches = [] # number of switches.
    for rec in data.values():
        prev_cell_type = None
        cell_types = [cell["cell_type"] for cell in rec["context"][::-1]]+["code"]
        num_switches = -1
        for cell_type in cell_types:
            if cell_type != prev_cell_type:
                seq_lens[cell_type].append(1)
                num_switches += 1
            else:
                seq_lens[cell_type][-1] += 1
            prev_cell_type = cell_type
        switches.append(num_switches)
    # print(np.sum(seq_lens["code"]))
    # print(len(seq_lens["code"]))
    return {
        "avg. code seq len": round(np.mean(seq_lens["code"]), 3),
        "avg. markdown seq len": round(np.mean(seq_lens["markdown"]), 3),
        "avg. no. of transitions": round(np.mean(switches), 3),
        "max code seq len": np.max(seq_lens["code"]),
        "max markdown seq len": np.max(seq_lens["markdown"]),
        "max no. of transitions": np.max(switches),
    }
# main method.
if __name__ == "__main__":
    d = json.load(open(""))
    titles = cluster_titles(d)
    print(titles)