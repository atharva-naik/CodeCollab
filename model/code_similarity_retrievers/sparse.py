#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# sparse retrievers.

import os
import ast
import json
import torch
import difflib
import numpy as np
from typing import *
from tqdm import tqdm
import torch.nn as nn
from scipy import sparse
from collections import defaultdict
from datautils import camel_case_split
from sentence_transformers import util
from pyserini.search.lucene import LuceneSearcher
from datautils.code_cell_analysis import process_nb_cell
from sklearn.feature_extraction.text import TfidfVectorizer
from datautils.code_cell_analysis import make_ast_parse_safe

# distribution of code constructs for code_KB (derived from JuICe train).
CODE_CONSTRUCTS_DIST = {'Import': 764428, 'alias': 1586296, 'Assign': 6982519, 'Expr': 5137624, 'Name': 35679408, 'Subscript': 4319724, 'Call': 12852409, 'Store': 9165559, 'Attribute': 11372132, 'Tuple': 1834522, 'Load': 45503167, 'keyword': 4491816, 'BinOp': 2715267, 'Constant': 19453121, 'Slice': 682666, 'Lambda': 104939, 'BitAnd': 61543, 'Compare': 879097, 'arguments': 594105, 'NotEq': 57154, 'arg': 1182530, 'For': 545303, 'Div': 419195, 'Mult': 662631, 'BitOr': 16356, 'Eq': 434266, 'List': 1465113, 'Sub': 411687, 'Lt': 90538, 'ImportFrom': 632317, 'Add': 971785, 'UnaryOp': 539907, 'USub': 490585, 'ListComp': 205166, 'comprehension': 241756, 'Pass': 14004, 'With': 72107, 'withitem': 73240, 'DictComp': 12493, 'Dict': 278673, 'Starred': 21026, 'While': 23934, 'AugAssign': 134692, 'IfExp': 30831, 'Mod': 146671, 'FloorDiv': 19586, 'GeneratorExp': 16281, 'FunctionDef': 488896, 'Return': 422148, 'BoolOp': 56704, 'And': 38276, 'Not': 32737, 'If': 425762, 'In': 60565, 'Pow': 131337, 'Gt': 113859, 'MatMult': 6144, 'Try': 27157, 'ExceptHandler': 27576, 'Or': 18428, 'Continue': 12504, 'IsNot': 11009, 'ClassDef': 26983, 'Yield': 6588, 'LtE': 37009, 'NotIn': 22179, 'Raise': 10233, 'Set': 5434, 'Delete': 19358, 'Del': 23199, 'SetComp': 1162, 'BitXor': 1142, 'GtE': 41384, 'AsyncFunctionDef': 270, 'AsyncWith': 85, 'Await': 305, 'Is': 15844, 'Assert': 23996, 'Break': 15160, 'RShift': 1178, 'Global': 6350, 'JoinedStr': 19973, 'FormattedValue': 24022, 'LShift': 704, 'AnnAssign': 561, 'AsyncFor': 25, 'YieldFrom': 364, 'UAdd': 2882, 'Invert': 13703, 'Nonlocal': 122}

ACCEPTABLE_CONSTRUCTS = ['Import', 'keyword', 'Slice', 'Lambda', 'arguments', 'arg', 'For', 'ImportFrom', 'ListComp', 'comprehension', 'With', 'withitem', 'DictComp', 'While', 'IfExp', 'GeneratorExp', 'FunctionDef', 'Return', 'If', 'MatMult', 'Try', 'ExceptHandler', 'Continue', 'IsNot', 'ClassDef', 'Yield', 'Raise', 'Delete', 'Del', 'SetComp', 'AsyncFunctionDef', 'AsyncWith', 'Await', 'Assert', 'Break', 'JoinedStr', 'AsyncFor', 'YieldFrom']
ACCEPTABLE_CONSTRUCTS = {a:True for a in ACCEPTABLE_CONSTRUCTS}
# @make_ast_parse_safe
# def get_code_topics(code: str):
#     """extract unique variable, function names and aliases"""
#     topic_terms = set()
#     for node in ast.walk(ast.parse(code)):
#         if isinstance(node, ast.Name): name = ast.unparse(node)
#         elif isinstance(node, ast.Call): name = ast.unparse(node.func)
#         elif isinstance(node, ast.alias):
#             if node.asname is not None:
#                 name = node.name + " " + node.asname
#             else: name = node.name
#         else: continue
#         for dot_split_term in name.split("."): # split by dots first.        
#             for underscore_split_term in dot_split_term.split("_"): # split by underscore second.
#                 for term in camel_case_split(underscore_split_term): # split by camel case finally.
#                     topic_terms = topic_terms.union(term.lower().split())
    
#     return " ".join(sorted(list(topic_terms)))

# @make_ast_parse_safe
# def get_code_ast_seq(code: str):
#     return " ".join([node.__class__.__name__ for node in ast.walk(ast.parse(code))][1:])
class CustomDiffer(difflib.Differ):
    def compare(self, code1: str, code2: str):
        code1 = ast.unparse(ast.parse(process_nb_cell(code1)))
        code2 = ast.unparse(ast.parse(process_nb_cell(code2)))
        code1 = code1.split("\n")
        code2 = code2.split("\n")
        return super().compare(code1, code2)

@make_ast_parse_safe
def joint_get_code_topics_and_ast_seq(code: str) -> Tuple[str, str]:
    tree = ast.parse(code)
    topic_terms = set()
    func_names_and_constructs = set()
    for node in ast.walk(tree):
        is_call = False
        node_name = node.__class__.__name__
        if ACCEPTABLE_CONSTRUCTS.get(node_name, False):
            func_names_and_constructs.add(node_name.lower())
        if isinstance(node, ast.Name): name = ast.unparse(node)
        elif isinstance(node, ast.Call): 
            is_call = True
            name = ast.unparse(node.func)
        elif isinstance(node, ast.alias):
            if node.asname is not None:
                name = node.name + " " + node.asname
            else: name = node.name
        else: continue
        for dot_split_term in name.split("."): # split by dots first.        
            for underscore_split_term in dot_split_term.split("_"): # split by underscore second.
                for term in camel_case_split(underscore_split_term): # split by camel case finally.
                    topic_terms = topic_terms.union(set(term.lower().split()))
                    if is_call: func_names_and_constructs = func_names_and_constructs.union(set(term.lower().split()))
    code_topics = " ".join(sorted(list(topic_terms)))
    # ast_constructs = defaultdict(lambda:0)
    # ast_seq = [node.__class__.__name__ for node in ast.walk(tree)][1:]
    # for construct in ast_seq: 
    #     ast_constructs[construct] += 1
    # Z = min(list(ast_constructs.values()))
    # normed_ast_seq = []
    # for k,v in ast_constructs.items():
    #     for _ in range(v // Z): normed_ast_seq.append(k)
    # ast_seq = " ".join(normed_ast_seq)
    return code_topics, " ".join(list(func_names_and_constructs))

# a sparse feature based retriever for matching up.
class SparseRetriever:
    def __init__(self):
        self.inverted_index = {}

    def build_inverted_index(self):
        pass

# index documents to create inverted index for CodeBM25
class CodeBM25SparseIndexer:
    """Given a set of code texts,
    extract ast structure/statement
    sequence and code topics (unique)
    variable, alias and function terms."""
    def __init__(self, codes: List[str], 
                 ast_save_path: str="./codebm25_indices/JuICe_train/code_asts",
                 topics_save_path: str="./codebm25_indices/JuICe_train/code_topics"):
        """create the corpus files if they don't exist"""
        self.ast_save_path = ast_save_path
        self.topics_save_path = topics_save_path
        os.makedirs(ast_save_path, exist_ok=True)
        os.makedirs(topics_save_path, exist_ok=True)
        ast_jsonl_path = os.path.join(ast_save_path, "corpus.jsonl")
        topics_jsonl_path = os.path.join(topics_save_path, "corpus.jsonl")
        if os.path.exists(ast_jsonl_path): return
        if os.path.exists(topics_jsonl_path): return
        open(ast_jsonl_path, "w")
        open(topics_jsonl_path, "w")
        for i, code in tqdm(enumerate(codes), total=len(codes)):
            try: code_topics, ast_seq = joint_get_code_topics_and_ast_seq(code)
            except ValueError: code_topics, ast_seq = "", ""
            with open(ast_jsonl_path, 'a') as f:
                f.write(json.dumps({"id": i, "contents": ast_seq})+"\n")
            with open(topics_jsonl_path, 'a') as f:
                f.write(json.dumps({"id": i, "contents": code_topics})+"\n")

    def createIndex(self):
        """invoke bash scripts to make pyserini index"""
        os.system("chmod +x scripts/create_bm25_indices.sh")
        os.system("scripts/create_bm25_indices.sh")

# searcher for CodeBM25 sparse retriever that combines both structure (serialized AST) and topic info.
class EnsembleCodeBM25Searcher:
    def __init__(self, ast_index_path: str="./codebm25_indices/JuICe_train/code_asts",
                 topics_index_path: str="./codebm25_indices/JuICe_train/code_topics",
                 ast_high_recall_size: int=1000, topics_high_recall_size: int=1000):
        self.ast_searcher = LuceneSearcher(ast_index_path)
        self.topics_searcher = LuceneSearcher(topics_index_path)
        self.ast_high_recall_size = ast_high_recall_size
        self.topics_high_recall_size = topics_high_recall_size

    def search(self, query: str, k: int=10):
        topics_query, ast_query = joint_get_code_topics_and_ast_seq(query)
        ast_hits = {hit.docid: hit.score for hit in self.ast_searcher.search(ast_query, k=self.ast_high_recall_size)}
        topics_hits = {hit.docid: hit.score for hit in self.topics_searcher.search(topics_query, k=self.topics_high_recall_size)}
        # print(list(ast_hits.items())[:10])
        # print(list(topics_hits.items())[:10])
        results = defaultdict(lambda:0)
        all_docids = set(ast_hits.keys()).union(topics_hits.keys())
        for docid in all_docids:
            results[docid] += ast_hits.get(docid, 0)
            results[docid] += topics_hits.get(docid, 0)
        results = {
            k:v for k,v in sorted(
                results.items(), 
                key=lambda x: x[1],
                reverse=True, 
            )
        }
        
        return {
            "combined": list(results.items())[:k],
            "ast": list(ast_hits.items())[:k],
            "topics": list(topics_hits.items())[:k],
        }

# BM25 sparse retriever implemented in sklearn (copied from some random GitHub gist). Not efficient.
class BM25SparseRetriever(object):
    def __init__(self, b: float=0.75, k1: float=1.6, 
                code2words_path: str="./JuICe_train_code2words.json"):
        self.vectorizer = TfidfVectorizer(norm=None, smooth_idf=False)
        self.b = b
        self.k1 = k1
        self.code2words_path = code2words_path
        if os.path.exists(code2words_path):
            self.code2words = {k: v if v is not None else "" for k,v in json.load(open(code2words_path)).items()}

    def save_code2word(self, all_codes: List[str]):
        code2words = {}
        for code in tqdm(all_codes):
            text = self.transform_code_to_text(code)
            code2words[code] = text
        self.code2words = code2words
        with open(self.code2words_path, "w") as f:
            json.dump(self.code2words, f, indent=4)

    def transform_code_to_text(self, code: str):
        """convert a piece of code to a stream of variable names and API calls."""
        code = process_nb_cell(code)
        try:
            all_terms = []
            for node in ast.walk(ast.parse(code)):
                if isinstance(node, ast.Name):
                    name = ast.unparse(node)
                elif isinstance(node, ast.Call):
                    name = ast.unparse(node.func)
                elif isinstance(node, ast.alias):
                    if node.asname is not None:
                        name = node.name + " " + node.asname
                    else: name = node.name
                else: continue
                for dot_split_term in name.split("."): # split by dots first.        
                    for underscore_split_term in dot_split_term.split("_"): # split by underscore second.
                        for term in camel_case_split(underscore_split_term): # split by camel case finally.
                            all_terms.append(term.lower())

            return " ".join(all_terms)
        except SyntaxError as e: return ""
            # print(e, code)
    def fit(self, X: List[str], verbose: bool=False):
        """ Fit IDF to documents X 
        X: is a list of codes."""
        X = [self.code2words[x] for x in tqdm(X)]
        self.vectorizer.fit(X)
        y = super(TfidfVectorizer, self.vectorizer).transform(X)
        self.avdl = y.sum(1).mean()

    def transform(self, q, X):
        """ Calculate BM25 between query q and documents X """
        b, k1, avdl = self.b, self.k1, self.avdl
        try: q = self.code2words[q]
        except KeyError:
            new_q = self.transform_code_to_text(q)
            if new_q == "": new_q = q
            else: q = new_q
        X = [self.code2words[x] for x in tqdm(X)]
        # apply CountVectorizer
        X = super(TfidfVectorizer, self.vectorizer).transform(X)
        len_X = X.sum(1).A1
        q, = super(TfidfVectorizer, self.vectorizer).transform([q])
        assert sparse.isspmatrix_csr(q)

        # convert to csc for better column slicing
        X = X.tocsc()[:, q.indices]
        denom = X + (k1 * (1 - b + b * len_X / avdl))[:, None]
        # idf(t) = log [ n / df(t) ] + 1 in sklearn, so it need to be coneverted
        # to idf(t) = log [ n / df(t) ] with minus 1
        idf = self.vectorizer._tfidf.idf_[None, q.indices] - 1.
        numer = X.multiply(np.broadcast_to(idf, X.shape)) * (k1 + 1)                                                          
        scores = (numer / denom).sum(1).A1
        results = []
        for i in scores.argsort()[::-1]: results.append(i)

        return results

# main
if __name__ == "__main__":
    # find 10 NNs for each code in the code KB.
    K = 10
    code_KB = json.load(open("./JuICe_train_code_KB.json"))
    codes = list(code_KB.keys())
    searcher = EnsembleCodeBM25Searcher()
    
    save_path = "./analysis/codebm25_topic_and_struct_10nn.jsonl"
    assert not os.path.exists(save_path)
    open(save_path, "w")
    for code in tqdm(codes):
        results = searcher.search(code, k=K)
        with open(save_path, "a") as f:
            f.write(json.dumps(results)+"\n")

