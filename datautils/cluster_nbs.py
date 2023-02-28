import torch
import pathlib
import numpy as np
from typing import *
from tqdm import tqdm
from sklearn.cluster import KMeans
from collections import defaultdict
# from sklearn.cluster import k_means_
from sklearn.preprocessing import normalize
from torch.utils.data import Dataset, DataLoader
from datautils import read_jsonl, camel_case_split
from datautils.keyphrase_extraction import load_keyphrases
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from datautils.code_cell_analysis import split_func_name, extract_api_full_name_dict
from datautils.markdown_cell_analysis import process_markdown
from CodeBERT.GraphCodeBERT.codesearch.parser.utils import remove_comments_and_docstrings

def filter_out_short_prefixes(data: List[dict]) -> List[dict]:
    path_wise_nbs = defaultdict(lambda:[])
    for inst in data:
        path = inst["metadata"]["path"]
        path_wise_nbs[path].append(inst)
    longest_nbs = []
    for path, nbs in path_wise_nbs.items():
        i = np.argmax([len(nb["context"]) for nb in nbs])
        longest_nbs.append(nbs[i])

    return longest_nbs 

def extract_notebook(inst: dict) -> List[Tuple[str, str]]:
    nb = []
    context = inst["context"][::-1]
    for cell in context:
        cell_type = cell["cell_type"]
        if cell_type == "markdown": 
            content = cell["nl_original"]
        else: content = cell["code"]
        nb.append((content, cell_type))
    nb.append((inst['code'], "code"))

    return nb

def filter_data_as_mapping(data: List[dict], use_filename: bool=True, 
                           use_path_as_key: bool=False) -> List[dict]:
    path_wise_nbs = defaultdict(lambda:[])
    for id, inst in enumerate(data):
        path = inst["metadata"]["path"]
        if use_filename: 
            path = pathlib.Path(path).stem
        path_wise_nbs[path].append((inst, id))
    longest_nbs = {}
    for path, nbs_and_ids in path_wise_nbs.items():
        i = np.argmax([len(nb["context"]) for nb, id in nbs_and_ids])
        nb, id = nbs_and_ids[i]
        if use_path_as_key:
            if use_filename:
                id = pathlib.Path(nb["metadata"]["path"]).stem
            else: id = nb["metadata"]["path"]
        longest_nbs[id] = nb

    return longest_nbs 
# def create_cluster(sparse_data, nclust = 10):
#     # Manually override euclidean
#     def euc_dist(X, Y = None, Y_norm_squared = None, squared = False):
#         #return pairwise_distances(X, Y, metric = 'cosine', n_jobs = 10)
#         return cosine_similarity(X, Y)
#     k_means_.euclidean_distances = euc_dist
#     scaler = StandardScaler(with_mean=False)
#     sparse_data = scaler.fit_transform(sparse_data)
#     kmeans = k_means_.KMeans(n_clusters = nclust, n_jobs = 20, random_state = 3425)
#     _ = kmeans.fit(sparse_data)

#     return kmeans
def gather_nbs_by_metadata(path: str) -> Dict[str, Dict[int, dict]]:
    # gather notebook
    data = read_jsonl(path)
    nb_grader, in_class_nbs, topics = {}, {}, []
    for i, rec in enumerate(data):
        topic_path = pathlib.Path(rec["metadata"]["path"]).stem
        # .split("/")[-1].strip()
        topics.append(topic_path)
        is_nb_grader = ("nbgrader" in rec["metadata"].keys())
        if is_nb_grader: # NOTE: NB grader notebooks might be missing some metadata keys
            # in total 18 metadata keys exist and not all are present in each metadata.
            nb_grader[i] = rec
        else: in_class_nbs[i] = rec

    return {
        "nb_grader": nb_grader,
        "in_class_nbs": in_class_nbs, 
        "topics": topics
    }

def build_bop_notebook_repr(inst: Dict[str, Any], nb_keyphrases: Dict[int, dict],
                            vocab_mapping: Dict[str, int]={}) -> List[Dict[str, Union[str, List[str]]]]:
    """given a notebook instance build a BOP (Bag of Phrases) representation from it.
    Parameters:
    - `inst`: dictionary corresponding to the instance.
    - `nb_keyphrases`: list of keyphrases for this notebook."""
    phrase_counts = defaultdict(lambda:0)
    kp_dict = {}
    vocab_size = len(vocab_mapping)
    for rec in nb_keyphrases:
        kp_dict[rec["md_id"]] = rec["keyphrases"]
    for i, cell in enumerate(inst["context"][::-1]):
        cell_type = cell["cell_type"]
        if cell_type == "code":
            # full_api_names = extract_api_full_name_dict(inst["code"])
            if cell["api_sequence"] is None: continue
            for api in cell["api_sequence"]:
                if api == "NO_API_SEQUENCE": continue
                # full_api = full_api_names[api]
                phrase = " ".join(split_func_name(api))
                phrase_counts[phrase] += 1

        if cell_type == "markdown":
            for phrase in kp_dict[i]:
                phrase_counts[phrase] += 1
    # full_api_names = extract_api_full_name_dict(inst['code'])
    if inst["api_sequence"] is not None:
        for api in inst["api_sequence"]:
            # full_api = full_api_names[api]
            if api == "NO_API_SEQUENCE": continue
            # api_phrase = " ".join(split_func_name(full_api))
            # cell_repr["content"].append(api_phrase)
            phrase = " ".join(split_func_name(api))
            phrase_counts[phrase] += 1

    if vocab_size == 0: return phrase_counts
    else:
        vec = np.zeros(vocab_size)
        for phrase, count in phrase_counts.items():
            phrase_id = vocab_mapping[phrase]
            vec[phrase_id] = count

        return vec

def fit_bop_model(path: str, keyphrase_path: str):
    data = filter_data_as_mapping(read_jsonl(path))
    nb_kps = load_keyphrases(keyphrase_path)
    vocab_mapping = set()
    for id in tqdm(data):
        phrase_counts = build_bop_notebook_repr(data[id], nb_kps[id])
        for phrase in phrase_counts:
            vocab_mapping.add(phrase)
    vocab_mapping = sorted(list(vocab_mapping))
    vocab_mapping = {k:v for v,k in enumerate(vocab_mapping)}
    print(f"using vocab size: {len(vocab_mapping)}")
    phrase_vecs = []
    for id in tqdm(data):
        phrase_vec = build_bop_notebook_repr(data[id], nb_kps[id], vocab_mapping)
        phrase_vecs.append(phrase_vec)
    phrase_vecs = np.stack(phrase_vecs)

    return data, phrase_vecs

# bag of phrases model.
class BOPModel:
    def __init__(self, kp_extractor):
        self.kp_extractor = kp_extractor
        
class DenseModelDataset(Dataset):
    def __init__(self, cell_seq: List[Tuple[str, str]],
                 tokenizer, **tok_args):
        super(DenseModelDataset, self).__init__()
        self.tokenizer = tokenizer
        self.cell_seq = cell_seq
        self.tok_args = tok_args
        

    def __len__(self):
        return len(self.cell_seq)

    def __getitem__(self, i):
        content, cell_type = self.cell_seq[i]
        if cell_type == "markdown":
            content = process_markdown(content)
        else: 
            content = remove_comments_and_docstrings(content, lang="python")
        tokenized_input = self.tokenizer(content, return_tensors="pt", truncation=True,
                                         padding="max_length", max_length=200)
        
        return [tokenized_input["input_ids"][0], tokenized_input["attention_mask"][0]]
        
# way to model NB cells with dense representations.
class DenseNBModel:
    """class to model NBs using dense transformer representations."""
    def __init__(self, path: str="microsoft/codebert-base"):
        from transformers import RobertaModel, RobertaTokenizer
        self.transformer = RobertaModel.from_pretrained(path)
        self.tokenizer = RobertaTokenizer.from_pretrained(path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def to(self, device: str):
        self.device = device
        self.transformer.to(device)

    def encode(self, inst: dict):
        cell_seq = extract_notebook(inst)
        dataset = DenseModelDataset(cell_seq, tokenizer=self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=8)
        op = []
        for batch in tqdm(dataloader):
            for i in range(len(batch)):
                batch[i] = batch[i].to(self.device)
            mini_batch_op = self.transformer(batch[0], batch[1]).pooler_output
            for cell_rep in mini_batch_op:
                op.append(cell_rep)

        return torch.stack(op)

# compute pair-wise similarity between NB cell sequences
def compute_nb_cell_seq_sim(nb1: torch.Tensor, nb2: torch.Tensor):
    """compute the similarity between Jupyter NB (sequence of cell representations)"""
    nb1 /= ((nb1**2).sum(dim=-1)**0.5).unsqueeze(dim=-1)
    nb2 /= ((nb2**2).sum(dim=-1)**0.5).unsqueeze(dim=-1)
    s1 = (nb1 @ nb2.T).max(dim=0).values.mean()
    s2 = (nb1 @ nb2.T).max(dim=-1).values.mean()
    
    return s1+s2

def test_dense_nb_model():
    model = DenseNBModel()
    val_insts = read_jsonl("./data/juice-dataset/dev.jsonl")
    reps = model.encode(val_insts[0])
    print(reps.shape)

def get_nb_clusters(path: str, keyphrase_path: str, **kmeans_params):
    data, X_bop = fit_bop_model(path, keyphrase_path)
    kmeans = KMeans(**kmeans_params).fit(normalize(X_bop))
    nb_clusters = defaultdict(lambda:[])
    for nb, label in zip(data.values(), kmeans.labels_):
        nb_clusters[label].append(nb)
    # kmeans = create_cluster(X_bop, n_clust)
    return nb_clusters, kmeans

def get_interesting_topics(nb_clusters: Dict[str, dict]) -> Dict[str, dict]:
    filt_nbs = defaultdict(lambda:[])
    for topic, nbs in nb_clusters.items():
        skip_terms = ["lab", "labs", "checkpoint", "exercise", "exercises",
                      "solution", "solutions", "question", "questions", 
                      "assignment", "assignments", "week", "weeks", 'project',
                      "problem", "problems"]
        topic_words = []
        for w in topic.replace("_", " ").replace("-", " ").split():
            topic_words += camel_case_split(w, do_lower=True) # splitting came case using regex.
        topic = " ".join([w for w in topic_words if w not in skip_terms])
        filt_nbs[topic] += nbs

    return filt_nbs

def gather_notebooks_by_topic(path: str, serialize_nbs: bool=True, 
                              use_full_path: bool=False, 
                              filt_non_alts: bool=False,
                              consider_longest_ones: bool=False) -> Dict[str, List[dict]]:
    """group notebooks by the base file name as a proxy for the topic.
    
    Parameters:
    - `path`: path to the validation data JSONL.
    - `seralize_nbs`: convert JuICe instances to a sequence of cells in the correct chronological order.
    - `use_full_path`: use the full `path` field in the metadata in order to group notebooks.
    - `filt_non_alts`: fitler out notebook synsets of size 1 (no potentially alternative implements).
    """
    # gather notebooks
    data = read_jsonl(path)
    if consider_longest_ones:
        data = filter_out_short_prefixes(data)
    nb_groups = defaultdict(lambda:[])
    for i, nb in enumerate(data):
        if use_full_path: topic_path = nb["metadata"]["path"]
        else: topic_path = pathlib.Path(nb["metadata"]["path"]).stem
        if serialize_nbs: nb = extract_notebook(nb)
        nb_groups[topic_path].append(nb)
    filt_nbs = {}
    if filt_non_alts:
        for topic, nbs in nb_groups.items():
            if len(nbs) > 1: filt_nbs[topic] = nbs 
        nb_groups = filt_nbs

    return nb_groups

def is_nb_prefix(nb1: List[Tuple[str, str]], nb2: List[Tuple[str, str]]):
    min_len = min(len(nb1), len(nb2))
    for i in range(min_len):
        c1 = nb1[i][0]
        c2 = nb2[i][0]
        if c1.strip() != c2.strip(): 
            for l1, l2 in zip(c1.split(), c2.split()):
                l1 = l1.strip()
                l2 = l2.strip()
                if l1 != l2: pass# print(l1 ,"||", l2)
            return False
    return True

def name_to_paths_dict(data: List[dict], 
                       filt_out_lt_1: bool=True) -> Dict[str, List[str]]:
    name_to_paths = defaultdict(lambda: set())
    for i, nb in enumerate(data):
        path = nb["metadata"]["path"]
        fname = pathlib.Path(path).stem
        name_to_paths[fname].add(path)
    for topic, paths in name_to_paths.items():
        name_to_paths[topic] = list(paths)
    if filt_out_lt_1:
        filt_op = {}
        for k,v in name_to_paths.items():
            if len(v) > 1:
                filt_op[k] = v
        name_to_paths = filt_op

    return name_to_paths

def nb_prefix_inter_path_filename_synset_analysis(
        path: str, consider_longest_ones: bool=True,
    ):
    """compare within notebooks having the same full path."""
    val_data = read_jsonl(path)
    if consider_longest_ones:
        val_data = filter_out_short_prefixes(val_data)
        print("filtering out the shortest prefixes gives instances:", len(val_data))
    nb_clusters = gather_notebooks_by_topic(
        path, use_full_path=True, 
        filt_non_alts=False, serialize_nbs=False,
        consider_longest_ones=consider_longest_ones,
    )
    name_to_paths = name_to_paths_dict(val_data)
    tot_pairs = 0
    prefix_pairs = 0
    non_prefix_pairs = []
    for name, paths in name_to_paths.items():
        for i in range(len(paths)-1):
            for j in range(i+1, len(paths)):
                p1c = nb_clusters[paths[i]]
                p2c = nb_clusters[paths[j]]
                for ii in range(len(p1c)):
                    for jj in range(len(p2c)):
                        nb1 = p1c[ii]
                        nb2 = p2c[jj]
                        tot_pairs += 1
                        if is_nb_prefix(extract_notebook(nb1), extract_notebook(nb2)): 
                            prefix_pairs += 1
                        else: non_prefix_pairs.append((nb1, nb2))
    print(f"percent of prefix-pairs: {(100*prefix_pairs/tot_pairs):.2f}% ({prefix_pairs}/{tot_pairs})")
    print(f"non-prefix pairs found: {len(non_prefix_pairs)}")
    
    return non_prefix_pairs

def nb_prefix_intra_path_synset_analysis(path: str):
    """compare within notebooks having the same full path."""
    nb_clusters = gather_notebooks_by_topic(path, use_full_path=True)
    tot_pairs = 0
    prefix_pairs = 0
    non_prefix_pairs = []
    for path, nbs in nb_clusters.items():
        for i in range(len(nbs)-1):
            for j in range(i+1, len(nbs)):
                tot_pairs += 1
                if is_nb_prefix(nbs[i], nbs[j]):
                    prefix_pairs += 1
                else: non_prefix_pairs.append((nbs[i], nbs[j]))
    print(f"percent of prefix-pairs: {(100*prefix_pairs/tot_pairs):.2f}% ({prefix_pairs}/{tot_pairs})")
    print(f"non-prefix pairs found: {len(non_prefix_pairs)}")
    
    return non_prefix_pairs

def compare_alternate_soln(path: str) -> Tuple[int]:
    nbs_clustered_by_topic = gather_notebooks_by_topic(path)
    count_with_alternatives = 0
    topic_pair_wise_overlaps = defaultdict(lambda: defaultdict(lambda: {}))
    for topic, nbs in nbs_clustered_by_topic.items():
        if len(nbs) > 1: count_with_alternatives += 1
        for i in range(len(nbs)-1):
            for j in range(i+1, len(nbs)):
                nb1 = nbs[i]
                nb2 = nbs[j]
                max_len = max(len(nb1), len(nb2))
                min_len = min(len(nb1), len(nb2))
                common_len = 0 # till what point are the two solutions aligned with each other
                # print(len(nb1), len(nb2), min_len)
                for k in range(min_len):
                    nb1_cell, nb1_ct = nb1[k]
                    nb2_cell, nb2_ct = nb2[k]
                    if nb1_ct != nb2_ct: break
                for k in range(min_len):
                    nb1_cell, nb1_ct = nb1[k]
                    nb2_cell, nb2_ct = nb2[k]
                    if nb1_cell != nb2_cell: break
                        # break
                        # print("\x1b[34;1mNB1:\x1b[0m", nb1[k])
                        # print("\x1b[34;1mNB2:\x1b[0m", nb2[k])
                    common_len += 1
                topic_pair_wise_overlaps[topic][str((i,j))] = {
                    "min_len": min_len,
                    "max_len": max_len,
                    "common_len": 0,
                }
    for topic, topic_dict in topic_pair_wise_overlaps.items():
        for k,v in topic_dict.items():
            topic_dict[k] = dict(v)
        topic_pair_wise_overlaps[topic] = dict(topic_dict)
    # print(topic, len(nbs))
    return count_with_alternatives, topic_pair_wise_overlaps