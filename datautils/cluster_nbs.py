import re
import pathlib
from typing import *
from datautils import read_jsonl
from collections import defaultdict

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

def camel_case_split(identifier, do_lower: bool=False):
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    if do_lower: return [m.group(0).lower() for m in matches]
    return [m.group(0) for m in matches]

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
                              filt_non_alts: bool=False) -> Dict[str, List[dict]]:
    """group notebooks by the base file name as a proxy for the topic.
    Parameters:
    - `seralize_nbs`: convert JuICe instances to a sequence of cells in the correct chronological order.
    - `use_full_path`: use the full `path` field in the metadata in order to group notebooks.
    - `filt_non_alts`: fitler out notebook synsets of size 1 (no potentially alternative implements).
    """
    # gather notebooks
    data = read_jsonl(path)
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
        if nb1[i][0].strip() != nb2[i][0].strip(): return False
    return True

def name_to_paths_dict(data: List[dict], filt_out_lt_1: bool=True) -> Dict[str, List[str]]:
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

def nb_prefix_inter_path_filename_synset_analysis(path: str):
    """compare within notebooks having the same full path."""
    val_data = read_jsonl(path)
    nb_clusters = gather_notebooks_by_topic(path, use_full_path=True, filt_non_alts=False)
    name_to_paths = name_to_paths_dict(val_data)
    tot_pairs = 0
    prefix_pairs = 0
    non_prefix_pairs = []
    for path, nbs in nb_clusters.items():
        for i in range(len(nbs)-1):
            for j in range(i+1, len(nbs)):
                tot_pairs += 1
                if is_nb_prefix(nbs[i], nbs[j]):
                    prefix_pairs += 1
                else: non_prefix_pairs.append(nbs[i], nbs[j])
    
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