# do next step prediction given the previous steps.
import os 
import json
import torch
import random
import numpy as np
from typing import *
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

random.seed(42)

def compute_lcs_length(X: List[str], Y: List[str]) -> int:
    """function to compute longest common sub sequence."""
    # Declaring the array for storing the dp values
    m = len(X)
    n = len(Y)
    L = [[None]*(n+1) for i in range(m+1)]
    # Following steps build L[m+1][n+1] in bottom up fashion
    # Note: L[i][j] contains length of LCS of X[0..i-1]
    # and Y[0..j-1]
    for i in range(m+1):
        for j in range(n+1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif X[i-1] == Y[j-1]:
                L[i][j] = L[i-1][j-1]+1
            else:
                L[i][j] = max(L[i-1][j], L[i][j-1])
    # L[m][n] contains the length of LCS of X[0..n-1] & Y[0..m-1]
    return L[m][n]

def load_and_split_next_step_dataset(path: str, test_size: float=0.2):
    data = json.load(open(path))
    train_data = defaultdict(lambda: [])
    test_data = defaultdict(lambda: [])
    for intent, subs in data.items():
        train_indices, test_indices = train_test_split(
            range(len(subs)), 
            random_state=42, 
            test_size=test_size,
        )
        train_data[intent] = [subs[i] for i in train_indices]
        test_data[intent] = [subs[i] for i in test_indices]

    return train_data, test_data

def squish_duplicates(seq):
    """Given a sequence `seq` like A A A B C C return A B C"""
    squished_seq = []
    prev = None
    for step in seq:
        if prev != step:
            squished_seq.append(step)
        prev = step

    return squished_seq

class NextStepFromStepAndCodeDataset(Dataset):
    """Predict the next step when the input is a
    sequence of previous steps and codes"""
    def __init__(self, data: Dict[str, List[dict]], filt_thresh: float=0.8):
        # data = json.load(path)
        self.data = []
        self.filt_thresh = filt_thresh
        for intent, submissions in data.items():
            for sub in submissions:
                self.data.append(sub)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i: int):
        return [(chunk["META_code"], chunk["META_plan_op"], chunk["META_plan_op_score"]) for chunk in self.data[i]["chunks"] if chunk["META_plan_op_score"] >= self.filt_thresh]

# next step retriever.
class NextStepRetriever:
    def __init__(self, data: Dict[str, List[dict]], 
                 thresh: float=0.8, do_squish_duplicates: bool=True):
        self.data = data
        self.flat_data = [] # flattened data (not categorized intent wise)
        self.step_seqs = [] # step sequences.
        self.intent_wise_step_seqs = defaultdict(lambda: [])
        self.thresh = thresh
        print("fitting NSR model ...")
        for intent, subs in self.data.items():
            for sub in tqdm(subs, desc=f"collecting paths for: {intent}"):
                step_seq = [chunk["META_plan_op"] for chunk in sub["chunks"] if chunk["META_plan_op_score"] > thresh]
                if do_squish_duplicates: step_seq = self.squish_duplicates(step_seq)
                self.step_seqs.append(step_seq)
                self.intent_wise_step_seqs[intent].append(step_seq)
                self.flat_data.append(sub)
        self.intent_wise_step_seqs = dict(self.intent_wise_step_seqs)

    def squish_duplicates(self, seq):
        """Given a sequence `seq` like A A A B C C return A B C"""
        squished_seq = []
        prev = None
        for step in seq:
            if prev != step:
                squished_seq.append(step)
            prev = step

        return squished_seq

    def next_step(self, prompt_seq: List[str], 
                  intent: Union[str, None]=None):
        num_prompt_steps = len(prompt_seq)
        if intent is not None:
            step_seqs = self.intent_wise_step_seqs[intent]
        else: step_seqs = self.step_seqs
        lcs_scores = []
        for step_seq in step_seqs:
            lcs_scores.append(
                compute_lcs_length(
                    step_seq[:num_prompt_steps],
                    prompt_seq,
                )
            )
        max_ind = np.argmax(lcs_scores)
        try:
            return step_seqs[max_ind][num_prompt_steps]
        except IndexError: return None

# next step generation model.
class NextChunkPredictor:
    def __init__(self, ):
        pass

def test_dataset_class():
    dataset = NextStepFromStepAndCodeDataset(
        "./data/FCDS/code_qa_submissions_and_chunks.json",
        filt_thresh=0.8,
    )
    print(len(dataset))
    print("average sequence length:", np.mean([len(item) for item in dataset]))
    dataset = NextStepFromStepAndCodeDataset(
        "./data/FCDS/code_qa_submissions_and_chunks.json",
        filt_thresh=0,
    )
    print(len(dataset))
    print("average sequence length:", np.mean([len(item) for item in dataset]))

def test_nsr():
    train_data, test_data = load_and_split_next_step_dataset("./data/FCDS/code_qa_submissions_and_chunks.json")
    print(f"length of train: {len(train_data)}")
    print(f"length of test: {len(test_data)}")
    thresh: float=0.8
    do_squish_duplicates: bool=True
    nsr = NextStepRetriever(train_data)
    acc = 0
    intent_wise_acc = defaultdict(lambda: 0)
    intent_wise_tot = defaultdict(lambda: 0)
    depth_wise_tot = defaultdict(lambda: 0)
    depth_wise_acc = defaultdict(lambda: 0)
    for intent, subs in test_data.items():
        for sub in tqdm(subs, desc=f"eval for {intent}"):
            step_seq = [chunk["META_plan_op"] for chunk in sub["chunks"] if chunk["META_plan_op_score"] > thresh]
            if do_squish_duplicates: step_seq = squish_duplicates(step_seq)
            for i in range(1, len(step_seq)-1):
                step_sub_seq = step_seq[:i]
                true_next_step = step_seq[i]
                pred_next_step = nsr.next_step(step_sub_seq, intent)
                acc += int(true_next_step == pred_next_step)
                depth_wise_acc[i] += int(true_next_step == pred_next_step)
                depth_wise_tot[i] += 1
                intent_wise_acc[intent] += int(true_next_step == pred_next_step)
                intent_wise_tot[intent] += 1
    depth_wise_tot = dict(depth_wise_tot)
    depth_wise_acc = dict(depth_wise_acc)
    intent_wise_tot = dict(intent_wise_tot)
    intent_wise_acc = dict(intent_wise_acc)
    acc = acc / sum(list(depth_wise_tot.values()))
    for depth, tot in depth_wise_tot.items():
        depth_wise_acc[depth] = depth_wise_acc[depth] / tot
    for intent, tot in intent_wise_tot.items():
        intent_wise_acc[intent] = intent_wise_acc[intent] / tot

    def print_metric_dict(metric_dict: dict):
        for key, value in metric_dict.items():
            print(f"\x1b[34;1m{key}:\x1b[0m", round(100*value, 2))

    print(round(100*acc, 2))
    print_metric_dict(depth_wise_tot)
    print_metric_dict(depth_wise_acc)
    print_metric_dict(intent_wise_tot)
    print_metric_dict(intent_wise_acc)

# main
if __name__ == "__main__":
    # test_dataset_class()
    test_nsr()