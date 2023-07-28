# do next step prediction given the previous steps.
import os 
import json
import torch
import numpy as np
from typing import *
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import Dataset


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

class NextStepFromStepAndCodeDataset(Dataset):
    """Predict the next step when the input is a
    sequence of previous steps and codes"""
    def __init__(self, path, filt_thresh: float=0.8):
        data = json.load(path)
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
    def __init__(self, path: str, thresh: float=0.8, 
                 squish_duplicates: bool=True):
        self.path = path 
        self.data = json.load(open(path)) # intent wise data.
        self.flat_data = [] # flattened data (not categorized intent wise)
        self.step_seqs = [] # step sequences.
        self.intent_wise_step_seqs = defaultdict(lambda: [])
        self.thresh = thresh
        for intent, subs in self.data.items():
            for sub in subs:
                step_seq = [chunk["META_plan_op"] for chunk in sub["chunks"] if chunk["META_plan_op_score"] > thresh]
                if squish_duplicates: step_seq = self.squish_duplicates(step_seq)
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

# main
if __name__ == "__main__":
    test_dataset_class()