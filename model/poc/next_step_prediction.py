# do next step prediction given the previous steps.
import os 
import json
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

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

# next chunk predictor model.
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