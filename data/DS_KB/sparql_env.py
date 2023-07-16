# environment to simulate execution of SPARQL queries.
import torch
import gymnasium as gym
from torch.utils.data import Dataset

class DataScienceKB:
    def __init__(self):
        pass

class JupyterNBDataset(Dataset):
    def __init__(self, path: str):
        pass

class DataScienceKBSPARQLEnv(gym.Env):
    def __init__(self, kb: DataScienceKB, dataset):
        self.kb = kb

# main
if __name__ == "__main__":
    pass