#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# sparse retrievers.

import os
import ast
import json
import torch
import numpy as np
from typing import *
from torch.utils.data import Dataset, DataLoader
# code-to-code similarity computation.

class CodeDataset(Dataset):
    def __init__(self, data_dict: Dict[str, torch.Tensor]):
        super().__init__()
        self.data = data_dict
        self.keys = list(data_dict.keys())

    def __len__(self):
        return len(self.data[self.keys[0]])

    def __getitem__(self, i):
        out = {}
        for key in self.data:
            out[key] = self.data[key][i]
        
        return out