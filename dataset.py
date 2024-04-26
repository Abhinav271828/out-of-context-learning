import torch
from torch.utils.data import Dataset
import random

class ShuffleDyck2(Dataset):
    def __init__(self, filename, percent=1.0):
        self.vocab = {"(": 0, ")": 1, "[": 2, "]": 3}
        data = []
        with open(filename) as f:
            data = [[self.vocab[c] for c in line[:-1]] for line in f]
        
        data = [d for d in data if random.random() <= percent]
        self.data = torch.tensor(data)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index]
