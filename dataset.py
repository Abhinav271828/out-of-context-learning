import torch
from torch.utils.data import Dataset


class ShuffleDyck2(Dataset):
    def __init__(self):
        self.vocab = {"(": 0, ")": 1, "[": 2, "]": 3}
        data = []
        with open("data/shuffle_dyck_2.txt") as f:
            data = [[self.vocab[c] for c in line[:-1]] for line in f]
        self.data = torch.tensor(data)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index]
