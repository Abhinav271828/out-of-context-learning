import torch
import os
import random
from tqdm import tqdm

VOCAB = ["a", "b", "c", "#"]

# char to index
char_to_index = {char: i for i, char in enumerate(VOCAB)}


def make_data_set_arrays_split(data_path, max_len: int = 20):
    # loop through each file in the data_path directory
    data = []
    for file in os.listdir(data_path):
        # read the file and append the data to the self.data list
        with open(os.path.join(data_path, file), "r") as f:
            data.append(f.readlines())

    # flatten the list of lists
    data_fr = []
    for i, arr in enumerate(data):
        for string in tqdm(arr):
            string = string.strip("\n")
            # if len()
            if len(string) <= max_len:
                string += "#" * (max_len - len(string))  # pad with #
            if len(string) > max_len:
                continue  # skip strings that are too long
            data_fr.append(
                (
                    string.strip("\n"),
                    i,
                )
            )

    # turn each string to a tensor of indices
    data = []
    for string, i in tqdm(data_fr):
        data.append(
            (
                torch.tensor(
                    [char_to_index[char] for char in string], dtype=torch.float32
                ),
                i,
            )
        )

    # shuffle
    random.shuffle(data)
    return data


class AutoEncoderDataset(torch.utils.data.Dataset):
    def __init__(self, data_arr, max_len: int = 20):

        self.data = data_arr
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "string": self.data[idx][0],
            "language": self.data[idx][1],
        }


def create_dataset_splits(
    data_path: str,
    max_len: int = 20,
    train_percent: float = 0.8,
    val_percent: float = 0.1,
    test_percent: float = 0.1,
):
    data = make_data_set_arrays_split(data_path, max_len)
    train_size = int(train_percent * len(data))
    val_size = int(val_percent * len(data))
    test_size = len(data) - train_size - val_size

    train_data = data[:train_size]
    val_data = data[train_size : train_size + val_size]
    test_data = data[train_size + val_size :]

    return (
        AutoEncoderDataset(train_data, max_len),
        AutoEncoderDataset(val_data, max_len),
        AutoEncoderDataset(test_data, max_len),
    )
