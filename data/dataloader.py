import torch
from torch.utils.data import DataLoader, Dataset
import os

from typing import Tuple


class MyDataset(Dataset):
    def __init__(self, data: torch.tensor, begin: int, end: int, time_steps: int, disjoint: bool):
        super(MyDataset, self).__init__()
        self.all_data = data[begin:end]
        self.disjoint = disjoint
        self.time_steps = time_steps + 1

    """
    mandatory function to implement as part of dataset
    """

    def __len__(self) -> int:
        if self.disjoint:
            return len(self.all_data) // self.time_steps
        else:
            return len(self.all_data) - self.time_steps

    """
     mandatory function to implement as part of dataset
    """

    def __getitem__(self, index):
        begin = 0
        if self.disjoint:
            begin = index * self.time_steps
        item = self.all_data[begin:begin + self.time_steps]
        item_inputs = item[:-1]
        item_predictions = item[1:]
        return item_inputs, item_predictions


def create_vocabulary_dict(data: str) -> dict:
    vocab = {c: o for o, c in enumerate(sorted(set(data)))}
    return vocab


def data_as_long_tensor(data: str, vocabulary, device) -> torch.tensor:
    return torch.tensor([vocabulary[c] for c in data], dtype=torch.long, device=device)


# TODO: what about that function?
def long_tensor_as_data(tensor):
    pass


def load_data(file_name, splits_percents, batch_size, time_steps, disjoint, device) -> Tuple[list, dict]:
    file = open(file_name, encoding="utf-8", mode="r")
    all_data = file.read()
    vocabulary = create_vocabulary_dict(all_data)
    splits = [len(all_data) * percent // 100 for percent in splits_percents]
    data_as_tensor = data_as_long_tensor(all_data, vocabulary, device)
    data_loaders = []
    for i in range(len(splits) - 1):
        current_dataset = MyDataset(data_as_tensor, splits[i], splits[i + 1], time_steps, disjoint)
        if i == 0:  # train
            data_loader = DataLoader(current_dataset, batch_size=batch_size, shuffle=True)  # TODO shuffle?
        else:  # validation and test
            data_loader = DataLoader(current_dataset, batch_size=batch_size, shuffle=False)  # TODO shuffle?
        data_loaders.append(data_loader)
    file.close()
    return data_loaders, vocabulary
