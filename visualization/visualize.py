from net.network import Network
from data.dataloader import DataLoader
from data import dataloader
from torch import argmax
from parameters import *
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from IPython.display import display, IFrame
from matplotlib_venn import venn3


def create_venn(train_loader: DataLoader):
    sets = {}
    for net_params in [("rnn", 1, 256), ("gru", 1, 128), ("lstm", 2, 64)]:
        net = Network.load_network(*net_params)
        net.eval()
        val = set()
        offset = 0
        for x, y in train_loader:
            correct = net.forward(x).argmax(-1) == y.flatten()
            val |= {n + offset for n, b in enumerate(correct) if b}
            offset += len(y.flatten())
        sets[net.name()] = val
    venn3(sets.values(), sets.keys())
    plt.show()


def create_model_performance_table(test_loader: DataLoader):
    model_names = ["lstm", "rnn", "gru"]
    n_layers = [1, 2, 3]
    hidden_sizes = [32, 64, 128, 256]
    corrects, losses = [], []
    for MODEL_NAME in model_names:
        for N_LAYERS in n_layers:
            for HIDDEN_SIZE in hidden_sizes:
                net = Network.load_from_file(MODEL_NAME, N_LAYERS, HIDDEN_SIZE).eval()
                correct, loss = [], []
                for x, y in test_loader:
                    out = net.forward(x)
                    correct += (out.argmax(-1) == y.flatten()).tolist()
                    loss += [(torch.nn.functional.cross_entropy(out, y.flatten())).tolist()]
                corrects.append(np.mean(correct))
                losses.append(np.mean(loss))

    for title, arr in (("accuracy", corrects), ("loss", losses)):
        arr = np.array(arr).reshape((len(hidden_sizes), len(model_names) * len(n_layers)))
        columns = pd.MultiIndex.from_product([model_names, n_layers])
        df = pd.DataFrame(arr, columns=columns, index=hidden_sizes)
        df.columns.name = title
        display(df.round(3))


if __name__ == '__main__':
    file_path = "data/warandpeace.txt"
    (train, test, val), vocabulary = dataloader.load_data(file_path, SPLITS, BATCH_SIZE, SEQ_LEN, True, DEVICE)
    create_model_performance_table(test)
