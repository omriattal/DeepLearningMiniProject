from net.network import Network
from data.dataloader import DataLoader
from data import dataloader
from torch import argmax
from parameters import *
import numpy as np
import torch
import matplotlib.pyplot as plt
from IPython.display import display, IFrame
from matplotlib_venn import venn3
from collections import Counter
import json
import os.path as path
from net.mylstm import MyLSTM

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


def _plot_gate(gates):
    num_gates, num_layers, _, _ = gates.shape
    color_base = ["red", "green", "blue", "yellow"]
    gate_names = {2: ["Update", "Reset"], 3: ["Input", "Forget", "Output"]}
    gate_names = gate_names[num_gates]
    fig = plt.figure()
    fig.suptitle('Saturation Plot', fontsize=16)
    for gate_id, gate in enumerate(gates):
        # plot forget gate -------------------------------------------
        ax = fig.add_subplot(1, num_gates, gate_id + 1, aspect='equal', title=f'{gate_names[gate_id]} Gate',
                             xlabel='fraction right saturated', ylabel='fraction left saturated')
        for layer, (right, left) in enumerate(gate):
            # scatterplot with bigger points for more common points
            clr = Counter(zip(left, right))
            l, r = np.array(list(clr.keys())).T
            rad = 100 * np.array(list(clr.values())) / max(clr.values())
            c = np.full(len(l), color_base[layer])
            plt.scatter(l, r, rad, c, alpha=1.0 / num_layers)
        ax.plot(np.linspace(0, 1), np.linspace(1, 0))  # diagonal
        plt.draw()  # update plot
    plt.show()


def _get_saturation(gate, left_thresh, right_thresh):
    # takes: [layer, textlength, hiddensize), returns [2, layer, textlength]
    return (gate < left_thresh).mean(axis=-1), (gate > right_thresh).mean(axis=-1)


def visualize_gate(*gates):
    gates_lr_layer_text = np.array([_get_saturation(gate, 0.1, 0.9) for gate in gates])
    # reshape to [gate, layer, leftright, data]
    gate_layer_lr_text = gates_lr_layer_text.transpose((0, 2, 1, 3))
    _plot_gate(gate_layer_lr_text)


def create_gate_plots(model_name, num_layers, hidden_size, test_loader):
    network = Network.load_network(model_name, num_layers, hidden_size)
    network.eval()
    if model_name == "lstm":
        input_gates, forget_gates, cell_gates, output_gates, cell_states = network.extract_from_loader(test_loader)
        visualize_gate(input_gates, forget_gates, output_gates)
    elif model_name == "gru":
        # TODO: complete GRU case.
        network.extract_from_loader(test_loader)


if __name__ == '__main__':
    file_path = "data/warandpeace.txt"
    (train, test, val), vocabulary = dataloader.load_data(file_path, SPLITS, BATCH_SIZE, SEQ_LEN, True, DEVICE)
    # create_model_performance_table(test)
    # create_venn(test)
    create_gate_plots("lstm", 1, 32, test)


