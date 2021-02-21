from net.network import Network
from data.dataloader import DataLoader, MyDataset
from data import dataloader
from torch import argmax
import pandas as pd
from parameters import *
import numpy as np
import torch
import matplotlib.pyplot as plt
from IPython.display import display, IFrame
from matplotlib_venn import venn3
from collections import Counter
import json
import os.path as path
import os


def create_venn(train_loader: DataLoader):
    sets = {}
    data_size = len(train_loader.dataset.data)
    for net_params in [("rnn", 2, 256), ("gru", 2, 32), ("lstm", 1, 256)]:
        net = Network.load_network(*net_params)
        net.eval()
        val = set()
        offset = 0
        for x, y in train_loader:
            correct = net.forward(x).argmax(-1) == y.flatten()
            val |= {n + offset for n, b in enumerate(correct) if b}
            offset += len(y.flatten())
        sets[net.network_name()] = val
    venn3(sets.values(), sets.keys(), subset_label_formatter=lambda num: f"{(num / data_size):1.0%}")
    plt.show()


def create_model_performance_table(test_loader: DataLoader):
    model_names = ["lstm", "rnn", "gru"]
    num_layerss = [1, 2, 3]
    hidden_sizes = [32, 64, 128, 256]
    corrects, losses = [], []
    loss_func = torch.nn.functional.cross_entropy
    for model_name in model_names:
        for num_layer in num_layerss:
            for hidden_size in hidden_sizes:
                net = Network.load_network(model_name, num_layer, hidden_size)
                net.eval()  # set the network in evaluation mode.
                correct, loss = [], []
                for x, y in test_loader:
                    out = net.forward(x)
                    predictions = out.argmax(-1)
                    correct += (predictions == y.flatten()).tolist()
                    loss += [(loss_func(out, y.flatten())).tolist()]
                corrects.append(np.mean(correct))
                losses.append(np.mean(loss))

    for title, arr in (("accuracy", corrects), ("loss", losses)):
        arr = np.array(arr).reshape((len(hidden_sizes), len(model_names) * len(num_layerss)))
        columns = pd.MultiIndex.from_product([model_names, num_layerss])
        df = pd.DataFrame(arr, columns=columns, index=hidden_sizes)
        df.columns.name = title
        display(df.round(3))


def plot_gate(gates):
    num_gates, num_layers, _, _ = gates.shape
    color_base = ["red", "green", "blue"]
    gate_names = {2: ["Update", "Reset"], 3: ["Input", "Forget", "Output"]}
    gate_names = gate_names[num_gates]
    fig = plt.figure()
    fig.suptitle('Saturation Plot', fontsize=16)
    for gate_id, gate in enumerate(gates):
        ax = fig.add_subplot(1, num_gates, gate_id + 1, aspect='equal', title=f'{gate_names[gate_id]} Gate',
                             xlabel='fraction right saturated', ylabel='fraction left saturated')
        for layer, (right, left) in enumerate(gate):
            clr = Counter(zip(left, right))
            l, r = np.array(list(clr.keys())).T
            rad = 100 * np.array(list(clr.values())) / max(clr.values())
            c = np.full(len(l), color_base[layer])
            plt.scatter(l, r, rad, c, alpha=1.0 / num_layers)
        ax.plot(np.linspace(0, 1), np.linspace(1, 0))
        plt.draw()
    plt.show()


def get_saturation(gate, left_thresh, right_thresh):
    return (gate < left_thresh).mean(axis=-1), (gate > right_thresh).mean(axis=-1)


def visualize_gate(*gates):
    gates_lr_layer_text = np.array([get_saturation(gate, 0.1, 0.9) for gate in gates])
    gate_layer_lr_text = gates_lr_layer_text.transpose((0, 2, 1, 3))
    plot_gate(gate_layer_lr_text)


def create_gate_plots(model_name, num_layers, hidden_size, test_loader):
    network = Network.load_network(model_name, num_layers, hidden_size)
    network.eval()
    if model_name == "lstm":
        input_gates, forget_gates, cell_gates, output_gates, cell_states = network.extract_from_loader(test_loader)
        visualize_gate(input_gates, forget_gates, output_gates)
    elif model_name == "gru":
        reset_gate, update_gate = network.extract_from_loader(test_loader)
        visualize_gate(update_gate, reset_gate)


def visualize_cell(cell, x, hidden_size):
    char_cell = {'cell_size': hidden_size, 'seq': ''.join(x)}
    char_cell.update({f"cell_layer_{layer + 1}": cell[layer].tolist() for layer in range(len(cell))})
    with open(path.join("cell_visualization", 'char_cell.json'), 'w') as json_file:
        json.dump(char_cell, json_file)


def create_cell_visualization(model_name, num_layers, hidden_size, test_loader: DataLoader, vocabulary):
    network = Network.load_network(model_name, num_layers, hidden_size)
    network.eval()
    begin = 1
    last = 844
    if model_name == "lstm":
        input_gates, forget_gates, cell_gates, output_gates, cell_states = network.extract_from_loader(test_loader)
        visualize_cell(cell_states[:, begin:last],
                       dataloader.long_tensor_as_data(test_loader.dataset.data[begin:last], vocabulary), hidden_size)
    elif model_name == "gru":
        reset_gates, update_gates = network.extract_from_loader(test_loader)
        visualize_cell(reset_gates[:, begin:last],
                       dataloader.long_tensor_as_data(test_loader.dataset.data[begin:last], vocabulary), hidden_size)


if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # prevents errors
    file_path = "data/warandpeace.txt"
    (train, test, val), vocabulary = dataloader.load_data(file_path, SPLITS, BATCH_SIZE, SEQ_LEN, DEVICE)
    # Uncomment to reproduce one of the following.
    # create_model_performance_table(test)
    # create_venn(test)
    # create_gate_plots("gru", 3, 128, test)
    # create_cell_visualization("lstm", 1, 256, test, vocabulary)
