import numpy as np
import torch
from torch.nn import Linear, Dropout, Module
from .mylstm import MyLSTM
from .mygru import MyGRU
from .myrnn import MyRNN


class Network(Module):

    def __init__(self, vocabulary_size, hidden_size, model_name, num_layers, device):
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.hidden_size = hidden_size
        self.model_name = model_name  # lstm, gru or rnn
        self.num_layers = num_layers
        self.device = device
        self.predictor = Linear(hidden_size, vocabulary_size)
        if model_name == "lstm":
            self.net = MyLSTM(vocabulary_size, hidden_size, num_layers=num_layers, batch_first=True)
        elif model_name == "gru":
            self.net = MyGRU(vocabulary_size, hidden_size, num_layers=num_layers, batch_first=True)
        else:
            self.net = MyRNN(vocabulary_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.to(self.device)

    def one_hot(self, x):
        return torch.nn.functional.one_hot(x, num_classes=self.vocabulary_size).float()

    # TODO: implement extract from loader

    def forward(self, x):
        seq_len, batch_size = x.shape
        x = self.one_hot(x)
        x, hidden = self.net(x)  # regular forward for the network
        label = self.predictor(x)
        return label.view(batch_size * seq_len, -1)

    def extract_gates(self, x):
        x = self.one_hot(x)
        return self.net.forward_extract(x)[2]

    def network_name(self):
        return f"{self.model_name}-{self.num_layers}-{self.hidden_size}"

    def save_network(self):
        self._forward_hooks.clear()  # must be empty
        with open(f"models/{self.network_name()}.pkl", "wb") as file:
            torch.save(self, file)

    @staticmethod
    def load_network(model_name, num_layers, hidden_size):
        with open(f"models/{model_name}-{num_layers}-{hidden_size}.pkl", "rb") as file:
            net_from_file = torch.load(file)
            net = Network(net_from_file.vocabulary_size, net_from_file.hidden_size, net_from_file.model_name, net_from_file.num_layers, net_from_file.device)
            net.load_state_dict(net_from_file.state_dict())
            return net
