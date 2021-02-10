from torch import tanh, cat, mm
from torch.nn import RNN
from .mymodel import MyModel
from .gate_extractor import GateExtractor


class MyRNN(RNN, GateExtractor, MyModel):
    def my_forward_with_extraction(self, xt, hiddens):
        gate_list = []
        evolution_of_xt = []
        hiddens = [*hiddens]
        for layer in range(self.num_layers):
            weight_ih, weight_hh, _, _ = self.all_weights[layer]
            current_hidden = hiddens[layer]
            xt = tanh(xt @ weight_ih.T + current_hidden @ weight_hh.T)
            evolution_of_xt.append(xt)
            gate_list.append(xt.squeeze().tolist())  # will not be used
        return cat(evolution_of_xt), gate_list
