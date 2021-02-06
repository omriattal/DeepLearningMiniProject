from torch import tanh, cat, mm
from torch.nn import RNN
from .mymodel import MyModel
from .extractor import Extractor


class MyRNN(RNN, Extractor, MyModel):
    """
    shape of hidden: num_layers*batch*hidden_size
    """

    def recurrence(self, xt, hiddens):
        gate_list = []
        evolution_of_xt = []
        hiddens = [*hiddens]
        for layer in range(self.num_layers):
            weight_ih, weight_hh, _, _ = self.all_weights[layer]
            current_hidden = hiddens[layer]
            xt = tanh(
                xt @ weight_ih.T + current_hidden @ weight_hh.T)
            evolution_of_xt.append(xt)
            gate_list.append(xt.squeeze().tolist())
        return cat(evolution_of_xt), gate_list
