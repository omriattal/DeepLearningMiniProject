from torch import tanh, cat, mm, sigmoid
from torch.nn import GRU
from .mymodel import MyModel
from .extractor import Extractor


class MyGRU(GRU, Extractor, MyModel):

    def recurrence(self, xt, hiddens):
        gate_list = []
        evolution_of_xt = []
        hiddens = [*hiddens]
        for layer in range(self.num_layers):
            weights_input, weights_hidden, _, _ = self.all_weights[layer]
            weight_ir, weight_iz, weight_in = weights_input.view(3, self.hidden_size, -1)
            weight_hr, weight_hz, weight_hn = weights_hidden.view(3, self.hidden_size, -1)
            current_hidden = hiddens[layer]
            rt = sigmoid(xt @ weight_ir.T + current_hidden @ weight_hr.T)
            zt = sigmoid(xt @ weight_iz.T + current_hidden @ weight_hz.T)
            nt = tanh((xt @ weight_in.T) + rt * (current_hidden @ weight_hn))
            xt = (1 - zt) * nt + zt * current_hidden  # TODO: check dimensions on the left
            evolution_of_xt.append(xt)
            gate_list.append([rt.squeeze_(dim=0).tolist(), zt.squeeze(dim=0).tolist()])
        return cat(evolution_of_xt), gate_list
