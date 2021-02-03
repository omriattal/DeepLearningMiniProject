from torch import tanh, cat, mm, sigmoid
from torch.nn import GRU
from .mymodel import MyModel


class MyGRU(GRU, MyModel):

    def recurrence(self, xt, hiddens):
        gate_list = []
        evolution_of_xt = []
        hiddens = [*hiddens]
        for layer in range(self.num_layers):
            weights_input, weights_hidden, _, _ = self.all_weights[layer]
            weight_ir, weight_iz, weight_in = weights_input.view(3, self.hidden_size, self.input_size)  # TODO: or -1 at the end?
            weight_hr, weight_hz, weight_hn = weights_hidden.view(3, self.hidden_size, self.hidden_size)
            current_hidden = hiddens[layer]
            rt = sigmoid(mm(xt, weight_ir.T) + mm(current_hidden, weight_hr.T))
            zt = sigmoid(mm(xt, weight_iz.T) + mm(current_hidden, weight_hz.T))
            nt = tanh(mm(xt, weight_in.T) + rt * (mm(current_hidden, weight_hn)))
            xt = (1 - zt) * nt + zt * current_hidden
            evolution_of_xt.append(xt)
            gate_list.append([rt.squeeze_(dim=0).tolist(), zt.squeeze(dim=0).tolist()])
        return evolution_of_xt, gate_list
