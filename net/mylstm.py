from torch import tanh, sigmoid, cat, mm
from torch.nn import LSTM
from .mymodel import MyModel
from .gate_extractor import GateExtractor


class MyLSTM(LSTM, GateExtractor, MyModel):
    def my_forward_with_extraction(self, xt, hidden):
        hiddens, cells = hidden
        evolution_of_xt, evolution_of_yt, gate_list = [], [], []
        for layer in range(self.num_layers):
            weights_input, weights_hidden, _, _ = self.all_weights[layer]
            weight_ii, weight_if, weight_ic, weight_io = weights_input.view(4, self.hidden_size, -1)
            weight_hi, weight_hf, weight_hc, weight_ho = weights_hidden.view(4, self.hidden_size, -1)

            input_gate = sigmoid(xt @ weight_ii.T + hiddens[layer] @ weight_hi.T)
            forget_gate = sigmoid(xt @ weight_if.T + hiddens[layer] @ weight_hf.T)
            main_gate = tanh(xt @ weight_ic.T + hiddens[layer] @ weight_hc.T)
            output_gate = sigmoid(xt @ weight_io.T + hiddens[layer] @ weight_ho.T)
            next_cell_state = forget_gate * cells[layer] + input_gate * main_gate
            xt = output_gate * tanh(next_cell_state)

            evolution_of_yt.append(next_cell_state)
            evolution_of_xt.append(xt)
            gate_list.append([gate.squeeze(dim=0).tolist() for gate in
                              [input_gate, forget_gate, main_gate, output_gate, next_cell_state]])
        return (cat(evolution_of_xt), cat(evolution_of_yt)), gate_list

