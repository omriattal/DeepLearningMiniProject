from torch import cat, zeros, stack
from torch.nn import LSTM
from mymodel import MyModel


class Extractor:
    def forward_with_extract(self: MyModel, x, verify=True):
        x_orig = x  # saved for a later forward pass
        self.eval() # set the model for evaluation mode
        if self.batch_first:  # if the input and output dimensions are batch*seq_len*feature_size
            x = x_orig.transpose(0, 1)  # make it seq_len*batch*input_size

        if isinstance(self, LSTM):  # class inherits an LSTM
            hidden = [zeros(self.num_layers, 1, self.hidden_size, device=x.device) for _ in range(2)]  # we need two inputs for forward pass
        else:
            hidden = zeros(self.num_layers, 1, self.hidden_size, device=x.device)  # only one input
        outputs = []
        all_gates = []
        for t in range(len(x)):  # seq_len
            xt = x[t]
            hidden, gates = self.recurrence(xt, hidden)
            if type(hidden) == tuple:
                last_output = hidden[0][-1]
            else:
                last_output = hidden[-1]
            outputs.append(last_output)
            all_gates.append(gates)
        outputs = stack(outputs)
        if self.batch_first:
            outputs = outputs.transpose(0, 1)
        self.train()  # return to training mode. TODO: remove,also next line.

        if verify:
            outputs2, hidden2 = self.forward(x_orig)
            assert (outputs - outputs2).mean() < 0.0001
            assert (cat(hidden) - cat(hidden2) if isinstance(self, LSTM) else hidden - hidden2).mean() < 0.0001

        return outputs, hidden, all_gates
