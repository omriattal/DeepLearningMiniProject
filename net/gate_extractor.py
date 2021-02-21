from torch import cat, zeros, stack
from torch.nn import LSTM


class GateExtractor:
    def forward_extract(self, x):
        self.eval()  # set the model for evaluation mode
        if self.batch_first:  # if the input and output dimensions are batch*seq_len*input_size
            x = x.transpose(0, 1)  # make it seq_len*batch*input_size

        is_lstm = isinstance(self, LSTM)
        if is_lstm:  # class inherits an LSTM
            hidden = [zeros(self.num_layers, 1, self.hidden_size, device=x.device) for _ in range(2)]  # we need two inputs for forward pass
        else:
            hidden = zeros(self.num_layers, 1, self.hidden_size, device=x.device)  # only one input
        outputs = []
        all_gates = []
        """
        move the input x forward into the network with my forward
        """
        for t in range(len(x)):  # seq_len
            xt = x[t]
            hidden, gates = self.my_forward_with_extraction(xt, hidden)
            if is_lstm:
                last_output = hidden[0][-1]  # take the last input only
            else:
                last_output = hidden[-1]  # take the last input only
            outputs.append(last_output)
            all_gates.append(gates)
        outputs = stack(outputs)
        if self.batch_first:
            outputs = outputs.transpose(0, 1)
        self.train()  # return to training mode
        return outputs, hidden, all_gates
