import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import math


# ______________________ LAYERS MODELS ______________________ #
class MyLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_parameter('bias', None)
        self.weight = torch.nn.Parameter(torch.zeros(self.out_features, self.in_features, requires_grad=True))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.weight, -math.sqrt(1 / self.in_features), math.sqrt(1 / self.in_features))

    def forward(self, my_input):
        x, y = my_input.shape
        if y != self.in_features:
            print(f'Wrong Input Features. Please use tensor with {self.in_features} Input Features')
            return 0
        output = torch.mm(my_input, self.weight.t())
        return output

    def __str__(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None)


class MyRNNLayer(nn.Module):
    def __init__(self, input_size, hidden_size, nonlinearity='tanh'):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nonlinearity = nonlinearity
        self.register_parameter('bias_ih', None)
        self.register_parameter('bias_hh', None)
        self.weight_ih = torch.nn.Parameter(torch.zeros(self.input_size, self.hidden_size, requires_grad=True))
        self.weight_hh = torch.nn.Parameter(torch.zeros(self.hidden_size, self.hidden_size, requires_grad=True))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.weight_ih, -math.sqrt(1 / self.hidden_size), math.sqrt(1 / self.hidden_size))
        nn.init.uniform_(self.weight_hh, -math.sqrt(1 / self.hidden_size), math.sqrt(1 / self.hidden_size))

    def forward(self, my_input, hidden=None):
        if hidden is None:
            hidden = torch.zeros(my_input.size(0), self.hidden_size)
        if self.nonlinearity == 'tanh':
            next_hidden = torch.tanh(torch.mm(my_input, self.weight_ih) + torch.mm(hidden, self.weight_hh))
            return next_hidden
        if self.nonlinearity == 'relu':
            next_hidden = torch.relu(torch.mm(my_input, self.weight_ih) + torch.mm(hidden, self.weight_hh))
            return next_hidden


class MyLSTMLayer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.register_parameter('bias_ih', None)
        self.register_parameter('bias_hh', None)
        self.weight_ii = torch.nn.Parameter(torch.randn(
            self.input_size, self.hidden_size, requires_grad=True))
        self.weight_if = torch.nn.Parameter(torch.randn(
            self.input_size, self.hidden_size, requires_grad=True))
        self.weight_ig = torch.nn.Parameter(torch.randn(
            self.input_size, self.hidden_size, requires_grad=True))
        self.weight_io = torch.nn.Parameter(torch.randn(
            self.input_size, self.hidden_size, requires_grad=True))
        self.weight_hi = torch.nn.Parameter(torch.randn(
            self.hidden_size, self.hidden_size, requires_grad=True))
        self.weight_hf = torch.nn.Parameter(torch.randn(
            self.hidden_size, self.hidden_size, requires_grad=True))
        self.weight_hg = torch.nn.Parameter(torch.randn(
            self.hidden_size, self.hidden_size, requires_grad=True))
        self.weight_ho = torch.nn.Parameter(torch.randn(
            self.hidden_size, self.hidden_size, requires_grad=True))

    def forward(self, my_input, h_0=None, c_0=None):
        if h_0 is None:
            h_0 = torch.zeros(my_input.size(0), self.hidden_size)
        if c_0 is None:
            c_0 = torch.zeros(my_input.size(0), self.hidden_size)
        i = torch.sigmoid(
            torch.mm(my_input, self.weight_ii) + torch.mm(h_0, self.weight_hi))
        f = torch.sigmoid(
            torch.mm(my_input, self.weight_if) + torch.mm(h_0, self.weight_hf))
        g = torch.tanh(torch.mm(my_input, self.weight_ig) + torch.mm(h_0, self.weight_hg))
        o = torch.sigmoid(
            torch.mm(my_input, self.weight_io) + torch.mm(h_0, self.weight_ho))
        c_1 = torch.mul(f, c_0) + torch.mul(i, g)
        h_1 = torch.mul(o, torch.tanh(c_1))
        return h_1, c_1


# ______________________ INITIALIZE PARAMETERS ______________________ #

seed = 1234
torch.manual_seed(seed)
np.random.seed(seed)

T = 20  # time_lag, how many blanks
K = 3  # how many digits in input vector
batch_size = 128
iter = 5000
n_train = iter * batch_size
n_classes = 9
hidden_size = 64
n_characters = n_classes + 1
lr = 1e-3
print_every = 20


def plot_graph(time_lag, X_LSTM, X_RNN, X_MLP, training_steps):
    plt.plot(training_steps, X_MLP, label="MLP")
    plt.plot(training_steps, X_RNN, label="RNN")
    plt.plot(training_steps, X_LSTM, label="LSTM")
    plt.plot([(10 * math.log(8, math.e)) / (time_lag + 20)] * training_steps[-1], label="BASELINE")

    plt.xlabel('Training Step')
    plt.ylabel('Loss Value')

    plt.legend()
    plt.show()


# ______________________ COPY DATA FUNCTIONS ______________________ #

# Copy data
def copy_data(T, K, batch_size):
    seq = np.random.randint(1, high=9, size=(batch_size, K))
    zeros1 = np.zeros((batch_size, T))
    zeros2 = np.zeros((batch_size, K - 1))
    zeros3 = np.zeros((batch_size, K + T))
    marker = 9 * np.ones((batch_size, 1))
    x = torch.LongTensor(np.concatenate((seq, zeros1, marker, zeros2), axis=1))
    y = torch.LongTensor(np.concatenate((zeros3, seq), axis=1))
    return x, y


# one hot encoding
def onehot(out, input):
    out.zero_()
    in_unsq = torch.unsqueeze(input, 2)
    out.scatter_(2, in_unsq, 1)


# ______________________ MODELS ______________________ #

class RNNModel(nn.Module):
    def __init__(self, m, k):
        super(RNNModel, self).__init__()
        self.m = m
        self.k = k
        self.rnn = MyRNNLayer(m + 1, k)
        self.V = MyLinear(k, m)
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, inputs):
        state = torch.zeros(inputs.size(0), self.k, requires_grad=False)
        outputs = []
        for input in torch.unbind(inputs, dim=1):
            state = self.rnn(input, state)
            outputs.append(self.V(state))
        return torch.stack(outputs, dim=1)

    def loss(self, logits, y):
        return self.loss_func(logits.view(-1, 9), y.view(-1))


class LSTMMODEL(nn.Module):
    def __init__(self, m, k):
        super(LSTMMODEL, self).__init__()
        self.m = m
        self.k = k
        self.lstm = MyLSTMLayer(m + 1, k)
        self.V = MyLinear(k, m)
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, inputs):
        hidden_state = torch.zeros(inputs.size(0), self.k, requires_grad=False)
        cell_state = torch.zeros(inputs.size(0), self.k, requires_grad=False)
        outputs = []
        for input in torch.unbind(inputs, dim=1):
            hidden_state, cell_state = self.lstm(input, hidden_state, cell_state)
            outputs.append(self.V(cell_state))
        return torch.stack(outputs, dim=1)

    def loss(self, logits, y):
        return self.loss_func(logits.view(-1, 9), y.view(-1))


class MLPModel(nn.Module):
    def __init__(self, m, k):
        super(MLPModel, self).__init__()
        self.m = m
        self.k = k
        self.hidden_layer = MyLinear(m + 1, k)
        self.V = MyLinear(k, m)
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, inputs):
        outputs = []
        for input in torch.unbind(inputs, dim=1):
            state = self.hidden_layer(input)
            outputs.append(self.V(state))
        return torch.stack(outputs, dim=1)

    def loss(self, logits, y):
        return self.loss_func(logits.view(-1, 9), y.view(-1))


def compare_output_with_label(logits, bY) -> int:
    bY_np = bY.detach().numpy()
    logits_np = logits.detach().numpy()

    for batch_index in range(logits_np.shape[0]):
        print(f"{logits_np[batch_index][-3::]} | {bY_np[batch_index][-3::]}")
        # for sequence_index in (range(logits_np.shape[1])):
        #     print(f"{logits_np[batch_index][sequence_index][-3::]} | {bY_np[batch_index][-3::]}")

    return 0


def main():
    graph_plotting_info = []
    steps_list = []
    for step in range(iter):
        if step % print_every == 0:
            steps_list.append(step)
    X, Y = copy_data(T, K, n_train)
    ohX = torch.FloatTensor(batch_size, T + 2 * K, n_characters)
    onehot(ohX, X[:batch_size])

    model_option = [MLPModel, RNNModel, LSTMMODEL]

    for class_model in model_option:
        current_model_steps_loss = []

        model = class_model(n_classes, hidden_size)
        model.train()
        opt = torch.optim.RMSprop(model.parameters(), lr=lr)
        for step in range(iter):
            bX = X[step * batch_size: (step + 1) * batch_size]
            bY = Y[step * batch_size: (step + 1) * batch_size]
            onehot(ohX, bX)
            opt.zero_grad()
            logits = model(ohX)
            loss = model.loss(logits, bY)
            loss.backward()
            opt.step()
            if step % print_every == 0:
                print('Step={}, Loss={:.4f}'.format(step, loss.item()))
                current_model_steps_loss.append(loss.item())

            if (step == 4980):
                compare_output_with_label(logits, bY)

        graph_plotting_info.append(current_model_steps_loss)
    plot_graph(T, graph_plotting_info[2], graph_plotting_info[1], graph_plotting_info[0], steps_list)


if __name__ == "__main__":
    main()
