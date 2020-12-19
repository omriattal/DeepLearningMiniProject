import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import math

# Our LSTM Layer
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

    def forward(self, my_input, h_0, c_0):
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

    def update(self, my_input, hidden, cell_state, label, lr=0.01):
        h_output, c_output = self.forward(my_input, hidden, cell_state)
        mse_loss = nn.MSELoss()
        loss_output_c = mse_loss(c_output, label)
        loss_output_c.backward()


# Our RNN Layer
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
        nn.init.uniform_(self.weight_ih, -math.sqrt(1/self.hidden_size),math.sqrt(1/self.hidden_size))
        nn.init.uniform_(self.weight_hh, -math.sqrt(1/self.hidden_size),math.sqrt(1/self.hidden_size))


    def update(self, input, hidden, label, lr=0.01):
        output = self.forward(input, hidden)
        mse_loss = nn.MSELoss()
        loss_output = mse_loss(output, label)
        loss_output.backward()

    def forward(self, input, hidden=None):
        if hidden is None:
            hidden = torch.zeros(input.size(0), self.hidden_size)
        if self.nonlinearity == 'tanh':
            next_hidden = torch.tanh(torch.mm(input, self.weight_ih) + torch.mm(hidden, self.weight_hh))
            return next_hidden
        if self.nonlinearity == 'relu':
            next_hidden = torch.relu(torch.mm(input, self.weight_ih) + torch.mm(hidden, self.weight_hh))
            return next_hidden


# Our MLP Layer
class MyLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_parameter('bias', None)
        self.weight = torch.nn.Parameter(torch.zeros(self.out_features, self.in_features, requires_grad=True))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.weight, -math.sqrt(1/self.in_features),math.sqrt(1/self.in_features))

    def forward(self, my_input):
        x, y = my_input.shape
        if y != self.in_features:
            print(f'Wrong Input Features. Please use tensor with {self.in_features} Input Features')
            return 0
        output = torch.sigmoid(torch.mm(my_input.float(), self.weight.t()))
        return output

    def update(self, my_input, label, lr=0.01):
        output = self.forward(my_input)
        mse_loss = nn.MSELoss()
        loss_output = mse_loss(output, label)
        loss_output.backward()

    def __str__(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None)


# Copy task base code with our modifications
print(np.__version__)
print(torch.__version__)

# Set the seed of PRNG manually for reproducibility
seed = 1234
torch.manual_seed(seed)
np.random.seed(seed)


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
        self.hidden_layer = MyLinear(m, k)
        self.V = MyLinear(k, m)
        self.loss_func = nn.MSELoss()

    def forward(self, inputs):
        outputs = []
        for my_input in torch.unbind(inputs, dim=0):
            my_input_as_tensor = my_input.reshape(1, -1)
            hidden_layer_output = self.hidden_layer(my_input_as_tensor)
            output_layer_output = self.V(hidden_layer_output)
            outputs.append(output_layer_output.reshape(-1))
        return torch.stack(outputs, dim=0)

    def loss(self, logits, y):
        return self.loss_func(logits, y.float())


T = 20
K = 3
batch_size = 128
iter = 5000
n_train = iter * batch_size
n_classes = 9
hidden_size = 64
n_characters = n_classes + 1
lr = 1e-3
print_every = 20


def main_lstm():
    # create the training data
    X, Y = copy_data(T, K, n_train)
    print('{}, {}'.format(X.shape, Y.shape))
    ohX = torch.FloatTensor(batch_size, T + 2 * K, n_characters)
    onehot(ohX, X[:batch_size])
    print('{}, {}'.format(X[:batch_size].shape, ohX.shape))
    model = LSTMMODEL(n_classes, hidden_size)
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


def main_rnn():
    # create the training data
    X, Y = copy_data(T, K, n_train)
    print('{}, {}'.format(X.shape, Y.shape))
    ohX = torch.FloatTensor(batch_size, T + 2 * K, n_characters)
    onehot(ohX, X[:batch_size])
    print('{}, {}'.format(X[:batch_size].shape, ohX.shape))
    model = RNNModel(n_classes, hidden_size)
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


def main_mlp():
    # create the training data
    X, Y = copy_data(T, K, n_train)
    print('{}, {}'.format(X.shape, Y.shape))
    model = MLPModel(T + 2 * K, hidden_size)
    model.train()
    opt = torch.optim.RMSprop(model.parameters(), lr=lr)
    for step in range(iter):
        bX = X[step * batch_size: (step + 1) * batch_size]
        bY = Y[step * batch_size: (step + 1) * batch_size]
        opt.zero_grad()
        logits = model(bX)
        loss = model.loss(logits, bY)
        loss.backward()
        opt.step()
        if step % print_every == 0:
            print('Step={}, Loss={:.4f}'.format(step, loss.item()))


if __name__ == "__main__":
    main_mlp()
    # main_rnn()
