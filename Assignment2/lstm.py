import numpy as np
import torch
import torch.nn as nn


class MyLSTMLayer (nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.register_parameter('bias_ih', None)
        self.register_parameter('bias_hh', None)
        self.reset_parameters()

    def reset_parameters(self):
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

    def forward(self, input, h_0, c_0):
        if h_0 is None:
            h_0 = torch.zeros(input.size(0), self.hidden_size)
        if c_0 is None:
            c_0 = torch.zeros(input.size(0), self.hidden_size)
        self.i = torch.sigmoid(
            torch.mm(input, self.weight_ii) + torch.mm(h_0, self.weight_hi))
        self.f = torch.sigmoid(
            torch.mm(input, self.weight_if) + torch.mm(h_0, self.weight_hf))
        self.g = torch.tanh(torch.mm(input, self.weight_ig) +
                            torch.mm(h_0, self.weight_hg))
        self.o = torch.sigmoid(
            torch.mm(input, self.weight_io) + torch.mm(h_0, self.weight_ho))
        c_1 = torch.mul(self.f, c_0) + torch.mul(self.i, self.g)
        h_1 = torch.mul(self.o, torch.tanh(c_1))
        return h_1, c_1

    def update(self, input, hidden, cell_state, label, lr=0.01):
        h_output, c_output = self.forward(input, hidden, cell_state)
        mse_loss = nn.MSELoss()
        loss_output_c = mse_loss(c_output, label)
        loss_output_c.backward()


if __name__ == "__main__":
    lstm = MyLSTMLayer(256, 30)
    input = torch.randn(10, 256)
    hx = torch.randn(10, 30)
    cx = torch.randn(10, 30)
    lstm.update(input, hx, cx, torch.zeros(10, 30))
