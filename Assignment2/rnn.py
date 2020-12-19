import numpy as np
import torch
import torch.nn as nn


class MyRNNLayer(nn.Module):
	def __init__(self, input_size, hidden_size, nonlinearity='tanh'):
		super().__init__()

		self.input_size = input_size
		self.hidden_size = hidden_size
		self.nonlinearity = nonlinearity
		self.register_parameter('bias_ih', None)
		self.register_parameter('bias_hh', None)
		self.weight_hh = torch.nn.Parameter(torch.randn(self.hidden_size, self.hidden_size, requires_grad=True))
		self.weight_ih = torch.nn.Parameter(torch.randn(self.input_size, self.hidden_size, requires_grad=True))


	def update(self, my_input, hidden, label, lr=0.01):
		output = self.forward(my_input, hidden)
		mse_loss = nn.MSELoss()
		loss_output = mse_loss(output, label)
		loss_output.backward()

	def forward(self, my_input, hidden=None):
		if hidden is None:
			hidden = torch.zeros(my_input.size(0), self.hidden_size)
		if self.nonlinearity == 'tanh':
			next_hidden = torch.tanh(torch.mm(my_input, self.weight_ih) + torch.mm(hidden, self.weight_hh))
			return next_hidden
		if self.nonlinearity == 'relu':
			next_hidden = torch.relu(torch.mm(my_input, self.weight_ih) + torch.mm(hidden, self.weight_hh))
			return next_hidden


def main():
	rnn = MyRNNLayer(256, 30)
	my_input = torch.randn(10, 256)
	hx = torch.randn(10, 30)
	rnn.update(my_input, hx, torch.zeros(10, 30))


if __name__ == "__main__":
	main()
