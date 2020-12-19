import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from Assignment2 import lstm, rnn, mlp

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
		self.rnn = rnn.MyRNNLayer(m + 1, k)
		self.V = mlp.MyLinear(k, m)
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
		self.lstm = lstm.MyLSTMLayer(m + 1, k)
		self.V = mlp.MyLinear(k, m)
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
		self.hidden_layer = mlp.MyLinear(m, k)
		self.V = mlp.MyLinear(k, m)
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
	model = MLPModel(T+2*K, hidden_size)
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
	main_lstm()
