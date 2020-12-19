import numpy as np
import torch
import torch.nn as nn
import math

class MyLinear(nn.Module):
	def __init__(self, in_features, out_features):
		super().__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.register_parameter('bias', None)
		self.weight = torch.nn.Parameter(torch.randn(self.out_features, self.in_features, requires_grad=True))

	def forward(self, my_input):
		x, y = my_input.shape
		if y != self.in_features:
			print(f'Wrong Input Features. Please use tensor with {self.in_features} Input Features')
			return 0
		output = torch.sigmoid(torch.mm(my_input.float(), self.weight.t()))
		return output

	def __str__(self):
		return 'in_features={}, out_features={}, bias={}'.format(
			self.in_features, self.out_features, self.bias is not None)

def main():
	my_linear = MyLinear(256, 30)
	my_input = torch.randn(10, 256)
	output = my_linear.forward(my_input)
	my_linear.update(my_input, torch.zeros(10, 30))


if __name__ == "__main__":
	main()
