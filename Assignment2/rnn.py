import numpy as np
import torch
import torch.nn as nn

class MyRNNLayer (nn.Module):
  def __init__(self, input_size, hidden_size, nonlinearity = 'tanh'):
    super().__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.nonlinearity = nonlinearity
    self.register_parameter('bias_ih', None)
    self.register_parameter('bias_hh', None)
    self.reset_parameters()
  
  def reset_parameters(self):
    self.weight_ih = torch.nn.Parameter(torch.randn(self.input_size, self.hidden_size,requires_grad = True))
    self.weight_hh = torch.nn.Parameter(torch.randn(self.hidden_size, self.hidden_size,requires_grad = True))


  def update(self,input,hidden,label,lr = 0.01):
    output = self.forward(input,hidden)
    mse_loss = nn.MSELoss()
    loss_output=mse_loss(output,label)
    loss_output.backward()

  def forward(self, input , hidden = None):
    if hidden is None:
      hidden = torch.zeros(input.size(0),self.hidden_size)
    if self.nonlinearity == 'tanh':
      next_hidden = torch.tanh(torch.mm(input, self.weight_ih) + torch.mm(hidden, self.weight_hh))
      return next_hidden
    if self.nonlinearity == 'relu':  
      next_hidden = torch.relu(torch.mm(input, self.weight_ih) + torch.mm(hidden, self.weight_hh))
      return next_hidden
  
if __name__ == "__main__":    
    rnn = MyRNNLayer(256,30)
    input = torch.randn(10,256)
    hx = torch.randn(10,30)
    rnn.update(input,hx,torch.zeros(10,30))