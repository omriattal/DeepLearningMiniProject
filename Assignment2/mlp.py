import numpy as np
import torch
import torch.nn as nn

class MyLinear(nn.Module):
  def __init__(self, in_features, out_features):
    super().__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.register_parameter('bias', None)
    self.reset_parametrs()
  
  def forward(self, input):
    x,y = input.shape
    if y != self.in_features:
      print(f'Wrong Input Features. Please use tensor with {self.in_features} Input Features')
      return 0
    output = torch.sigmoid(input.matmul(self.weight.t()))
    return output
  
  def reset_parametrs(self):
    self.weight = torch.nn.Parameter(torch.randn(self.out_features, self.in_features, requires_grad=True))

  def update(self,input,label,lr = 0.01):
    output = self.forward(input)
    mse_loss = nn.MSELoss()
    loss_output=mse_loss(output,label)
    loss_output.backward()

  def __str__(self):
    return 'in_features={}, out_features={}, bias={}'.format(
        self.in_features, self.out_features, self.bias is not None )
  
if __name__ == "__main__":
    my_linear = MyLinear(256,30)
    input = torch.randn(10, 256)
    output = my_linear.forward(input)
    my_linear.update(input,torch.zeros(10,30))