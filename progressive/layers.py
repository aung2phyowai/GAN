import torch
from torch import nn

class Reshape(nn.Module):
    def __init__(self,shape):
        super(nn.Reshape,self)
        self.shape = shape

    def forward(self,input):
        shape = self.shape
        for i in range(len(shape)):
            if type(shape[i]) is list or type(shape[i]) is tuple:
                assert len(shape[i])==1
                shape[i] = input.size(shape[i][0])
        return torch.view(input,self.shape)
