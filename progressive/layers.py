import torch
from torch import nn

class Reshape(nn.Module):
    def __init__(self,shape):
        super(Reshape,self).__init__()
        self.shape = shape

    def forward(self,input):
        shape = list(self.shape)
        for i in range(len(shape)):
            if type(shape[i]) is list or type(shape[i]) is tuple:
                assert len(shape[i])==1
                shape[i] = input.size(shape[i][0])
        return input.view(shape)

class PixelNorm(nn.Module):
    def forward(input,eps=0.0000001):
        tmp = torch.sqrt(input.pow(2).mean(dim=1,keepdim=True)+eps)
        input /= tmp
        return input
