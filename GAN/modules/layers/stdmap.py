import torch
from torch import nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

class StdMap1d(nn.Module):
    """
    Calculates full standard deviation of filters and appends std as new filter

    Parameters
    ----------
    group_size : int, optional
        How many inputs are grouped together
        More inputs result in better estimation, less in more variance
        Setting it to -1 takes all inputs (default: -1)

    References
    ----------
    Karras, T., Aila, T., Laine, S., & Lehtinen, J. (2017).
    Progressive Growing of GANs for Improved Quality, Stability,
    and Variation. Retrieved from http://arxiv.org/abs/1710.10196
    """
    def __init__(self,group_size=-1):
        super(StdMap1d,self).__init__()
        self.group_size = group_size

    def forward(self,input):
        group_size = int(np.minimum(self.group_size,int(input.size(0))))
        if self.group_size<1:
            group_size = int(input.size(0))
        safe_idx = int(input.size(0)/group_size)*group_size
        input = input[:safe_idx]
        std = input.contiguous().view(group_size,-1,
                                        input.size(1),input.size(2))
        std = std-std.mean(dim=0,keepdim=True)
        std = torch.sqrt((std**2).mean(dim=0)+1e-8).mean(dim=2,keepdim=True).mean(dim=1,keepdim=True)
        std_map = std.repeat(group_size,1,input.size(2))
        input = torch.cat((input,std_map),dim=1)
        return input


class StdMap2d(nn.Module):
    """
    Calculates full standard deviation of filters and appends std as new filter

    Parameters
    ----------
    group_size : int, optional
        How many inputs are grouped together
        More inputs result in better estimation, less in more variance
        Setting it to -1 takes all inputs (default: -1)

    References
    ----------
    Karras, T., Aila, T., Laine, S., & Lehtinen, J. (2017).
    Progressive Growing of GANs for Improved Quality, Stability,
    and Variation. Retrieved from http://arxiv.org/abs/1710.10196
    """
    def __init__(self,group_size=-1):
        super(StdMap2d,self).__init__()
        self.group_size = group_size

    def forward(self,input):
        group_size = int(np.minimum(self.group_size,int(input.size(0))))
        if self.group_size<1:
            group_size = int(input.size(0))
        safe_idx = int(input.size(0)/group_size)*group_size
        input = input[:safe_idx]
        std = input.contiguous().view(group_size,-1,
                                        input.size(1),input.size(2),input.size(3))
        std = std-std.mean(dim=0,keepdim=True)
        std = torch.sqrt((std**2).mean(dim=0)+1e-8).mean(dim=3,keepdim=True).mean(dim=2,keepdim=True).mean(dim=1,keepdim=True)
        std_map = std.repeat(group_size,1,input.size(2),input.size(3))
        input = torch.cat((input,std_map),dim=1)
        return input
