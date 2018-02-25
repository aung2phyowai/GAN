import torch
from torch import nn
import numpy as np

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
    def forward(input,eps=1e-8):
        tmp = torch.sqrt(input.pow(2).mean(dim=1,keepdim=True)+eps)
        input /= tmp
        return input

class WeightScale(object):
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        w = getattr(module, self.name + '_unscaled')
        c = getattr(module, self.name + '_c')
        return c*w

    @staticmethod
    def apply(module, name):
        fn = WeightScale(name)
        weight = getattr(module, name)
        # remove w from parameter list
        del module._parameters[name]

        #Constant from He et al. 2015
        c = np.sqrt(2./np.prod(list(weight.size())[1:]))
        setattr(module, name + '_c', torch.from_numpy(c))
        module.register_parameter(name + '_unscaled', Parameter(weight.data))
        setattr(module, name, fn.compute_weight(module))
        # recompute weight before every forward()
        module.register_forward_pre_hook(fn)
        return fn

    def remove(self, module):
        weight = self.compute_weight(module)
        delattr(module, self.name)
        del module._parameters[self.name + '_unscaled']
        del module._parameters[self.name + '_c']
        module.register_parameter(self.name, Parameter(weight.data))

    def __call__(self, module, inputs):
        setattr(module, self.name, self.compute_weight(module))

def weight_scale(module, name='weight'):
    WeightScale.apply(module, name)

    return module

def remove_weight_norm(module, name='weight'):
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, WeightScale) and hook.name == name:
            hook.remove(module)
            del module._forward_pre_hooks[k]
            return module

    raise ValueError("weight_scale of '{}' not found in {}"
                     .format(name, module))
