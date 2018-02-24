"""
Spectral Normalization from https://openreview.net/forum?id=B1QRgziT-
"""
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import torch
import numpy as np

class SpectralNorm(object):
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name)
        u = getattr(module, self.name + '_u')

        weight_size = list(weight.size())
        weight_tmp = weight.data.view(weight_size[0],-1)
        v = weight_tmp.t().matmul(u)
        v = v/v.norm()
        u = weight_tmp.matmul(v)
        u = u/u.norm()
        o = u.t().matmul(weight_tmp.matmul(v))
        weight_tmp = weight_tmp/o
        weight.data = weight_tmp.view(*weight_size)

        setattr(module, self.name + '_u', u)
        setattr(module, self.name, weight)

    @staticmethod
    def apply(module, name):
        fn = SpectralNorm(name)

        weight = getattr(module, name)
        u = torch.Tensor(weight.size(0),1)
        u.normal_()

        module.register_buffer(name + '_u', u)
        module.register_forward_pre_hook(fn)

        return fn

    def remove(self, module):
        del module._buffers[name + '_u']

    def __call__(self, module, input):
        self.compute_weight(module)


def spectral_norm(module, name='weight', dim=0):
    r"""Applies weight normalization to a parameter in the given module.

    .. math::
         \mathbf{w} = g \dfrac{\mathbf{v}}{\|\mathbf{v}\|}

    Weight normalization is a reparameterization that decouples the magnitude
    of a weight tensor from its direction. This replaces the parameter specified
    by `name` (e.g. "weight") with two parameters: one specifying the magnitude
    (e.g. "weight_g") and one specifying the direction (e.g. "weight_v").
    Weight normalization is implemented via a hook that recomputes the weight
    tensor from the magnitude and direction before every :meth:`~Module.forward`
    call.

    By default, with `dim=0`, the norm is computed independently per output
    channel/plane. To compute a norm over the entire weight tensor, use
    `dim=None`.

    See https://arxiv.org/abs/1602.07868

    Args:
        module (nn.Module): containing module
        name (str, optional): name of weight parameter
        dim (int, optional): dimension over which to compute the norm

    Returns:
        The original module with the weight norm hook

    Example::

        >>> m = weight_norm(nn.Linear(20, 40), name='weight')
        Linear (20 -> 40)
        >>> m.weight_g.size()
        torch.Size([40, 1])
        >>> m.weight_v.size()
        torch.Size([40, 20])

    """
    SpectralNorm.apply(module, name)
    return module



def remove_spectral_norm(module, name='weight'):
    r"""Removes the weight normalization reparameterization from a module.

    Args:
        module (nn.Module): containing module
        name (str, optional): name of weight parameter

    Example:
        >>> m = weight_norm(nn.Linear(20, 40))
        >>> remove_weight_norm(m)
    """
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, SpectralNorm) and hook.name == name:
            hook.remove(module)
            del module._forward_pre_hooks[k]
            return module

    raise ValueError("weight_norm of '{}' not found in {}"
                     .format(name, module))