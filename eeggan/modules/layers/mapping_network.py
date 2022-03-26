#import things
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from eeggan.torch_utils.ops import bias_act

class FullyConnectedLayer(torch.nn.Module):
    def __init__(self,
        in_features,                # Number of input features.
        out_features,               # Number of output features.
        bias            = True,     # Apply additive bias before the activation function?
        activation      = 'linear', # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 1,        # Learning rate multiplier.
        bias_init       = 0,        # Initial value for the additive bias.
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.weight = torch.nn.Parameter(torch.randn([out_features, in_features]) / lr_multiplier)
        self.bias = torch.nn.Parameter(torch.full([out_features], np.float32(bias_init))) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x):
        w = self.weight.to(x.dtype) * self.weight_gain
        b = self.bias
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain

        if self.activation == 'linear' and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
            x = bias_act.bias_act(x, b, act=self.activation)
        return x

    def extra_repr(self):
        return f'in_features={self.in_features:d}, out_features={self.out_features:d}, activation={self.activation:s}'

class MappingNetwork(torch.nn.Module):
    def __init__(self,
        z_dim, # Input latent (Z) dimensionality, 0 = no latent.w_dim,                      # Intermediate latent (W) dimensionality.
	    w_dim,  # Intermediate latent (W) dimensionality.
        num_layers = 8,   # Number of mapping layers.
        intermediete_layer_features = None, # List of intermediete layers dimensionality, if = None equal to w_dim
        activation = 'lrelu', # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier = 0.01,  # Learning rate multiplier for the mapping layers.
        w_avg_beta = 0.998,  # Decay for tracking the moving average of W during training, None = do not track.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.num_layers = num_layers
        self.w_avg_beta = w_avg_beta

        if intermediete_layer_features == None: # If not specified, use w_dim
            intermediete_layer_features = [w_dim] * (num_layers - 1) # List of intermediete layers dimensionality, if = None equal to w_dim
        assert(len(intermediete_layer_features) == num_layers - 1) # Check that the number of layers is correct

        features_list = [z_dim] + intermediete_layer_features + [w_dim] # List of features for each layer

        for idx in range(num_layers):
            in_features = features_list[idx]
            out_features = features_list[idx + 1]
            layer = FullyConnectedLayer(in_features, out_features, activation=activation, lr_multiplier=lr_multiplier)
            setattr(self, f'fc{idx}', layer)


    def forward(self, z, truncation_psi=1, truncation_cutoff=None, update_wavg=False):
        # Embed, normalize
        z = z.to(torch.float32)
        dim = 1
        eps = 1e-8
        z = z * (z.square().mean(dim=dim, keepdim=True) + eps).rsqrt() # Normalize the latent vector.

        # Main layers.
        for idx in range(self.num_layers): # For each layer
            layer = getattr(self, f'fc{idx}') # Get the layer
            z = layer(z) # Apply the layer

        # Update moving average of W.
        if update_wavg and self.w_avg_beta is not None: # Update moving average of W.
            self.w_avg.copy_(z.detach().mean(dim=0).lerp(self.w_avg, self.w_avg_beta)) # Update moving average of W.

        # Apply truncation.
        if truncation_psi != 1: # Truncation
            assert self.w_avg_beta is not None # We need to track the moving average of W.
            if truncation_cutoff is None: # Use the average of the current and previous z vectors.
                z = self.w_avg.lerp(z, truncation_psi)
            else:
                z[:, :truncation_cutoff] = self.w_avg.lerp(z[:, :truncation_cutoff], truncation_psi)
        return z

    def extra_repr(self): # For printing the model.
        return f'z_dim={self.z_dim:d}, w_dim={self.w_dim:d}' # For printing the model.

a = torch.randn([100, 100])
map = MappingNetwork(100, 100, num_layers=3)
z = map(a)
