class MappingNetwork(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality, 0 = no latent.w_dim,                      # Intermediate latent (W) dimensionality.
	    w_dim,                      # Intermediate latent (W) dimensionality.
        num_layers      = 8,        # Number of mapping layers.
        intermediete_layer_features = None, # List of intermediete layers dimensionality, if = None equal to w_dim
        activation      = 'lrelu',  # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 0.01,     # Learning rate multiplier for the mapping layers.
        w_avg_beta      = 0.998,    # Decay for tracking the moving average of W during training, None = do not track.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.num_layers = num_layers
        self.w_avg_beta = w_avg_beta

        if intermediete_layer_features  == None:
					intermediete_layer_features = [w_dim] * (num_layers - 1)

				assert(len(intermediete_layer_features) == num_layers - 1)
        features_list = [z_dim] +  + [w_dim]

        for idx in range(num_layers):
            in_features = features_list[idx]
            out_features = features_list[idx + 1]
            layer = FullyConnectedLayer(in_features, out_features, activation=activation, lr_multiplier=lr_multiplier)
            setattr(self, f'fc{idx}', layer)


    def forward(self, z, truncation_psi=1, truncation_cutoff=None, update_wavg=False):
        # Embed, normalize
        z = z.to(torch.float32)
        z = z * (z.square().mean(dim=dim, keepdim=True) + eps).rsqrt()

        # Main layers.
        for idx in range(self.num_layers):
            layer = getattr(self, f'fc{idx}')
            z = layer(z)

        # Update moving average of W.
        if update_wavg and self.w_avg_beta is not None:
            self.w_avg.copy_(z.detach().mean(dim=0).lerp(self.w_avg, self.w_avg_beta))

        # Apply truncation.
        if truncation_psi != 1:
            assert self.w_avg_beta is not None
            if truncation_cutoff is None:
                z = self.w_avg.lerp(z, truncation_psi)
            else:
                z[:, :truncation_cutoff] = self.w_avg.lerp(z[:, :truncation_cutoff], truncation_psi)
        return z

    def extra_repr(self):
        return f'z_dim={self.z_dim:d}, w_dim={self.w_dim:d}'