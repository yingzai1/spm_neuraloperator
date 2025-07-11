import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Callable, Sequence

from flax.linen.initializers import glorot_normal

activation = nn.relu

class BranchNet(nn.Module):
    """Same ResNet as in PINN for Branch"""
    layers: list
    activation: Callable = activation
    kernel_init: Callable = glorot_normal()
    bias_init: Callable = nn.initializers.zeros

    def setup(self):
        self.layer_1 = nn.Dense(self.layers[0])
        self.layer_cathode_1 = nn.Dense(self.layers[0])
        
        self.network = [ResBlock(features=width) for width in self.layers[1:-1]]

        self.output_layer = nn.Dense(self.layers[-1])

    def __call__(self, I, c0):

        a = jnp.concatenate((I, c0), axis=-1)
        # jax.debug.print("anode after stacking shape: {}",anode.shape)
        a = self.layer_1(a)
        # jax.debug.print("anode shape after first layer: {}",anode.shape)
        for layer in self.network:
            a = layer(a)
            a = activation(a)

        beta = self.output_layer(a)
        return beta

class ResBlock(nn.Module):
    features: int
    kernel_init: Callable = glorot_normal()

    @nn.compact
    def __call__(self, x):
        residual = x  # Save input for the residual connection
        y = nn.Dense(self.features, kernel_init=self.kernel_init)(x)
        y = activation(y)
        y = nn.Dense(self.features, kernel_init=self.kernel_init)(y)
        y = y + residual  # Add the residual connection
        y = activation(y)
        return y

class TrunkNet(nn.Module):
    """Trunk network for the DeepONet."""
    # Input dimension: 2 (t, r)
    # Output dimension: M (latent dimension)
    layers: list
    p: int
    activation: Callable = nn.relu


    @nn.compact
    def __call__(self, y):
        # y.shape = (K, 2)
        # Trunk net applied pointwise, can be vectorized over the batch dimension K.
        # MLP expects (batch, input_dim).
        for layer in self.layers:
            y = nn.Dense(layer, kernel_init=glorot_normal())(y)
        tau = nn.Dense(self.p, kernel_init=glorot_normal())(y)

        return tau


class DeepONet(nn.Module):
    """DeepONet that combines BranchNet and TrunkNet."""
    # Dimensions:
    # Branch input: (85,)
    # Trunk input: (K, 2)
    # Output: (K,)
    branch_layers: list
    trunk_layers: list
    activation: Callable = nn.relu

    @nn.compact
    def __call__(self, I, c0, trunk_input):
        # branch_input: (85,)
        # trunk_input: (K, 2)

        # 1) Compute branch representation (M,)
        branch_out = BranchNet(layers=self.branch_layers, activation=self.activation)(I, c0)
        # branch_out.shape = (M,)

        # 2) Compute trunk representation for each point (K, M)
        trunk_out = TrunkNet(layers=self.trunk_layers, activation=self.activation, p = self.branch_layers[-1])(trunk_input)
        # trunk_out.shape = (K, M)

        # 3) Compute output as dot product:
        # For each of the K points, output = sum_j (branch_out[j] * trunk_out[i, j])
        # This can be computed as a matrix-vector product:
        return jnp.einsum('jm,m->j', trunk_out, branch_out)


# # Example instantiation:
# # M is a chosen latent dimension. For example, M = 128.
# # We can define a few hidden layers for branch and trunk:
# branch_layers = [64, 64, 64, 64, 64, 64, 64, 128]  # Ends in 128 which is M
# trunk_features = branch_layers[-1]

# deep_onet = DeepONet(branch_features=branch_layers, trunk_features=trunk_features)

# # To initialize the parameters, we need dummy inputs:
# key1, key2 = jax.random.split(jax.random.PRNGKey(0))
# dummy_branch_input = jax.random.normal(key1, (85,))
# dummy_trunk_input = jax.random.normal(key2, (750, 2))

# params = deep_onet.init(jax.random.PRNGKey(42), dummy_branch_input, dummy_trunk_input)
# output = deep_onet.apply(params, dummy_branch_input, dummy_trunk_input)
# print("Output shape:", output.shape)  # should be (750,)

def generate_trunk_points(r_vals, t_vals):
    # By passing r_vals first and using indexing='ij', 
    # R_grid will correspond to radius and T_grid to time.
    # R_grid.shape = (len(r_vals), len(t_vals))
    # T_grid.shape = (len(r_vals), len(t_vals))
    # This means the first dimension is radius and the second is time, 
    # matching how train_cn is indexed (r, t).
    R_grid, T_grid = jnp.meshgrid(r_vals, t_vals, indexing='ij')
    
    # Flatten into K x 2, preserving the (r, t) ordering in the flattening:
    # The resulting points will be in an order consistent with flattening (r, t) arrays.
    trunk_points = jnp.stack([T_grid.flatten(), R_grid.flatten()], axis=-1)
    return trunk_points
