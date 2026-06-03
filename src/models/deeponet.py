import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Callable, Sequence
from flax.linen.initializers import glorot_normal

activation = nn.relu

class BranchNet(nn.Module):
    """Branch network for DeepONet."""
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
        a = self.layer_1(a)
        
        for layer in self.network:
            a = layer(a)
            a = activation(a)

        beta = self.output_layer(a)
        return beta

class ResBlock(nn.Module):
    """Residual block for the branch network."""
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
    layers: list
    p: int
    activation: Callable = nn.relu

    @nn.compact
    def __call__(self, y):
        # y.shape = (K, 2)
        # Trunk net applied pointwise, can be vectorized over the batch dimension K.
        for layer in self.layers:
            y = nn.Dense(layer, kernel_init=glorot_normal())(y)
        tau = nn.Dense(self.p, kernel_init=glorot_normal())(y)
        return tau

class DeepONet(nn.Module):
    """DeepONet that combines BranchNet and TrunkNet."""
    branch_layers: list
    trunk_layers: list
    activation: Callable = nn.relu

    @nn.compact
    def __call__(self, I, c0, trunk_input):
        # branch_input: current and initial concentration
        # trunk_input: (K, 2) spatial-temporal coordinates
        
        # 1) Compute branch representation (M,)
        branch_out = BranchNet(layers=self.branch_layers, activation=self.activation)(I, c0)
        
        # 2) Compute trunk representation for each point (K, M)
        trunk_out = TrunkNet(layers=self.trunk_layers, activation=self.activation, p=self.branch_layers[-1])(trunk_input)
        
        # 3) Compute output as dot product:
        # For each of the K points, output = sum_j (branch_out[j] * trunk_out[i, j])
        return jnp.einsum('jm,m->j', trunk_out, branch_out)

def generate_trunk_points(r_vals, t_vals):
    """Generate trunk points for DeepONet evaluation."""
    # By passing r_vals first and using indexing='ij', 
    # R_grid will correspond to radius and T_grid to time.
    R_grid, T_grid = jnp.meshgrid(r_vals, t_vals, indexing='ij')
    
    # Flatten into K x 2, preserving the (r, t) ordering in the flattening:
    trunk_points = jnp.stack([T_grid.flatten(), R_grid.flatten()], axis=-1)
    return trunk_points 