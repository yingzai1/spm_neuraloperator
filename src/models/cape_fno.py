import jax.numpy as jnp
from flax import linen as nn
from .fno import FourierLayer2

class CAPEMask2(nn.Module):
    """Context-Aware Parameter Encoding mask for 2 parameters (D, R)."""
    hidden_size: int
    k_modes: tuple

    @nn.compact
    def __call__(self, x: jnp.ndarray, D: jnp.ndarray, R: jnp.ndarray) -> jnp.ndarray:
        B, H, W, C = x.shape
        output_size = C
        p = jnp.concatenate((D, R), axis=-1)  # Concatenate D and R along the last dimension

        a = nn.Dense(self.hidden_size)(p)
        a = nn.gelu(a)
        a = nn.Dense(output_size)(a)
        a = a[:, None, None, :]
        
        z1 = nn.Conv(output_size, kernel_size=(1, 1))(x)
        z2 = nn.Conv(output_size, kernel_size=(3, 3), padding="SAME", feature_group_count=output_size)(x)
        z3 = FourierLayer2(k_modes=self.k_modes, out_channels=output_size)(x)
        
        v = a * (z1 + z2 + z3)
        y = nn.gelu(nn.Conv(output_size, (1, 1))(x) + v)
        out = nn.Conv(output_size, (1, 1))(y)

        return out

class CAPEFNO2(nn.Module):
    """Context-Aware Parameter Encoding FNO with 2 parameters (diffusivity and radius)."""
    k_modes: tuple
    fno_depth: int
    cape_hidden_size: int
    hidden_channels: int
    input_channels: int
    output_channels: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, D: jnp.ndarray, R: jnp.ndarray) -> jnp.ndarray:
        cape = CAPEMask2(hidden_size=self.cape_hidden_size, k_modes=self.k_modes)
        
        # u_grid: (B,H,W,in_channels)
        x = nn.Dense(self.hidden_channels)(x)
        x = cape(x, D, R)  # apply CAPE before spectral stack
        
        # Apply several Fourier layers
        for _ in range(self.fno_depth):
            x_res = FourierLayer2(k_modes=self.k_modes, out_channels=self.hidden_channels)(x)
            # Add a pointwise nonlinearity, e.g. GELU
            x = x_res + nn.Conv(self.hidden_channels, kernel_size=(1, 1))(x)
            x = nn.relu(x)

        # Projection layer: map back to desired output dimension
        x = nn.Dense(self.output_channels)(x)
        return x 