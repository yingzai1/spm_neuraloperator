import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import random
from typing import Tuple

def complex_mul(a, b):
    """Complex multiplication helper."""
    return jnp.stack([
        a[..., 0]*b[..., 0] - a[..., 1]*b[..., 1],
        a[..., 0]*b[..., 1] + a[..., 1]*b[..., 0]
    ], axis=-1)

class FourierLayer2(nn.Module):
    """A single Fourier layer that performs low-rank parameterization in Fourier space."""
    k_modes: Tuple[int, int]
    out_channels: int

    @nn.compact
    def __call__(self, x):
        # x: (batch, H, W, C)
        k_h, k_w = self.k_modes
        # Perform Fourier transform
        x_ft = jnp.fft.rfftn(x, axes=(1, 2), norm=None)  # (batch, H, W//2+1, C) for real FFT
        # x_ft is complex, shape: (..., C), last dimension is complex along axis?

        # We can represent the complex numbers as real-valued arrays. jnp.fft.rfftn returns complex64 or complex128.
        # Let's split into real and imaginary parts for parameterization:
        # shape of x_ft: (batch, H, W//2+1, C_complex)
        # We'll treat channel dimension as last dimension still: 
        # Actually, rfftn in JAX returns complex arrays. We can store weights as complex or as pairs of real parameters.

        # Truncate to k_modes in both spatial dimensions
        # For simplicity, assume H, W might be >= k_modes. We'll slice the top-left corner of the frequency domain:

        H = x_ft.shape[1]
        W = x.shape[2]
        W_half = x_ft.shape[2]
        k_h = min(k_h, (H+1)//2)
        k_w = min(k_w, W_half)

        x_ft_bottom_row = x_ft[:, :k_h, :k_w, :]
        x_ft_top_row = x_ft[:, -k_h:, :k_w, :]
        x_ft_trunc = jnp.concatenate((x_ft_bottom_row, x_ft_top_row), axis=1)

        # Create trainable parameters for Fourier layer
        # Weight shape: (h_modes, w_modes, in_channels, out_channels)
        # We represent weights as complex: two real-valued parameters for real and imag parts.
        in_channels = x.shape[-1]
        W_real = self.param('W_real', nn.initializers.he_normal(), (2*k_h, k_w, in_channels, self.out_channels))
        W_imag = self.param('W_imag', nn.initializers.zeros, (2*k_h, k_w, in_channels, self.out_channels))

        # Apply weights in Fourier space
        # Expand x_ft_trunc to (batch, h_modes, w_modes, in_channels) and do a complex matmul
        # We'll treat them as elementwise multiplications and sums over in_channels.
        # x_ft_trunc: complex array. Let's separate real/imag:
        x_real = jnp.real(x_ft_trunc)
        x_imag = jnp.imag(x_ft_trunc)
        x_complex = jnp.stack([x_real, x_imag], axis=-1)  # (batch, h_modes, w_modes, in_channels, 2)
        W_complex = jnp.stack([W_real, W_imag], axis=-1)  # (h_modes, w_modes, in_channels, out_channels, 2)

        def complex_mmul(x_c, w_c):
            # x_c: (batch, h_modes, w_modes, in_channels, 2)
            # w_c: (h_modes, w_modes, in_channels, out_channels, 2)
            # We'll just do a double sum:
            # out_c: (batch, h_modes, w_modes, out_channels, 2)
            
            # assert x_c.ndim == 5 and w_c.ndim == 5, "need (B,Hₘ,Wₘ,Cin,2)"
            # assert x_c.shape[-1] == 2 and w_c.shape[-1] == 2, "last axis = complex"
            # assert x_c.shape[3] == w_c.shape[2], "Cin mismatch"

            out_re = jnp.einsum('bhwi,hwio->bhwo', x_c[..., 0], w_c[..., 0]) - jnp.einsum('bhwi,hwio->bhwo', x_c[..., 1], w_c[..., 1])
            out_im = jnp.einsum('bhwi,hwio->bhwo', x_c[..., 0], w_c[..., 1]) + jnp.einsum('bhwi,hwio->bhwo', x_c[..., 1], w_c[..., 0])
            return jnp.stack([out_re, out_im], axis=-1)

        out_complex = complex_mmul(x_complex, W_complex)

        # Pad back to original Fourier size
        # Insert zeros for the truncated frequencies:
        # pad_h = H - 2*h_modes
        # pad_w = W_half - w_modes
        # create an all-zero tensor of the full FFT size
        out_ft = jnp.zeros((x.shape[0], H, W_half, self.out_channels, 2),
                           dtype=out_complex.dtype)
        # insert the +kh rows (0 … k-1)
        out_ft = out_ft.at[:,  :k_h, :k_w, :, :].set(out_complex[:, :k_h, ...])

        # insert the –kh rows (H-k … H-1)
        out_ft = out_ft.at[:, -k_h:, :k_w, :, :].set(out_complex[:, k_h:, ...])

        # Convert back to complex
        out_ft_complex = out_ft[..., 0] + 1j * out_ft[..., 1]

        # Inverse Fourier transform
        x_out = jnp.fft.irfftn(out_ft_complex, s=(H, W), axes=(1, 2), norm=None) 
        # The original spatial dimension must match the inverse transform domain (H,W).

        return x_out
    
class FourierLayer(nn.Module):
    """A single Fourier layer that performs low-rank parameterization in Fourier space."""
    k_modes: int  # Number of modes to keep in Fourier space
    out_channels: int
    
    @nn.compact
    def __call__(self, x):
        # x: (batch, H, W, C)
        batch_size, H, W, in_channels = x.shape
        
        # Perform Fourier transform
        x_ft = jnp.fft.rfftn(x, axes=(1, 2), norm=None)  # (batch, H, W//2+1, C)
        
        # Truncate to k_modes
        k_h = min(self.k_modes, H//2)
        k_w = min(self.k_modes, x_ft.shape[2])
        
        # Extract the relevant modes
        x_ft_low = x_ft[:, :k_h, :k_w, :]
        x_ft_high = x_ft[:, -k_h:, :k_w, :] if k_h > 0 else jnp.zeros_like(x_ft_low)
        
        # Combine low and high frequency modes
        x_ft_trunc = jnp.concatenate([x_ft_low, x_ft_high], axis=1)
        
        # Complex weights
        W_real = self.param('W_real', nn.initializers.xavier_normal(), 
                           (2*k_h, k_w, in_channels, self.out_channels))
        W_imag = self.param('W_imag', nn.initializers.xavier_normal(), 
                           (2*k_h, k_w, in_channels, self.out_channels))
        
        # Apply linear transformation in Fourier space
        x_real = jnp.real(x_ft_trunc)
        x_imag = jnp.imag(x_ft_trunc)
        
        out_real = jnp.einsum('bhwi,hwio->bhwo', x_real, W_real) - jnp.einsum('bhwi,hwio->bhwo', x_imag, W_imag)
        out_imag = jnp.einsum('bhwi,hwio->bhwo', x_real, W_imag) + jnp.einsum('bhwi,hwio->bhwo', x_imag, W_real)
        
        # Reconstruct full frequency domain
        out_ft = jnp.zeros((batch_size, H, x_ft.shape[2], self.out_channels), dtype=complex)
        out_complex = out_real + 1j * out_imag
        
        out_ft = out_ft.at[:, :k_h, :k_w, :].set(out_complex[:, :k_h, :, :])
        if k_h > 0:
            out_ft = out_ft.at[:, -k_h:, :k_w, :].set(out_complex[:, k_h:, :, :])
        
        # Inverse Fourier transform
        x_out = jnp.fft.irfftn(out_ft, s=(H, W), axes=(1, 2), norm=None)
        
        return x_out

class FNOBlock(nn.Module):
    """A single FNO block consisting of Fourier layer + MLP."""
    k_modes: int
    hidden_channels: int
    
    @nn.compact
    def __call__(self, x):
        # Fourier layer
        fourier_out = FourierLayer(self.k_modes, self.hidden_channels)(x)
        
        # MLP (applied pointwise)
        mlp_out = nn.Dense(self.hidden_channels)(x)
        mlp_out = nn.gelu(mlp_out)
        mlp_out = nn.Dense(self.hidden_channels)(mlp_out)
        
        # Residual connection and activation
        x_out = fourier_out + mlp_out
        x_out = nn.gelu(x_out)
        
        return x_out

class FNO(nn.Module):
    """Fourier Neural Operator for learning operators on structured grids."""
    k_modes: int = 10
    fno_depth: int = 6  
    hidden_channels: int = 32
    output_channels: int = 1
    
    @nn.compact
    def __call__(self, x):
        # x: (batch, H, W, in_channels)
        
        # Lifting layer
        x = nn.Dense(self.hidden_channels)(x)
        
        # FNO blocks
        for _ in range(self.fno_depth):
            x = FNOBlock(self.k_modes, self.hidden_channels)(x)
        
        # Projection layer
        x = nn.Dense(self.hidden_channels)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.output_channels)(x)
        
        return x 