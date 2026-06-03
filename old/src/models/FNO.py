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
        # Let’s split into real and imaginary parts for parameterization:
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
    k_modes: int
    out_channels: int

    @nn.compact
    def __call__(self, x):
        # x: (batch, H, W, C)

        # Perform Fourier transform
        x_ft = jnp.fft.rfftn(x, axes=(1, 2), norm=None)  # (batch, H, W//2+1, C) for real FFT
        # x_ft is complex, shape: (..., C), last dimension is complex along axis?

        # We can represent the complex numbers as real-valued arrays. jnp.fft.rfftn returns complex64 or complex128.
        # Let’s split into real and imaginary parts for parameterization:
        # shape of x_ft: (batch, H, W//2+1, C_complex)
        # We'll treat channel dimension as last dimension still: 
        # Actually, rfftn in JAX returns complex arrays. We can store weights as complex or as pairs of real parameters.

        # Truncate to k_modes in both spatial dimensions
        # For simplicity, assume H, W might be >= k_modes. We'll slice the top-left corner of the frequency domain:

        H = x_ft.shape[1]
        W = x.shape[2]
        W_half = x_ft.shape[2]
        h_modes = min(self.k_modes, H)
        w_modes = min(self.k_modes, W_half)
        x_ft_trunc = x_ft[:, :h_modes, :w_modes, :]

        # Create trainable parameters for Fourier layer
        # Weight shape: (h_modes, w_modes, in_channels, out_channels)
        # We represent weights as complex: two real-valued parameters for real and imag parts.
        in_channels = x.shape[-1]
        W_real = self.param('W_real', nn.initializers.he_normal(), (h_modes, w_modes, in_channels, self.out_channels))
        W_imag = self.param('W_imag', nn.initializers.he_normal(), (h_modes, w_modes, in_channels, self.out_channels))

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
        # Pad back to original Fourier size
        # Insert zeros for the truncated frequencies:
        pad_h = H - h_modes
        pad_w = W_half - w_modes
        out_ft = jnp.pad(out_complex, ((0,0),(0,pad_h),(0,pad_w),(0,0),(0,0))) 
        # out_ft now is (batch, H, W//2+1, out_channels, 2)

        # Convert back to complex
        out_ft_complex = out_ft[..., 0] + 1j * out_ft[..., 1]

        # Inverse Fourier transform
        x_out = jnp.fft.irfftn(out_ft_complex, s=(H, W), axes=(1, 2), norm=None) 
        # The original spatial dimension must match the inverse transform domain (H,W).

        return x_out.real #cut off the imaginary part, it should be negligible

class FNO(nn.Module):
    k_modes: int
    fno_depth: int
    input_channels: int = 4
    hidden_channels: int = 32  # channel size, can be set to 2 or 4
    output_channels: int = 1

    @nn.compact
    def __call__(self, x):

        x = nn.Dense(self.hidden_channels)(x)

        # Apply several Fourier layers
        for _ in range(self.fno_depth):
            x_res = FourierLayer(k_modes=self.k_modes, out_channels=self.hidden_channels)(x)
            # Add a pointwise nonlinearity, e.g. GELU
            x = x_res + nn.Dense(self.hidden_channels)(x) # Residual connection
            x = nn.relu(x)

        # Projection layer: map back to desired output dimension
        # Suppose output dimension is also hidden_channels or 1; adjust as needed
        x = nn.Dense(self.output_channels)(x)
        return x

class CAPEMask(nn.Module):
    
    hidden_size: int
    k_modes: int

    @nn.compact
    def __call__(self, x:jnp.ndarray, D:jnp.ndarray) -> jnp.ndarray:

        B, H, W, C = x.shape
        output_size = C

        a = nn.Dense(self.hidden_size)(D)
        a = nn.gelu(a)
        a = nn.Dense(output_size)(a)
        a = a[:,None,None,:]
        z1 = nn.Conv(output_size, kernel_size=(1,1))(x)
        z2 = nn.Conv(output_size, kernel_size=(3,3), padding = "SAME", feature_group_count=output_size)(x)
        z3 = FourierLayer(k_modes=self.k_modes,
                    out_channels=output_size)(x)
        
        v = a * (z1 + z2 + z3)
        y  = nn.gelu(nn.Conv(output_size, (1, 1))(x) + v)
        out = nn.Conv(output_size, (1, 1))(y)

        return out
    
class CAPE_FNO(nn.Module):
    
    k_modes: int
    fno_depth: int
    cape_hidden_size: int
    hidden_channels: int
    input_channels: int
    output_channels: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, D: jnp.ndarray) -> jnp.ndarray:

        cape = CAPEMask(hidden_size=self.cape_hidden_size, k_modes = self.k_modes)
        # u_grid: (B,H,W,in_channels)
        x = nn.Dense(self.hidden_channels)(x)
        x = cape(x, D)  # apply CAPE before spectral stack
        # Apply several Fourier layers
        for _ in range(self.fno_depth):
            x_res = FourierLayer(k_modes=self.k_modes, out_channels=self.hidden_channels)(x)
            # Add a pointwise nonlinearity, e.g. GELU
            x = x_res + nn.Conv(self.hidden_channels, kernel_size=(1, 1))(x)
            x = nn.relu(x)

        # Projection layer: map back to desired output dimension
        # Suppose output dimension is also hidden_channels or 1; adjust as needed
        x = nn.Dense(self.output_channels)(x)
        return x
    

class CAPEMask2(nn.Module):
    
    hidden_size: int
    k_modes: int

    @nn.compact
    def __call__(self, x:jnp.ndarray, D:jnp.ndarray, R:jnp.ndarray) -> jnp.ndarray:

        B, H, W, C = x.shape
        output_size = C
        p = jnp.concatenate((D,R),axis=-1)  # Concatenate D and R along the last dimension

        a = nn.Dense(self.hidden_size)(p)
        a = nn.gelu(a)
        a = nn.Dense(output_size)(a)
        a = a[:,None,None,:]
        z1 = nn.Conv(output_size, kernel_size=(1,1))(x)
        z2 = nn.Conv(output_size, kernel_size=(3,3), padding = "SAME", feature_group_count=output_size)(x)
        z3 = FourierLayer2(k_modes=self.k_modes,
                    out_channels=output_size)(x)
        
        v = a * (z1 + z2 + z3)
        y  = nn.gelu(nn.Conv(output_size, (1, 1))(x) + v)
        out = nn.Conv(output_size, (1, 1))(y)

        return out
    
class CAPE_FNO2(nn.Module):
    
    k_modes: int
    fno_depth: int
    cape_hidden_size: int
    hidden_channels: int
    input_channels: int
    output_channels: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, D: jnp.ndarray, R: jnp.ndarray) -> jnp.ndarray:

        cape = CAPEMask2(hidden_size=self.cape_hidden_size, k_modes = self.k_modes)
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
        # Suppose output dimension is also hidden_channels or 1; adjust as needed
        x = nn.Dense(self.output_channels)(x)
        return x
    
class CAPEMask3(nn.Module):
    
    hidden_size: int
    k_modes: int

    @nn.compact
    def __call__(self, x:jnp.ndarray, D:jnp.ndarray, R:jnp.ndarray, L:jnp.ndarray, eps:jnp.ndarray, A:jnp.ndarray) -> jnp.ndarray:

        B, H, W, C = x.shape
        output_size = C
        p = jnp.concatenate((D,R,L,eps,A),axis=-1)  # Concatenate D and R along the last dimension

        a = nn.Dense(self.hidden_size)(p)
        a = nn.gelu(a)
        a = nn.Dense(output_size)(a)
        a = a[:,None,None,:]
        z1 = nn.Conv(output_size, kernel_size=(1,1))(x)
        z2 = nn.Conv(output_size, kernel_size=(3,3), padding = "SAME", feature_group_count=output_size)(x)
        z3 = FourierLayer2(k_modes=self.k_modes,
                    out_channels=output_size)(x)
        
        v = a * (z1 + z2 + z3)
        y  = nn.gelu(nn.Conv(output_size, (1, 1))(x) + v)
        out = nn.Conv(output_size, (1, 1))(y)

        return out
    
class CAPE_FNO3(nn.Module):
    
    k_modes: int
    fno_depth: int
    cape_hidden_size: int
    hidden_channels: int
    input_channels: int
    output_channels: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, D: jnp.ndarray, R: jnp.ndarray, L:jnp.ndarray, eps:jnp.ndarray, A:jnp.ndarray) -> jnp.ndarray:

        cape = CAPEMask3(hidden_size=self.cape_hidden_size, k_modes = self.k_modes)
        # u_grid: (B,H,W,in_channels)
        x = nn.Dense(self.hidden_channels)(x)
        x = cape(x, D, R, L, eps, A)  # apply CAPE before spectral stack
        # Apply several Fourier layers
        for _ in range(self.fno_depth):
            x_res = FourierLayer2(k_modes=self.k_modes, out_channels=self.hidden_channels)(x)
            # Add a pointwise nonlinearity, e.g. GELU
            x = x_res + nn.Conv(self.hidden_channels, kernel_size=(1, 1))(x)
            x = nn.relu(x)

        # Projection layer: map back to desired output dimension
        # Suppose output dimension is also hidden_channels or 1; adjust as needed
        x = nn.Dense(self.output_channels)(x)
        return x