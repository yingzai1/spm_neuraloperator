import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import random

def complex_mul(a, b):
    """Complex multiplication helper."""
    return jnp.stack([
        a[..., 0]*b[..., 0] - a[..., 1]*b[..., 1],
        a[..., 0]*b[..., 1] + a[..., 1]*b[..., 0]
    ], axis=-1)

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
        W_imag = self.param('W_imag', nn.initializers.zeros, (h_modes, w_modes, in_channels, self.out_channels))

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
            out_re = jnp.einsum('bhwi,hwio->bhwo', x_c[..., 0], w_c[..., 0]) - jnp.einsum('bhwi,hwio->bhwo', x_c[..., 1], w_c[..., 1])
            out_im = jnp.einsum('bhwi,hwio->bhwo', x_c[..., 0], w_c[..., 1]) + jnp.einsum('bhwi,hwio->bhwo', x_c[..., 1], w_c[..., 0])
            return jnp.stack([out_re, out_im], axis=-1)

        out_complex = complex_mmul(x_complex, W_complex)

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

        return x_out

class FNO(nn.Module):
    k_modes: int
    fno_depth: int
    input_channels: int = 4
    hidden_channels: int = 32  # channel size, can be set to 2 or 4
    output_channels: int = 1
    param_channels: int = 0

    @nn.compact
    def __call__(self, x):
        # x: (batch, H, W, input_channels)
        # Lifting layer: map input to hidden channels
        if self.param_channels != 0:
            d = x[...,-1:]
            x = x[...,:-1]
            # Suppose the scalar is repeated in space. Take the mean:
            d_scalar = jnp.mean(d, axis=(1,2))    # shape (batch, 1)

            # Pass through an MLP
            d_emb = nn.Dense(self.param_channels)(d_scalar)       # shape (batch, 32)
            d_emb = nn.relu(d_emb)
            d_emb = nn.Dense(self.param_channels)(d_emb) # shape (batch, hidden_channels)
            d_emb = nn.relu(d_emb)

            # Broadcast to match (H, W)
            d_emb = d_emb[:, None, None, :]  # now (batch, 1, 1, hidden_channels)
            d_emb = jnp.tile(d_emb, (1, x.shape[1], x.shape[2], 1))  # (batch, H, W, hidden_channels)
            
            x = jnp.concatenate((x,d), axis =-1)

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

# Example usage:
# key = random.PRNGKey(0)
# model = FNO(k_modes=16, fno_depth=4, d_v=2, hidden_channels=4)
# input = jnp.ones((1,64,64,4))  # batch=1, domain=64x64, channels=4
# variables = model.init(key, input)
# out = model.apply(variables, input)
# out.shape  # should be (1,64,64,2) assuming d_v=2

def check():
    print('Geiloo')
    return