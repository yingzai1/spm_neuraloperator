import jax
import jax.numpy as jnp
from flax import linen as nn

def complex_mul(a, b):
    """Complex multiplication helper."""
    return jnp.stack([
        a[..., 0]*b[..., 0] - a[..., 1]*b[..., 1],
        a[..., 0]*b[..., 1] + a[..., 1]*b[..., 0]
    ], axis=-1)

class FourierLayer(nn.Module):
    """A single Fourier layer for a single-sample input of shape (H, W, C)."""
    k_modes: int
    out_channels: int

    @nn.compact
    def __call__(self, x):
        """
        x: shape (H, W, in_channels) -- no batch dimension
        returns: shape (H, W, out_channels)
        """

        # 1) Perform Fourier transform along the (H, W) axes
        # shape after rfftn: (H, W_half+1, in_channels) of complex data
        x_ft = jnp.fft.rfftn(x, axes=(0, 1), norm=None)

        H, W_half = x_ft.shape[0], x_ft.shape[1]
        in_channels = x.shape[-1]
        # The real spatial width is 2*(W_half-1) or so, but let's get W from x.shape
        W = x.shape[1]

        # 2) Truncate to k_modes
        h_modes = min(self.k_modes, H)
        w_modes = min(self.k_modes, W_half)
        x_ft_trunc = x_ft[:h_modes, :w_modes, :]

        # 3) Trainable weights in Fourier space
        # shape (h_modes, w_modes, in_channels, out_channels)
        W_real = self.param('W_real', nn.initializers.he_normal(),
                            (h_modes, w_modes, in_channels, self.out_channels))
        W_imag = self.param('W_imag', nn.initializers.zeros,
                            (h_modes, w_modes, in_channels, self.out_channels))

        # 4) Separate real and imag for x_ft_trunc
        x_real = jnp.real(x_ft_trunc)  # shape(h_modes, w_modes, in_channels)
        x_imag = jnp.imag(x_ft_trunc)
        # stack last axis => shape(h_modes, w_modes, in_channels, 2)
        x_complex = jnp.stack([x_real, x_imag], axis=-1)

        # likewise for the weights => shape(h_modes, w_modes, in_channels, out_channels, 2)
        W_complex = jnp.stack([W_real, W_imag], axis=-1)

        # 5) Complex "matmul": sum over in_channels dimension
        # We'll define a small helper to do the einsum
        def complex_mmul(x_c, w_c):
            """
            x_c: (h_modes, w_modes, in_channels, 2)
            w_c: (h_modes, w_modes, in_channels, out_channels, 2)
            Output: (h_modes, w_modes, out_channels, 2)
            """
            # x_c[..., 0] is real part, x_c[..., 1] is imag
            # same for w_c
            # We'll do a double sum with in_channels
            # shape(h_modes, w_modes, out_channels)
            out_re = (jnp.einsum('hwi,hwio->hwo', x_c[..., 0], w_c[..., 0])
                      - jnp.einsum('hwi,hwio->hwo', x_c[..., 1], w_c[..., 1]))
            out_im = (jnp.einsum('hwi,hwio->hwo', x_c[..., 0], w_c[..., 1])
                      + jnp.einsum('hwi,hwio->hwo', x_c[..., 1], w_c[..., 0]))
            return jnp.stack([out_re, out_im], axis=-1)

        out_complex = complex_mmul(x_complex, W_complex)
        # out_complex shape: (h_modes, w_modes, out_channels, 2)

        # 6) Pad back to original Fourier size
        pad_h = H - h_modes
        pad_w = W_half - w_modes
        # out_ft shape => (H, W_half, out_channels, 2)
        out_ft = jnp.pad(out_complex,
                         ((0, pad_h), (0, pad_w), (0, 0), (0, 0)))
        # => (H, W_half, out_channels, 2)

        # 7) Convert to complex
        out_ft_complex = out_ft[..., 0] + 1j * out_ft[..., 1]
        # shape (H, W_half, out_channels)

        # 8) Inverse FFT
        # Now the last dimension is "out_channels" that doesn't participate in the FFT,
        # so we do an inverse along axes (0,1).
        # We'll vmap or reshape, or do a separate iFFT for each out_channel. Let's do a small loop:
        # Alternatively, we can do an iFFT channel by channel.
        # Or do a dimension roll so out_channels is the 0th dimension, iFFT each.
        # For simplicity, let's do a python loop:
        def ifft_per_channel(cslice):
            return jnp.fft.irfftn(cslice, s=(H, W), axes=(0,1), norm=None)
        # out shape => (H, W)

        # For vectorized approach, we can vmap across out_channels dimension:
        out_list = jax.vmap(ifft_per_channel, in_axes=2, out_axes=2)(out_ft_complex)
        # out_list shape => (H, W, out_channels)

        return out_list

class FNO(nn.Module):
    k_modes: int
    fno_depth: int
    input_channels: int = 4
    hidden_channels: int = 32
    output_channels: int = 1
    param_channels: int = 0

    @nn.compact
    def __call__(self, x):
        """
        x: shape (H, W, input_channels)
        returns: shape (H, W, output_channels)
        """
        # If param_channels != 0, we replicate that logic for a single sample
        if self.param_channels != 0:
            # Suppose x[...,-1:] is shape(H,W,1) => we do the same logic but remove the batch references
            d = x[..., -1:]  # shape(H,W,1)
            x = x[..., :-1]  # shape(H,W, input_channels-1)

            # "Take the mean" across (H,W) => shape(1,) for a single sample
            d_scalar = jnp.mean(d, axis=(0,1))  # shape ()

            # MLP for scalar => param_channels. We do a small Dense-based embedding.
            d_emb = nn.Dense(self.param_channels)(d_scalar[None])  # shape(1,param_channels)
            d_emb = nn.relu(d_emb)
            d_emb = nn.Dense(self.param_channels)(d_emb)
            d_emb = nn.relu(d_emb)
            # shape(1,param_channels)

            # broadcast to (H,W,param_channels)
            H, W = x.shape[0], x.shape[1]
            d_emb_2d = jnp.tile(d_emb[None, ...], (H*W, 1))  # shape(H*W, param_channels)
            d_emb_2d = d_emb_2d.reshape(H, W, self.param_channels)
            x = jnp.concatenate((x, d_emb_2d), axis=-1)
            # Now x has shape(H,W, input_channels-1 + param_channels)

        # 1) Lift input to hidden_channels
        x = nn.Dense(self.hidden_channels)(x)

        # 2) Apply several Fourier layers
        for _ in range(self.fno_depth):
            x_res = FourierLayer(k_modes=self.k_modes,
                                 out_channels=self.hidden_channels)(x)
            # Residual + pointwise dense
            x = x_res + nn.Dense(self.hidden_channels)(x)
            x = nn.relu(x)

        # 3) Final projection to output_channels
        x = nn.Dense(self.output_channels)(x)
        return x
