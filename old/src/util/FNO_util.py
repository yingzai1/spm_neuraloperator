import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
#import util.functions as functions
import pybamm
import numpy as np
import functools


def train_test_split(data, N_total, test_ratio=0.1, seed=None):
    """Split data into training and testing sets."""
    rng = np.random.default_rng(seed=seed)
    perm = rng.permutation(N_total)

    # 3 ── choose split point ---------------------------------------------------
    test_size = int(test_ratio * N_total)
    test_idx = perm[:test_size]
    train_idx = perm[test_size:]

    # 4 ── slice every array ----------------------------------------------------
    train_data = {k: v[train_idx] for k, v in data.items()}
    test_data = {k: v[test_idx] for k, v in data.items()}

    return train_data, test_data

def  normalise_diffusion(D, lower=-18, upper=-14):
    return 2*(D-lower)/(upper-lower) - 1


# Pad function
def _pad_along_time(arr, padding):
    # arr shape is (..., 75)
    return np.pad(arr, ((0,0), (padding, padding)))

def _pad_along_r(arr, padding):
    # arr shape is (..., 20)
    return np.pad(arr, ((0,0), (padding, padding)))

def preprocess_data(train_I, train_c0, train_cn, num_samples_I, num_samples_c0, padding_r=2, padding_t=5):
    """
    Pre-process the dataset into channels and padded shapes suitable for the FNO.
    
    Parameters
    ----------
    train_I : np.ndarray
        Current samples of shape (num_samples, 75).
    train_c0 : np.ndarray
        Initial concentration samples of shape (num_samples, 20).
    train_cn : np.ndarray
        Target concentrations of shape (num_samples, 20, 75).

    Returns
    -------
    X : jnp.ndarray
        Preprocessed inputs of shape (num_samples, 24, 85, 4).
    Y : jnp.ndarray
        Preprocessed targets of shape (num_samples, 24, 85, 1).
    """

    num_samples = train_I.shape[0]

    t = np.linspace(0, 1, num_samples_I)
    r = np.linspace(0, 1, num_samples_c0)

    R_orig, T_orig = np.meshgrid(r, t, indexing='ij')  # (20,75) each

    R =np.pad(R_orig, ((padding_r, padding_r),(padding_t, padding_t)))
    T = np.pad(T_orig, ((padding_r, padding_r),(padding_t, padding_t)))
    # R, T now have shape (24,85)

    # Original resolutions without padding
    H_orig = num_samples_c0
    W_orig = num_samples_I

    H = H_orig + 2 * padding_r  # 20 + 2*2 = 24
    W = W_orig + 2 * padding_t  # 75 + 2*5 = 85


    # Prepare arrays for output
    X = np.zeros((num_samples, H, W, 4))
    Y = np.zeros((num_samples, H, W, 1))

    # Pad I and c0 along appropriate axes
    # For each sample:
    # train_I[n]: shape (75,) -> pad to (85,)
    # train_c0[n]: shape (20,) -> pad to (24,)

    I_padded = _pad_along_time(train_I, padding_t)    # (num_samples, 85)
    c0_padded = _pad_along_r(train_c0, padding_r)      # (num_samples, 24)

    # Broadcast to (24,85) for each sample
    # I_padded: (num_samples, 85) -> (num_samples,24,85)
    # replicate along r-dim
    I_2D = np.tile(I_padded[:, None, :], (1, H, 1))

    # c0_padded: (num_samples,24) -> (num_samples,24,85)
    # replicate along t-dim
    c0_2D = np.tile(c0_padded[:, :, None], (1, 1, W))


    # R,T are the same for all samples, just expand
    R_3D = np.tile(R[None, ...], (num_samples, 1, 1)) # (num_samples,24,85)
    T_3D = np.tile(T[None, ...], (num_samples, 1, 1)) # (num_samples,24,85)

    # Stack channels: (num_samples,24,85,4)
    X = np.stack([I_2D, c0_2D, R_3D, T_3D], axis=-1)
    # X = np.stack([I_2D, c0_2D, Ds_2D], axis=-1) #Here we try to see if it works without the grid information

    # Process targets (train_cn): (num_samples,20,75)
    # pad targets along r and t to (24,85)
    # Along r: pad (20,) to (24,)
    # Along t: pad (75,) to (85,)

    # Pad cn along both dimensions:
    # We'll do this per-sample:
    cn_padded = np.pad(train_cn, ((0,0),(padding_r,padding_r),(padding_t,padding_t))) # (num_samples,24,85)

    # Add channel dimension for targets: (num_samples,24,85,1)
    Y = cn_padded[..., None]

    return X,Y#jax.device_put(X, cpu), jax.device_put(Y, cpu)

def remove_padding(data, padding_r, padding_t):
    return data[:, padding_r:-padding_r, padding_t:-padding_t, :]  # Adjust slicing to remove padding


def data_loader(X: np.ndarray, D, Y: np.ndarray, batch_size: int, shuffle: bool = True):
    """
    A simple data loader that yields batches of preprocessed FNO data.
    
    Parameters
    ----------
    X : jnp.ndarray
        Preprocessed input data of shape (num_samples, 24,85,4).
    Y : jnp.ndarray
        Preprocessed target data of shape (num_samples, 24,85,1).
    batch_size : int
        Number of samples per batch.
    shuffle : bool
        Whether to shuffle the dataset before yielding batches.

    Yields
    ------
    (X_batch, Y_batch) : tuple of jnp.ndarray
        X_batch : (batch_size, 24,85,4)
        Y_batch : (batch_size, 24,85,1)
    """
    num_samples = X.shape[0]
    indices = np.arange(num_samples)
    if shuffle:
        np.random.shuffle(indices)
    for start_idx in range(0, num_samples, batch_size):
        batch_indices = indices[start_idx:start_idx+batch_size]
        yield X[batch_indices],D[batch_indices], Y[batch_indices]

def data_loader_pe(X: np.ndarray, D,R, Y: np.ndarray, batch_size: int, shuffle: bool = True):
    """
    A simple data loader that yields batches of preprocessed FNO data.
    
    Parameters
    ----------
    X : jnp.ndarray
        Preprocessed input data of shape (num_samples, 24,85,4).
    Y : jnp.ndarray
        Preprocessed target data of shape (num_samples, 24,85,1).
    batch_size : int
        Number of samples per batch.
    shuffle : bool
        Whether to shuffle the dataset before yielding batches.

    Yields
    ------
    (X_batch, Y_batch) : tuple of jnp.ndarray
        X_batch : (batch_size, 24,85,4)
        Y_batch : (batch_size, 24,85,1)
    """
    num_samples = X.shape[0]
    indices = np.arange(num_samples)
    if shuffle:
        np.random.shuffle(indices)
    for start_idx in range(0, num_samples, batch_size):
        batch_indices = indices[start_idx:start_idx+batch_size]
        yield X[batch_indices],D[batch_indices], R[batch_indices], Y[batch_indices]

def data_loader_pe2(X: np.ndarray, D,R,L,eps,A, Y: np.ndarray, batch_size: int, shuffle: bool = True):
    """
    A simple data loader that yields batches of preprocessed FNO data.
    
    Parameters
    ----------
    X : jnp.ndarray
        Preprocessed input data of shape (num_samples, 24,85,4).
    Y : jnp.ndarray
        Preprocessed target data of shape (num_samples, 24,85,1).
    batch_size : int
        Number of samples per batch.
    shuffle : bool
        Whether to shuffle the dataset before yielding batches.

    Yields
    ------
    (X_batch, Y_batch) : tuple of jnp.ndarray
        X_batch : (batch_size, 24,85,4)
        Y_batch : (batch_size, 24,85,1)
    """
    num_samples = X.shape[0]
    indices = np.arange(num_samples)
    if shuffle:
        np.random.shuffle(indices)
    for start_idx in range(0, num_samples, batch_size):
        batch_indices = indices[start_idx:start_idx+batch_size]
        yield X[batch_indices],D[batch_indices], R[batch_indices], L[batch_indices], eps[batch_indices], A[batch_indices], Y[batch_indices]


@functools.partial(jax.jit, static_argnames=("padding_r", "padding_t"))  # everything is fused & runs on the accelerator
def preprocess_data_fast(train_I, train_c0,
                        padding_r: int = 2,
                        padding_t: int = 5):
    """
    Pre-process the dataset into channels and padded shapes suitable for the FNO.

    Parameters
    ----------
    train_I  : array (N, 75)      – current (time) history
    train_c0 : array (N, 20)      – initial concentration
    train_cn : array (N, 20, 75)  – target concentration
    padding_r, padding_t          – radial / temporal padding

    Returns
    -------
    X : array (N, 24, 85, 4)  – 4-channel input tensor
    Y : array (N, 24, 85, 1)  – 1-channel target tensor
    """
    # ---------- constants ----------
    N, W_orig   = train_I.shape          # 75
    H_orig      = train_c0.shape[1]      # 20
    H, W        = H_orig + 2*padding_r, W_orig + 2*padding_t   # 24, 85

    # ---------- coordinate grid (shared by all samples) ----------
    t = jnp.linspace(0.0, 1.0, W_orig)
    r = jnp.linspace(0.0, 1.0, H_orig)
    R, T = jnp.meshgrid(r, t, indexing='ij')                   # (20, 75)
    R = jnp.pad(R, ((padding_r, padding_r), (padding_t, padding_t)))
    T = jnp.pad(T, ((padding_r, padding_r), (padding_t, padding_t)))
    # broadcast cheaply – no data copy!
    R = jnp.broadcast_to(R, (N, H, W))
    T = jnp.broadcast_to(T, (N, H, W))

    # ---------- pad & broadcast the inputs ----------
    I_pad  = jnp.pad(train_I,  ((0, 0), (padding_t, padding_t)))          # (N, 85)
    c0_pad = jnp.pad(train_c0, ((0, 0), (padding_r, padding_r)))          # (N, 24)

    I_2D  = jnp.broadcast_to(I_pad[:,  None, :], (N, H, W))               # (N,24,85)
    c0_2D = jnp.broadcast_to(c0_pad[:, :, None], (N, H, W))               # (N,24,85)

    # ---------- assemble channels ----------
    X = jnp.stack((I_2D, c0_2D, R, T), axis=-1)                           # (N,24,85,4)
    
    return X.astype(jnp.float32)


def data_loader_noD(X: np.ndarray, Y: np.ndarray, batch_size: int, shuffle: bool = True):
    """
    A simple data loader that yields batches of preprocessed FNO data.
    
    Parameters
    ----------
    X : jnp.ndarray
        Preprocessed input data of shape (num_samples, 24,85,4).
    Y : jnp.ndarray
        Preprocessed target data of shape (num_samples, 24,85,1).
    batch_size : int
        Number of samples per batch.
    shuffle : bool
        Whether to shuffle the dataset before yielding batches.

    Yields
    ------
    (X_batch, Y_batch) : tuple of jnp.ndarray
        X_batch : (batch_size, 24,85,4)
        Y_batch : (batch_size, 24,85,1)
    """
    num_samples = X.shape[0]
    indices = np.arange(num_samples)
    if shuffle:
        np.random.shuffle(indices)
    for start_idx in range(0, num_samples, batch_size):
        batch_indices = indices[start_idx:start_idx+batch_size]
        yield X[batch_indices], Y[batch_indices]