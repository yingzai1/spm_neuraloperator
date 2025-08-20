import jax.numpy as jnp

def filter_anode_cathode(
    anode:   jnp.ndarray,  # shape (batch, dim1, dim2)
    cathode: jnp.ndarray,  # shape (batch, dim1, dim2)
    *,
    anode_lo: float, anode_hi: float,
    cathode_lo: float, cathode_hi: float,
):
    """
    Drop every sample whose anode OR cathode values fall outside
    their respective [lo, hi] intervals.

    Returns
    -------
    anode_filt   : jnp.ndarray
    cathode_filt : jnp.ndarray
    keep_mask    : jnp.ndarray, shape (batch,), dtype=bool
    """
    if anode.shape[0] != cathode.shape[0]:
        raise ValueError("First (batch) dimension must match for anode & cathode")

    # per-sample checks (True if ALL pixels/voxels are in range)
    keep_anode   = jnp.logical_and(anode   >= anode_lo,
                                   anode   <= anode_hi).all(axis=(1, 2))
    keep_cathode = jnp.logical_and(cathode >= cathode_lo,
                                   cathode <= cathode_hi).all(axis=(1, 2))

    keep_mask = jnp.logical_and(keep_anode, keep_cathode)

    return anode[keep_mask], cathode[keep_mask], keep_mask

def mean_absolute_error_set(predictions, ground_truth, axis=(1, 2)):
    # pred and target: (K,) arrays
    return jnp.mean(jnp.abs(predictions.squeeze() - ground_truth), axis=axis)

def mean_squared_error_set(predictions, ground_truth, axis=(1, 2)):
    # pred and target: (K,) arrays
    return jnp.mean((predictions.squeeze() - ground_truth) ** 2, axis=axis)

def relative_l2_error_set(predictions, ground_truth, axis=(1, 2)):
    # pred and target: (K,) arrays
    norm_ground_truth = jnp.linalg.norm(ground_truth, 2, axis=axis)
    norm_diff = jnp.linalg.norm(predictions.squeeze() - ground_truth, 2, axis=axis)

    return norm_diff / norm_ground_truth

def relative_linf_error_set(predictions, ground_truth, axis=(1, 2)):
    # pred and target: (K,) arrays
    norm_ground_truth = jnp.linalg.norm(ground_truth, jnp.inf, axis=axis)
    # print(ground_truth.shape, predictions.shape)
    # print(ground_truth.squeeze().shape, predictions.squeeze().shape)
    norm_diff = jnp.linalg.norm(predictions.squeeze() - ground_truth, jnp.inf, axis=axis)

    return norm_diff / norm_ground_truth

def calc_error_metrics(predictions, ground_truth, axis=(1, 2)):
    """
    Calculate error metrics for a set of predictions against ground truth.

    Parameters
    ----------
    predictions : jnp.ndarray, shape (K, dim1, dim2)
        Predicted values.
    ground_truth : jnp.ndarray, shape (K, dim1, dim2)
        Ground truth values.

    Returns
    -------
    errors : dict
        Dictionary containing the calculated error metrics.
    """
    return {
        'mae': mean_absolute_error_set(predictions, ground_truth, axis=axis),
        'mse': mean_squared_error_set(predictions, ground_truth, axis=axis),
        'rel_l2': relative_l2_error_set(predictions, ground_truth, axis=axis),
        'rel_linf': relative_linf_error_set(predictions, ground_truth, axis=axis),
    }

def calc_error_metrics_all(*metric_dicts):

    # Ensure all dictionaries share the same keys
    keys = metric_dicts[0].keys()
    if not all(d.keys() == keys for d in metric_dicts):
        raise ValueError("All metric dictionaries must have identical keys.")

    averaged = {}
    for k in keys:
        stacked = jnp.stack([d[k] for d in metric_dicts])  # shape (n_dicts, ...)
        averaged[k] = stacked.mean(axis=0)                 # preserves trailing dims

    return averaged