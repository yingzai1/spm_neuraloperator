import numpy as np
from scipy.stats import qmc

def generate_sobol_samples(samples_per_soc, config):
    """Generates Sobol samples based on the provided configuration."""
    param_info = config.get("parameter_bounds", {}) or {}
    soc_levels = np.asarray(config["soc_levels"])
    n_total = len(soc_levels) * samples_per_soc

    param_keys = list(param_info.keys())
    
    # Handle case where no parameters are defined for sampling
    if not param_keys:
        # Only sample SoC levels
        samples = {}
        if len(soc_levels) == 1:
            # Single SoC level - create n_total samples all with the same SoC
            samples["soc"] = np.full(n_total, soc_levels[0])
        else:
            # Multiple SoC levels - use Sobol sampling for SoC only
            sampler = qmc.Sobol(d=1, scramble=True, seed=None)
            n_power_of_2 = 2**int(np.ceil(np.log2(n_total)))
            u = sampler.random(n_power_of_2)[:n_total]
            soc_indices = (u[:, 0] * len(soc_levels)).astype(int)
            soc_indices = np.minimum(soc_indices, len(soc_levels) - 1)
            samples["soc"] = soc_levels[soc_indices]
        return samples
    
    # Original logic for when parameters are defined
    param_bounds = [v[1] for v in param_info.values()]
    lower_bounds, upper_bounds = zip(*param_bounds)

    # Create the sampler for SOC + all other parameters
    sampler = qmc.Sobol(d=1 + len(param_keys), scramble=True, seed=None)

    # CORRECTED: Round up to the next power of 2 for Sobol properties
    n_power_of_2 = 2**int(np.ceil(np.log2(n_total)))
    u = sampler.random(n_power_of_2)
    u = u[:n_total] # Truncate to the exact number of samples needed

    # Scale the samples
    all_lower = [0] + list(lower_bounds)
    all_upper = [1] + list(upper_bounds)
    u_scaled = qmc.scale(u, all_lower, all_upper)

    # Assign to dictionary
    samples = {}
    soc_indices = (u_scaled[:, 0] * len(soc_levels)).astype(int)
    # Ensure indices are within bounds
    soc_indices = np.minimum(soc_indices, len(soc_levels) - 1)
    samples["soc"] = soc_levels[soc_indices]

    for i, key in enumerate(param_keys, 1):
        samples[key] = u_scaled[:, i]

    return samples
