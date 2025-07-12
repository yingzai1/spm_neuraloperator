import numpy as np

def _constant_current(C, t):
    """Generates a constant current."""
    return np.full_like(t, C)

def _triangle_current(peak_current, t):
    """
    Generates a symmetric triangle wave current profile.
    The current ramps up to the peak in the first half and back down.
    """
    t_half = t.max() / 2
    # Ramp up from 0 to t_half, then ramp down from t_half to t.max()
    return np.where(t <= t_half, t / t_half, (t.max() - t) / t_half) * peak_current

def _pls_current(C, t_max, rng, t):
    """Generates a Pseudo-Random Level Signal (PLS) current."""
    pulses_per_hour = rng.integers(1, 11)
    n_pulses = max(1, int(pulses_per_hour * t_max / 3600))
    direction = rng.choice([1, -1])
    peak = direction * C * rng.uniform(0.2, 1.5)
    period = t_max / n_pulses
    duty_cycle = rng.uniform(0.2, 0.7)
    pulse_width = period * duty_cycle
    starts = np.arange(n_pulses) * period

    I = np.zeros_like(t, dtype=float)
    for st in starts:
        on = (t >= st) & (t < st + pulse_width)
        I[on] = peak
    return I

def _grf_current(C, t_max, rng, t):
    """
    Generates a Gaussian Random Field (GRF) current profile.
    """
    n = t.shape[0]
    t_i = t.reshape(-1, 1)
    t_j = t.reshape(1, -1)

    # Periodic covariance function
    delta_t = np.pi * (t_i - t_j) / t_max
    cov = np.exp(-2 * (np.sin(delta_t) / 1.0) ** 2)
    cov += 1e-6 * np.eye(n) # Jitter

    mean = np.zeros(n)
    field = rng.multivariate_normal(mean, cov)
    field_clipped = np.clip(field, -1.5, 1.5)

    return np.interp(t, t, field_clipped) * C

def generate_current_profile(family, C, t_max, rng, num_points=75):
    """Factory function to generate a single current profile."""
    t = np.linspace(0, t_max, num_points)
    
    if family == "CC":
        val = rng.uniform(-1, 1) * C
        return _constant_current(val, t)
    
    elif family == "PLS":
        return _pls_current(C, t_max, rng, t)
        
    elif family == "GRF":
        return _grf_current(C, t_max, rng, t)

    elif family == "Triangle":
        peak = rng.uniform(-1, 1) * C
        return _triangle_current(peak, t)

    else:
        raise ValueError(f"Unknown current family: {family}")
