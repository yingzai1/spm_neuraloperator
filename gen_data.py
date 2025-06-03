import os
import numpy as np
import pybamm
from scipy.stats import qmc
import multiprocessing as mp

import src.util.functions as functions


def get_soc_and_diffusivities_sobol(
    soc_levels,
    total_samples,
    log_lower=-15.,
    log_upper=-12.,
    rng=None,
):
    """Generate [soc, log10(D_an), log10(D_ca)] samples using a Sobol sequence."""
    n_soc = len(soc_levels)
    sampler = qmc.Sobol(d=3, scramble=True, seed=rng)
    u = sampler.random(total_samples)
    soc_levels = np.asarray(soc_levels)
    soc_idx = (u[:, 0] * n_soc).astype(int)
    soc_col = soc_levels[soc_idx]
    log_cols = qmc.scale(u[:, 1:], log_lower, log_upper)
    return np.column_stack((soc_col, log_cols))


def generate_currents(rng, num, family="CC", C=1.0, t=None):
    """Generate a list of current profiles."""
    if t is None:
        raise ValueError("Time vector t must be provided")

    I_list = []
    if family == "GRF":
        for _ in range(num):
            s = rng.integers(0, 2**31 - 1)
            I_func = functions.GaussianRFCurrent(s, C, t[-1])
            I_list.append(I_func(t))
    elif family == "Triangle":
        for _ in range(num):
            value = rng.uniform(-1, 1)
            I_func = functions.TriangleCurrent(value * C)
            I_list.append(I_func(t))
    elif family == "CC":
        for _ in range(num):
            value = rng.uniform(-1, 1)
            I_func = functions.ConstantCurrent(value * C)
            I_list.append(I_func(t))
    else:
        raise ValueError(f"Unknown current family: {family}")
    return I_list


def run_single(sim, t, soc, dan_log, dca_log, current):
    """Run a single PyBaMM simulation and return concentrations."""
    params = pybamm.ParameterValues("Prada2013")
    params["Current function [A]"] = pybamm.Interpolant(t, -1.0 * current, pybamm.t)
    params["Negative particle diffusivity [m2.s-1]"] = 10 ** dan_log
    params["Positive particle diffusivity [m2.s-1]"] = 10 ** dca_log
    sim.parameter_values = params
    sol = sim.solve(initial_soc=soc, t_eval=t)

    c0_anode = sol["Negative particle concentration"].entries[:, 0, 0]
    cn_anode = sol["Negative particle concentration"].entries[:, 0, :]
    c0_cathode = sol["Positive particle concentration"].entries[:, 0, 0]
    cn_cathode = sol["Positive particle concentration"].entries[:, 0, :]

    return cn_anode, c0_anode, cn_cathode, c0_cathode


def run_batch(idx, soc_chunk, dan_chunk, dca_chunk, child_seed, family, sim, t, C):
    """Worker function for multiprocessing. Returns results tagged with the chunk index."""
    rng = np.random.default_rng(child_seed)
    currents = generate_currents(rng, soc_chunk.shape[0], family=family, C=C, t=t)

    results = [
        run_single(sim, t, s[0], d[0], c[0], I)
        for s, d, c, I in zip(soc_chunk, dan_chunk, dca_chunk, currents)
    ]
    data = tuple(np.array(x) for x in zip(*results))
    return idx, data, np.array(currents)


def generate_data_parallel(payload, n_workers=48):
    """Run data generation in parallel and concatenate results.

    The payload items are tagged with their original chunk index so that the
    final arrays can be ordered consistently regardless of execution order.
    """
    assert n_workers == len(payload)
    with mp.Pool(n_workers) as pool:
        parts = pool.starmap(run_batch, payload)

    # sort results by the provided index to maintain deterministic ordering
    parts.sort(key=lambda x: x[0])
    data_parts = [p[1] for p in parts]
    current_parts = [p[2] for p in parts]
    cn_anode_parts, c0_anode_parts, cn_cathode_parts, c0_cathode_parts = zip(*data_parts)

    data = (
        np.concatenate(cn_anode_parts),
        np.concatenate(c0_anode_parts),
        np.concatenate(cn_cathode_parts),
        np.concatenate(c0_cathode_parts),
        np.concatenate(current_parts),
    )
    return data


def save_dataset(family, soc, dan, dca, currents, cn_anode, c0_anode, cn_cathode, c0_cathode):
    """Save generated arrays to an npz file."""
    os.makedirs("data", exist_ok=True)
    filename = f"data/{family}_{len(currents)}.npz"
    np.savez(
        filename,
        soc=soc,
        Dan=dan,
        Dca=dca,
        cn_anode=cn_anode,
        c0_anode=c0_anode,
        cn_cathode=cn_cathode,
        c0_cathode=c0_cathode,
        current=currents,
    )
    print(f"Saved data to {filename}")


def main():
    soc_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    families = ["GRF", "Triangle", "CC"]
    samples_per_soc = 20 * 15
    n_workers = 40

    n_total = len(soc_levels) * samples_per_soc
    lower, upper = -15.0, -12.0
    seed = 42
    rng = np.random.default_rng(seed)

    sample_space = get_soc_and_diffusivities_sobol(
        soc_levels, total_samples=n_total, log_lower=lower, log_upper=upper, rng=rng
    )
    soc_stack, dan_stack, dca_stack = np.split(sample_space, 3, axis=1)

    indices = np.array_split(np.arange(n_total), n_workers)

    spm = pybamm.lithium_ion.SPM()
    spm.events = []

    params = pybamm.ParameterValues("Prada2013")
    C = params["Nominal cell capacity [A.h]"]
    t_max = 3600
    t = np.linspace(0, t_max, 75)

    child_seeds = np.random.SeedSequence(seed).spawn(n_workers)

    for family in families:
        print(f"Processing family: {family}")

        payload = [
            (
                i,
                soc_stack[indices[i]],
                dan_stack[indices[i]],
                dca_stack[indices[i]],
                int(child_seeds[i].generate_state(1)[0]),
                family,
                spm,
                t,
                C,
            )
            for i in range(n_workers)
        ]

        cn_anode, c0_anode, cn_cathode, c0_cathode, currents = generate_data_parallel(
            payload, n_workers=n_workers
        )

        save_dataset(
            family,
            soc_stack,
            dan_stack,
            dca_stack,
            currents,
            cn_anode,
            c0_anode,
            cn_cathode,
            c0_cathode,
        )
        print(f"Finished processing family: {family}")


if __name__ == "__main__":
    main()