#!/usr/bin/env python
# coding: utf-8

# In[1]:


# %cd ~/repos/foo_cleaned


# In[2]:


import os
import numpy as np
import pybamm
from scipy.stats import qmc
import src.util.functions as functions
import multiprocessing as mp


# In[3]:


print(os.cpu_count())            # logical cores available to the process
print(mp.cpu_count())  # same thing


# In[4]:


def sobol_sample(total_samples, soc_levels,
                 param_bounds, rng=None):
    """
    Returns:
        samples: dict  { 'soc':  (N,1),   'log_Dan': (N,1), ... }
        order : list   the exact key order used (for saving)
    """
    n_soc  = len(soc_levels)
    keys   = list(param_bounds.keys())          # stable order
    D      = 1 + len(keys)                      # 1 column for SoC + one per param

    sampler = qmc.Sobol(d=D, scramble=True, seed=rng)
    u       = sampler.random(total_samples)

    # --- SoC (discrete) ---
    soc_idx = (u[:, 0] * n_soc).astype(int)
    soc_col = np.asarray(soc_levels)[soc_idx]
    samples = {'soc': soc_col.reshape(-1, 1)}

    # --- continuous parameters ---
    for j, k in enumerate(keys, start=1):
        low, high = param_bounds[k]

        col = qmc.scale(u[:, j:j+1],            # 2-D column slice (N,1)
                [low], [high])          # bounds must be 1-element lists

        samples[k.split("|")[0]] = col.reshape(-1, 1)  # keep short prefix only

    return samples, ['soc'] + [k.split("|")[0] for k in keys]


# In[5]:


def PLSCurrent(C, t_max, rng=np.random.default_rng()):
    """
    Return I_func(t) that produces a rectangular-pulse current profile.

    • pulses_per_hour ∈ [1, 10]   (uniform)
    • All pulses in one sample share ONE peak: ±(0.2‒1.5)·C.
    • Width is a fixed fraction (5‒30 %) of the period → typically 2-30 s.
    """
    pulses_per_hour = rng.integers(1, 11)                    # 1–10 pulses / h
    n_pulses = max(1, int(pulses_per_hour * t_max / 3600))   # fit into t_max

    direction = rng.choice([1, -1])                          # charge or discharge
    peak = direction * C * rng.uniform(0.2, 1.5)             # fixed amplitude

    period = t_max / n_pulses
    duty_cycle = rng.uniform(0.2, 0.7)                     # 20–70 %
    pulse_width = period * duty_cycle                        # hold-time in s

    starts = np.arange(n_pulses) * period                    # equally spaced

    def I_func(t):
        t = np.asarray(t)
        I = np.zeros_like(t, dtype=float)
        for st in starts:
            on = (t >= st) & (t < st + pulse_width)          # rectangular gate
            I[on] = peak
        return I

    return I_func


# In[6]:


def generate_data(key, num, family="CC"):
    """
    Generate a list of current profiles using NumPy for random numbers.

    The seed is used to ensure reproducibility.
    """
    rng = np.random.default_rng(key)

    I_list = []
    if family == "GRF":
        for _ in range(num):
            s = rng.integers(0, 2**31 - 1)
            I_func = functions.GaussianRFCurrent(s, C, t_max)
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

    elif family == "PLS":                                     # ← new branch
        for _ in range(num):
            I_func = PLSCurrent(C, t_max, rng)
            I_list.append(I_func(t))

    return I_list


# In[7]:


def get_targets(I_samples, sample_dict):
    """
    Parameters
    ----------
    I_samples  : list[np.ndarray]      – time-series currents, one per sample
    sample_dict: dict                  – keys = short names ('soc', 'log_Dan', …)
                                         values = (N,1) ndarrays from the Sobol draw

    Returns (exact order)
    ---------------------
    (
        targets_anode, c0_anode,
        targets_cathode, c0_cathode,
        currents, socs,
        *param_lists                                                      # one list
          for every key in sample_dict except 'soc', in the ORIGINAL KEY  # per param
          order appearing in sample_dict                                  #
    )
    """
    N = sample_dict["soc"].shape[0]

    # ➊  storage ------------------------------------------------------------
    tgt_an,  c0_an = [], []
    tgt_ca,  c0_ca = [], []
    currents, socs = [], []

    param_traces = {k: [] for k in sample_dict if k != "soc"}  # dynamic!

    # ➋  loop over samples --------------------------------------------------
    for i, I_func in enumerate(I_samples):
        params_local = params_bat.copy()           # global template

        # --- all Sobol parameters (except SoC) -----------------------------
        for short, arr in sample_dict.items():
            if short == "soc":
                continue
            if short == "A":
                # A is a special case, it is not implemented in pybamm, but width and height are
                params_local["Electrode height [m]"] = arr[i, 0]**0.5
                params_local["Electrode width [m]"] = arr[i, 0]**0.5
                param_traces[short].append(arr[i, 0]) 
                continue

            val = arr[i, 0]
            full_key = SHORT2FULL[short]

            # log-uniform variables carry the 'log_' prefix
            params_local[full_key] = val
            param_traces[short].append(val)

        # --- current profile ----------------------------------------------
        params_local["Current function [A]"] = pybamm.Interpolant(
            t, -1.0 * I_func, pybamm.t
        )

        # --- solve ---------------------------------------------------------
        sim = pybamm.Simulation(spm, parameter_values=params_local)
        try:
            sol = sim.solve(initial_soc=float(sample_dict["soc"][i, 0]), t_eval=t)
        except:
            print(f"Error solving for sample {i}, skipping.")
            continue

        # --- extract targets ----------------------------------------------
        c0_anode = sol["Negative particle concentration"].entries[:, 0, 0]
        cn_anode = sol["Negative particle concentration"].entries[:, 0, :]
        c0_cath  = sol["Positive particle concentration"].entries[:, 0, 0]
        cn_cath  = sol["Positive particle concentration"].entries[:, 0, :]

        # --- collect -------------------------------------------------------
        tgt_an.append(cn_anode);  c0_an.append(c0_anode)
        tgt_ca.append(cn_cath);   c0_ca.append(c0_cath)
        currents.append(I_func);  socs.append(float(sample_dict["soc"][i, 0]))

    # ➌  pack result tuple --------------------------------------------------
    return (
        tgt_an, c0_an, tgt_ca, c0_ca,
        currents, socs,
        *[param_traces[k] for k in param_traces]   # in key order
    )


# In[8]:


import numpy as np
from pathlib import Path


def save_data(
    family: str,
    samples_dict: dict,              # only used to know *which* params we sampled
    data: tuple,
    parameter_set: str = "Prada2013",
    out_dir: str | Path = "data",
):
    """
    Parameters
    ----------
    family        : "CC", "GRF", "Triangle", …
    samples_dict  : the *full* Sobol draw (keys = short names).  Only the keys
                    are used here, so it's fine if you dropped some samples later.
    data          : tuple returned by `generate_data_multiproccess`
    parameter_set : e.g. "Prada2013"
    out_dir       : folder to write the .npz file
    """
    (
        cn_anode,
        c0_anode,
        cn_cathode,
        c0_cathode,
        currents,
        socs,
        *param_arrays,             # 0-to-many arrays, one per extra parameter
    ) = data

    # ---------- build dict for np.savez ------------------------------------
    savez_args = dict(
        cn_anode   = cn_anode,
        c0_anode   = c0_anode,
        cn_cathode = cn_cathode,
        c0_cathode = c0_cathode,
        current    = currents,
        soc        = socs,
    )

    # Parameter arrays come back in the *same order* the keys appear
    # in samples_dict (excluding 'soc') thanks to insertion-order preservation.
    param_keys = [k for k in samples_dict if k != "soc"]

    if len(param_keys) != len(param_arrays):
        raise ValueError(
            f"Mismatch: {len(param_arrays)} arrays for {len(param_keys)} params"
        )

    for key, arr in zip(param_keys, param_arrays):
        savez_args[key] = arr

    # ---------- write file -------------------------------------------------
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_samples = len(socs)
    fname = out_dir / f"{parameter_set}_{family}_{total_samples}.npz"
    np.savez(fname, **savez_args)

    print(f"Saved data to {fname}")


# In[9]:


def run_batch(sample_dict, child_seed, family="CC"):
    """
    sample_dict : dict where each value is (M,1) – the whole chunk handled
                 by this worker. Must contain key 'soc'.
    """
    rng        = np.random.default_rng(child_seed)
    M          = sample_dict["soc"].shape[0]
    I_samples  = generate_data(rng, M, family=family)   # unchanged

    return get_targets(I_samples, sample_dict)


# In[10]:


def generate_data_multiproccess(payload, n_workers=48):
    """
    payload is now a list of      [(sample_dict, child_seed, family), ...]
    created exactly like before, just with one dict per worker.
    """
    assert n_workers == len(payload)

    with mp.Pool(n_workers) as pool:
        data_parts = pool.starmap(run_batch, payload)

    # --- unzip -------------------------------------------------------------
    # basic fields (6) + a variable-length tail of parameter traces
    cols = list(zip(*data_parts))          # list-of-lists, length = 6 + K
    (
        tgt_an_parts,
        c0_an_parts,
        tgt_ca_parts,
        c0_ca_parts,
        curr_parts,
        soc_parts,
        *param_parts_lists,                # here K = number of extra params
    ) = cols

    # --- concatenate basics -----------------------------------------------
    tgt_an_all = np.concatenate(tgt_an_parts, axis=0)
    c0_an_all  = np.concatenate(c0_an_parts, axis=0)
    tgt_ca_all = np.concatenate(tgt_ca_parts, axis=0)
    c0_ca_all  = np.concatenate(c0_ca_parts, axis=0)
    curr_all   = np.concatenate(curr_parts,   axis=0)
    soc_all    = np.concatenate(soc_parts,    axis=0)

    # --- concatenate each parameter trace ---------------------------------
    param_all = [np.concatenate(pp, axis=0) for pp in param_parts_lists]

    # --- build final tuple -------------------------------------------------
    data = (
        tgt_an_all, c0_an_all,
        tgt_ca_all, c0_ca_all,
        curr_all,   soc_all,
        *param_all
    )
    return data


# In[11]:


# ----------------------------------------
# GLOBAL CONFIG ― edit only this section
# ----------------------------------------
SOC_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
families = ["GRF"]
samples_per_soc = 13 #10000   #CFNO is 3000, DON is 2200 but soc only 1 value, FNO is 1000
n_workers = 40

# Every key is (i) a short label you like *and*  
# (ii) the exact PyBaMM parameter name when that differs, separated by "|"
# e.g.  "log_Dan|Negative particle diffusivity [m2.s-1]"
PARAM_BOUNDS = {
    "D_n|Negative particle diffusivity [m2.s-1]" : (1.0e-18, 1.0e-14),
    "D_p|Positive particle diffusivity [m2.s-1]" : (1.0e-18, 1.0e-14),
    "R_n|Negative particle radius [m]"               : (4.0e-6, 1.5e-5),
    "R_p|Positive particle radius [m]"               : (1.0e-8, 1.5e-5),
    "eps_n|Negative electrode active material volume fraction" : (0.3, 0.8),
    "eps_p|Positive electrode active material volume fraction" : (0.3, 0.8),
    "L_n|Negative electrode thickness [m]"           : (2e-5, 10e-5),
    "L_p|Positive electrode thickness [m]"           : (2e-5, 10e-5),
    # one height, one width – both electrodes share them
    # "H_elec|Electrode height [m]"                    : (60e-6, 250e-6),
    # "W_elec|Electrode width [m]"                    : (60e-6, 250e-6),
    "A|Electrode area [m2]"                          : (5e-3, 5e-1),  # 60x60 mm²
}


# In[12]:


# ── mapping short → full PyBaMM name, derived automatically ──────────────
SHORT2FULL = {
    short.split("|")[0]: short.split("|", 1)[1]
    for short in PARAM_BOUNDS                           # PARAM_BOUNDS from earlier
}


# In[13]:


# # N_train = len(soc_levels) * samples_per_soc * num_train
# # N_test = len(soc_levels) * samples_per_soc * num_test

# # N_total = N_train + N_test

# N_TOTAL       = len(SOC_LEVELS) * samples_per_soc
# rng           = np.random.default_rng(42)

# # master sampling
# samples_dict, ordered_names = sobol_sample(N_TOTAL, soc_levels=SOC_LEVELS, param_bounds=PARAM_BOUNDS, rng=rng)
# indices       = np.array_split(np.arange(N_TOTAL), n_workers)

# seed = 42
# rng = np.random.default_rng(seed)

# master_ss   = np.random.SeedSequence(seed)
# child_ss    = master_ss.spawn(n_workers)

# # Create the SPM model and remove events if needed
# spm = pybamm.lithium_ion.SPM()
# spm.events = []

# parameter_names = ["Prada2013"] #["Prada2013","Chen2020","Kokam1","Kokam2"]
# # Global battery parameters and time discretization:
# # params_bat = pybamm.ParameterValues("Prada2013")

# # ## To enforce constant Dan and Dca across all samples, we can use the first sample:
# # Dan_stack = np.full((N_total, 1), np.log10(params_bat["Negative particle diffusivity [m2.s-1]"]))
# # Dca_stack = np.full((N_total, 1), np.log10(params_bat["Positive particle diffusivity [m2.s-1]"]))

# # C = params_bat["Nominal cell capacity [A.h]"]
# # t_max = 3600
# # num_samples_I = 75
# # num_samples_c0 = 20
# # t = np.linspace(0, t_max, num_samples_I)
# # r = np.linspace(0, 1, num_samples_c0)

# for parameter_set in parameter_names:
#     print(f"Using parameter set: {parameter_set}")
#     params_bat = pybamm.ParameterValues(parameter_set)
#     # Dan_stack = np.full((N_total, 1), np.log10(params_bat["Negative particle diffusivity [m2.s-1]"]))
#     # Dca_stack = np.full((N_total, 1), np.log10(params_bat["Positive particle diffusivity [m2.s-1]"]))

#     C = params_bat["Nominal cell capacity [A.h]"]
#     t_max = 3600
#     num_samples_I = 75
#     num_samples_c0 = 20
#     t = np.linspace(0, t_max, num_samples_I)
#     r = np.linspace(0, 1, num_samples_c0)

#     for family in families:
#         print(f"Processing family: {family}")

#         C = params_bat["Nominal cell capacity [A.h]"]
#         t_max = 3600
#         num_samples_I = 75
#         num_samples_c0 = 20
#         t = np.linspace(0, t_max, num_samples_I)
#         r = np.linspace(0, 1, num_samples_c0)

#         # worker payload
#         payload = [
#             ({k: v[idx] for k, v in samples_dict.items()},    # sample_dict
#             int(child_ss[i].generate_state(1)[0]),           # child_seed
#             family)
#             for i, idx in enumerate(indices)
#         ]

#         data = generate_data_multiproccess(payload, n_workers=n_workers)
#         print("Data generation complete.")
#         save_data(family, samples_dict, data, parameter_set)
#         print(f"Saved data for family: {family}")
#         print(f"Finished processing family: {family}")


# In[14]:


prada = pybamm.ParameterValues('Prada2013')
chen = pybamm.ParameterValues('Chen2020')
ecker = pybamm.ParameterValues('Ecker2015')


# In[15]:


prada['Positive particle radius [m]'], chen['Positive particle radius [m]'], ecker['Positive particle radius [m]']


# In[16]:


prada["Negative particle radius [m]"], chen["Negative particle radius [m]"], ecker["Negative particle radius [m]"]


# In[17]:


prada["Negative electrode active material volume fraction"], chen["Negative electrode active material volume fraction"], ecker["Negative electrode active material volume fraction"]


# In[18]:


prada["Negative electrode thickness [m]"], chen["Negative electrode thickness [m]"], ecker["Negative electrode thickness [m]"]


# In[19]:


prada["Positive electrode thickness [m]"], chen["Positive electrode thickness [m]"], ecker["Positive electrode thickness [m]"]


# In[20]:


prada["Electrode height [m]"], chen["Electrode height [m]"], ecker["Electrode height [m]"]


# In[21]:


prada["Electrode width [m]"], chen["Electrode width [m]"], ecker["Electrode width [m]"]


# In[22]:


prada["Negative particle diffusivity [m2.s-1]"], chen["Negative particle diffusivity [m2.s-1]"], ecker["Negative particle diffusivity [m2.s-1]"](0.1,293).value


# In[ ]:


#!/usr/bin/env python3
# ------------------------------------------------------------
#  Batteries‑in‑PyBaMM: Sobol sampling → multiprocessing run
#                       → align rows → save .npz
# ------------------------------------------------------------
import numpy as np, multiprocessing as mp, pybamm
from scipy.stats import qmc
from pathlib import Path

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
SOC_LEVELS        = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
FAMILIES          = ["GRF","CC","Triangle","PLS"]              # one current family for the example
SAMPLES_PER_SOC   = 3002                 # total 13*11 = 143 samples
N_WORKERS         = 40

PARAM_BOUNDS = {
    # short name        | PyBaMM key                                      : (low, high)
    "D_n|Negative particle diffusivity [m2.s-1]"           : (1e-18, 1e-14),
    "D_p|Positive particle diffusivity [m2.s-1]"           : (1e-18, 1e-14),
    "R_n|Negative particle radius [m]"                     : (4e-6, 1.5e-5),
    "R_p|Positive particle radius [m]"                     : (1e-8, 1.5e-5),
    # "eps_n|Negative electrode active material volume fraction": (0.30, 0.80),
    # "eps_p|Positive electrode active material volume fraction": (0.30, 0.80),
    # "L_n|Negative electrode thickness [m]"                 : (2e-5, 10e-5),
    # "L_p|Positive electrode thickness [m]"                 : (2e-5, 10e-5),
    # "A|Electrode area [m2]"                                : (5e-3, 5e-1),
}
SHORT2FULL = {s.split("|")[0]: s.split("|", 1)[1] for s in PARAM_BOUNDS}

# PyBaMM globals (re‑used in every worker)
spm         = pybamm.lithium_ion.SPM(); spm.events = []
params_bat  = pybamm.ParameterValues("Prada2013")
C           = params_bat["Nominal cell capacity [A.h]"]
T_MAX       = t_max = 3600
NUM_T       = 75
t           = np.linspace(0, T_MAX, NUM_T)

# ------------------------------------------------------------
# SAMPLING
# ------------------------------------------------------------
def sobol_sample(N, soc_levels, param_bounds, rng=None):
    n_soc     = len(soc_levels)
    keys      = list(param_bounds.keys())
    sampler   = qmc.Sobol(d=1+len(keys), scramble=True, seed=rng)
    u         = sampler.random(N)

    samples   = {"soc": soc_levels[(u[:, 0]*n_soc).astype(int)].reshape(-1,1)}
    for j, k in enumerate(keys, 1):
        low, high = param_bounds[k]
        col = qmc.scale(u[:, j:j+1], [low], [high])
        samples[k.split("|")[0]] = col
    return samples, ["soc"] + [k.split("|")[0] for k in keys]

# ------------------------------------------------------------
# SIMULATION OF ONE SAMPLE
# ------------------------------------------------------------
def get_targets(I_list, sample_dict):
    """Run 1‑row dict + 1 current → return tuple of lists (length 1)."""
    tgt_an, c0_an, tgt_ca, c0_ca, cur, soc, params = [], [], [], [], [], [], []

    for I, soc0 in zip(I_list, sample_dict["soc"].ravel()):
        p = params_bat.copy()
        for short in (k for k in sample_dict if k != "soc"):
            val = sample_dict[short][0, 0]
            if short == "A":                   # special: derive height/width
                p["Electrode height [m]"] = p["Electrode width [m]"] = val**0.5
            else:
                p[SHORT2FULL[short]] = val

        p["Current function [A]"] = pybamm.Interpolant(t, -I, pybamm.t)
        sim = pybamm.Simulation(spm, parameter_values=p)
        try:
            sol = sim.solve(initial_soc=float(soc0), t_eval=t)
        except (pybamm.SolverError, pybamm.ModelError):
            return None                        # flag failure

        tgt_an.append(sol["Negative particle concentration"].entries[:, 0, :])
        c0_an.append(sol["Negative particle concentration"].entries[:, 0, 0])
        tgt_ca.append(sol["Positive particle concentration"].entries[:, 0, :])
        c0_ca.append(sol["Positive particle concentration"].entries[:, 0, 0])
        cur.append(I); soc.append(soc0)
        params.extend([val for short,val in
                       ((k, sample_dict[k][0,0]) for k in sample_dict if k!="soc")])
    # *params is flattened list, we return separate list for each param later
    return (tgt_an, c0_an, tgt_ca, c0_ca, cur, soc, *[[p] for p in params])

# ------------------------------------------------------------
# WORKER
# ------------------------------------------------------------
def run_batch(sample_dict, child_seed, family):
    rng        = np.random.default_rng(child_seed)
    M          = sample_dict["soc"].shape[0]
    currents   = generate_data(rng, M, family)

    out_lists  = None
    for i in range(M):
        one_dict = {k: v[i:i+1] for k,v in sample_dict.items()}
        res = get_targets([currents[i]], one_dict)
        if res is None:         # solver failed -> skip
            continue
        if out_lists is None:   # first success: create lists of lists
            out_lists = [list(x) for x in res]
        else:
            for lst,x in zip(out_lists, res):
                lst.extend(x)

    if out_lists is None:        # worker produced nothing
        K = len(sample_dict)-1   # num params without 'soc'
        dummy = [[]]*(6+K)
        return tuple(dummy)
    return tuple(np.asarray(lst) for lst in out_lists)

# ------------------------------------------------------------
# PARALLEL FRONT‑END
# ------------------------------------------------------------
def generate_data_multiproccess(payload, n_workers):
    with mp.Pool(n_workers) as pool:
        parts = pool.starmap(run_batch, payload)

    concatenated = tuple(np.concatenate(pieces, axis=0)
                         for pieces in zip(*parts))
    return concatenated

# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------
def save_data(family, samples_dict, data, parameter_set="Prada2013", out_dir="data"):
    (cn_a,c0_a,cn_c,c0_c,curr,soc,*param_arrays) = data
    out = dict(cn_anode=cn_a, c0_anode=c0_a,
               cn_cathode=cn_c, c0_cathode=c0_c,
               current=curr, soc=soc)

    param_keys = [k for k in samples_dict if k!="soc"]
    if len(param_keys)!=len(param_arrays):
        raise ValueError("field count mismatch")

    for k,arr in zip(param_keys, param_arrays):
        out[k]=arr

    Path(out_dir).mkdir(exist_ok=True)
    fname = Path(out_dir)/f"{parameter_set}_{family}_{len(soc)}.npz"
    np.savez(fname, **out)
    print(f"Saved {fname}")

# ------------------------------------------------------------
# MAIN DRIVER
# ------------------------------------------------------------
if __name__ == "__main__":
    N_TOTAL = len(SOC_LEVELS)*SAMPLES_PER_SOC
    samples_dict,_ = sobol_sample(N_TOTAL, np.asarray(SOC_LEVELS),
                                  PARAM_BOUNDS, rng=np.random.default_rng(42))
    indices   = np.array_split(np.arange(N_TOTAL), N_WORKERS)
    seeds     = np.random.default_rng(0).integers(0, 2**31-1, size=N_WORKERS)

    for family in FAMILIES:
        payload = [({k:v[idx] for k,v in samples_dict.items()},
                    int(seeds[i]), family)
                   for i,idx in enumerate(indices)]

        data = generate_data_multiproccess(payload, N_WORKERS)
        save_data(family, samples_dict, data, parameter_set="Prada2013")

