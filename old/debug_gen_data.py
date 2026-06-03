#!/usr/bin/env python3
# ------------------------------------------------------------
#  Batteries‑in‑PyBaMM: Sobol sampling → multiprocessing run
#                       → align rows → save .npz
# ------------------------------------------------------------
import numpy as np, multiprocessing as mp, pybamm
from scipy.stats import qmc
from pathlib import Path
import src.util.functions as functions

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
SOC_LEVELS        = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
FAMILIES          = ["GRF","CC","Triangle","PLS"]
SAMPLES_PER_SOC   = 13
N_WORKERS         = 4

PARAM_BOUNDS = {
    # short name        | PyBaMM key                                      : (low, high)
    "D_n|Negative particle diffusivity [m2.s-1]"           : (1e-18, 1e-14),
    "D_p|Positive particle diffusivity [m2.s-1]"           : (1e-18, 1e-14),
    "R_n|Negative particle radius [m]"                     : (4e-6, 1.5e-5),
    "R_p|Positive particle radius [m]"                     : (1e-8, 1.5e-5),
    "eps_n|Negative electrode active material volume fraction": (0.30, 0.80),
    "eps_p|Positive electrode active material volume fraction": (0.30, 0.80),
    "L_n|Negative electrode thickness [m]"                 : (2e-5, 10e-5),
    "L_p|Positive electrode thickness [m]"                 : (2e-5, 10e-5),
    "A|Electrode area [m2]"                                : (5e-3, 5e-1),
}
SHORT2FULL = {s.split("|")[0]: s.split("|", 1)[1] for s in PARAM_BOUNDS}

# PyBaMM globals (re‑used in every worker)
spm         = pybamm.lithium_ion.SPM(); spm.events = []
params_bat  = pybamm.ParameterValues("Prada2013")
C           = params_bat["Nominal cell capacity [A.h]"]
T_MAX       = t_max = 3600
NUM_T       = 75
t           = np.linspace(0, T_MAX, NUM_T)

def PLSCurrent(C, t_max, rng=np.random.default_rng()):
    pulses_per_hour = rng.integers(1, 11)
    n_pulses = max(1, int(pulses_per_hour * t_max / 3600))
    direction = rng.choice([1, -1])
    peak = direction * C * rng.uniform(0.2, 1.5)
    period = t_max / n_pulses
    duty_cycle = rng.uniform(0.2, 0.7)
    pulse_width = period * duty_cycle
    starts = np.arange(n_pulses) * period
    def I_func(t):
        t = np.asarray(t)
        I = np.zeros_like(t, dtype=float)
        for st in starts:
            on = (t >= st) & (t < st + pulse_width)
            I[on] = peak
        return I
    return I_func

def generate_data(key, num, family="CC"):
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
    elif family == "PLS":
        for _ in range(num):
            I_func = PLSCurrent(C, t_max, rng)
            I_list.append(I_func(t))
    return I_list

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

    # This loop runs once per call from run_batch
    for I, soc0 in zip(I_list, sample_dict["soc"].ravel()):
        p = params_bat.copy()
        param_values_for_this_run = {}
        for short in (k for k in sample_dict if k != "soc"):
            val = sample_dict[short][0, 0]
            param_values_for_this_run[short] = val
            if short == "A":
                p["Electrode height [m]"] = p["Electrode width [m]"] = val**0.5
            else:
                p[SHORT2FULL[short]] = val

        p["Current function [A]"] = pybamm.Interpolant(t, -I, pybamm.t)
        sim = pybamm.Simulation(spm, parameter_values=p, solver=pybamm.CasadiSolver())
        try:
            sol = sim.solve(initial_soc=float(soc0), t_eval=t)
        except (pybamm.SolverError, pybamm.ModelError) as e:
            print(f"Solver Error on sample. Params: {param_values_for_this_run}, SOC: {soc0}. Error: {e}")
            return None

        tgt_an.append(sol["Negative particle concentration"].entries[:, 0, :])
        c0_an.append(sol["Negative particle concentration"].entries[:, 0, 0])
        tgt_ca.append(sol["Positive particle concentration"].entries[:, 0, :])
        c0_ca.append(sol["Positive particle concentration"].entries[:, 0, 0])
        cur.append(I); soc.append(soc0)
        
        # Collect parameters for this run
        run_params = []
        param_keys = [k for k in sample_dict if k != "soc"]
        for k in param_keys:
             run_params.append(sample_dict[k][0,0])
        params.append(run_params)

    # Re-structure params to be a list of lists, one for each parameter
    if not params:
        final_params = [[] for _ in sample_dict if _ != "soc"]
    else:
        final_params = list(zip(*params))

    return (tgt_an, c0_an, tgt_ca, c0_ca, cur, soc, *final_params)

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
        if res is None:
            continue
        if out_lists is None:
            out_lists = [list(x) for x in res]
        else:
            for lst,x in zip(out_lists, res):
                lst.extend(x)

    if out_lists is None:
        K = len(sample_dict)-1
        dummy = [[]]*(6+K)
        return tuple(dummy)

    print(f"--- Debug info from run_batch (family: {family}) ---")
    print(f"Batch size: {M}, Successful simulations: {len(out_lists[0]) if out_lists and out_lists[0] else '0'}")
    for i, lst in enumerate(out_lists):
        try:
            shapes = [np.shape(item) for item in lst]
            unique_shapes = set(shapes)
            if len(unique_shapes) > 1:
                print(f"  [!] Inconsistent shapes in list {i}:")
                shape_counts = {s: shapes.count(s) for s in unique_shapes}
                print(f"      Shape counts: {shape_counts}")
            else:
                print(f"  List {i}: OK, {len(lst)} items with shape {next(iter(unique_shapes)) if unique_shapes else 'N/A'}")
        except Exception as e:
            print(f"  [!] Error inspecting list {i}: {e}")

    return tuple(np.asarray(lst, dtype=object) for lst in out_lists)

# ------------------------------------------------------------
# SINGLE-THREADED FRONT-END
# ------------------------------------------------------------
def generate_data_singlethreaded(payload):
    print("--- Running in single-threaded mode for debugging ---")
    parts = []
    for i, p in enumerate(payload):
        print(f"--- Processing batch {i+1}/{len(payload)} ---")
        part = run_batch(*p)
        if part and any(len(x) > 0 for x in part):
            parts.append(part)

    if not parts:
        print("Warning: All simulation batches failed or produced no data.")
        if not payload: return ()
        K = len(payload[0][0]) - 1
        return tuple([] for _ in range(6 + K))

    print("--- Aggregating results from all batches ---")
    
    # Check for consistent number of lists returned from each batch
    num_fields = len(parts[0])
    if any(len(p) != num_fields for p in parts):
        print("[!] Error: Inconsistent number of fields returned by batches.")
        # Find which one is problematic
        for i, p in enumerate(parts):
            if len(p) != num_fields:
                print(f"Batch {i} returned {len(p)} fields, expected {num_fields}")
        return None

    # Each 'piece' is a tuple of lists of arrays from all batches
    all_pieces = list(zip(*parts))
    
    concatenated = tuple(np.concatenate(pieces) for pieces in all_pieces)
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
        print(f"Error: field count mismatch. Got {len(param_arrays)} param arrays, expected {len(param_keys)}")
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
        print(f"--- Starting family: {family} ---")
        payload = [({k:v[idx] for k,v in samples_dict.items()},
                    int(seeds[i]), family)
                   for i,idx in enumerate(indices)]

        data = generate_data_singlethreaded(payload)
        
        if data is None or len(data[5]) == 0:
            print(f"No data generated for family {family}, skipping save.")
            continue
            
        save_data(family, samples_dict, data, parameter_set="Prada2013") 