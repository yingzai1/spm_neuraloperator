#!/bin/bash

# Create directory structure
echo "Creating directory structure..."
mkdir -p configs src/datagen && \

# --- Create run_datagen.py ---
echo "Creating run_datagen.py..."
cat << 'EOF' > run_datagen.py
#!/usr/bin/env python3
import yaml
import numpy as np
from src.datagen import sampling, processing

def main():
    """Main script to drive the data generation workflow."""
    # 1. Load Configuration
    with open("configs/datagen_config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    # Use the 'datagen' section of the config
    cfg = config['datagen']

    # 2. Generate Parameter Samples
    samples = sampling.generate_sobol_samples(cfg)

    # 3. Create and Run Simulation Tasks for Each Family
    for family in cfg["current_families"]:
        print(f"--- Preparing tasks for family: {family} ---")

        tasks = processing.create_tasks(samples, family, cfg)

        print(f"--- Running {len(tasks)} simulations for family: {family} ---")
        final_data = processing.run_in_parallel(tasks, cfg["n_workers"])

        # 4. Save Results
        if final_data and len(next(iter(final_data.values()))) > 0:
            filepath = f"{cfg['output_dir']}/{cfg['parameter_set']}_{family}.npz"
            processing.save_data(filepath, final_data)
        else:
            print(f"No successful simulations for family {family}.")

if __name__ == "__main__":
    main()
EOF
chmod +x run_datagen.py && \

# --- Create configs/datagen_config.yaml ---
echo "Creating configs/datagen_config.yaml..."
cat << 'EOF' > configs/datagen_config.yaml
datagen:
  # --- General Experiment Setup ---
  output_dir: "data/experiment_01"
  parameter_set: "Prada2013"
  n_workers: 4 # Use 1 for single-threaded debugging

  # --- Parameter Sampling ---
  sobol_samples_per_soc: 13
  soc_levels: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

  # --- Current Profiles ---
  current_families: ["CC", "PLS"]

  # --- PyBaMM Simulation Settings ---
  pybamm_settings:
    t_max_s: 3600
    t_num_points: 75
    model: "SPM"
    solver: "CasadiSolver"

  # --- Model Parameter Bounds for Sobol Sampling ---
  parameter_bounds:
    D_n: ["Negative particle diffusivity [m2.s-1]", [1.0e-18, 1.0e-14]]
    D_p: ["Positive particle diffusivity [m2.s-1]", [1.0e-18, 1.0e-14]]
    R_n: ["Negative particle radius [m]", [4.0e-6, 1.5e-5]]
    R_p: ["Positive particle radius [m]", [1.0e-8, 1.5e-5]]
    eps_n: ["Negative electrode active material volume fraction", [0.3, 0.8]]
    eps_p: ["Positive electrode active material volume fraction", [0.3, 0.8]]
    L_n: ["Negative electrode thickness [m]", [2.0e-5, 10.0e-5]] # <-- CORRECTED
    L_p: ["Positive electrode thickness [m]", [2.0e-5, 10.0e-5]] # <-- CORRECTED
    A: ["Electrode area [m2]", [5.0e-3, 5.0e-1]]
EOF
# --- Create src/datagen/__init__.py ---
echo "Creating src/datagen/__init__.py..."
touch src/datagen/__init__.py && \

# --- Create src/datagen/sampling.py ---
echo "Creating src/datagen/sampling.py..."
cat << 'EOF' > src/datagen/sampling.py
import numpy as np
from scipy.stats import qmc

def generate_sobol_samples(config):
    """Generates Sobol samples based on the provided configuration."""
    param_info = config["parameter_bounds"]
    soc_levels = np.asarray(config["soc_levels"])
    n_total = len(soc_levels) * config["sobol_samples_per_soc"]

    param_keys = list(param_info.keys())
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
EOF
# --- Create src/datagen/currents.py ---
echo "Creating src/datagen/currents.py..."
cat << 'EOF' > src/datagen/currents.py
import numpy as np

def _constant_current(C, t):
    """Generates a constant current."""
    return np.full_like(t, C)

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

def generate_current_profile(family, C, t_max, rng, num_points=75):
    """Factory function to generate a single current profile."""
    t = np.linspace(0, t_max, num_points)
    if family == "CC":
        val = rng.uniform(-1, 1) * C
        return _constant_current(val, t)
    elif family == "PLS":
        return _pls_current(C, t_max, rng, t)
    else:
        raise ValueError(f"Unknown current family: {family}")
EOF
# --- Create src/datagen/simulation.py ---
echo "Creating src/datagen/simulation.py..."
cat << 'EOF' > src/datagen/simulation.py
import pybamm
import numpy as np

def run_single_simulation(task_payload):
    """
    Runs a single PyBaMM simulation. Designed to be called by a worker process.
    """
    params, current, soc, config = task_payload

    model_options = config['pybamm_settings'].get('model', 'SPM')
    solver_name = config['pybamm_settings'].get('solver', 'CasadiSolver')

    if model_options == 'SPM':
        model = pybamm.lithium_ion.SPM()
    else:
        raise NotImplementedError(f"Model {model_options} not implemented.")

    model.events = []
    param_set = pybamm.ParameterValues(config["parameter_set"])
    short_to_full = {name: info[0] for name, info in config["parameter_bounds"].items()}

    for short_name, value in params.items():
        if short_name == "A":
            param_set["Electrode height [m]"] = param_set["Electrode width [m]"] = np.sqrt(value)
        else:
            param_set[short_to_full[short_name]] = value

    t_eval = np.linspace(0, config["pybamm_settings"]["t_max_s"], config["pybamm_settings"]["t_num_points"])
    param_set["Current function [A]"] = pybamm.Interpolant(t_eval, -current, pybamm.t)

    solver = getattr(pybamm, solver_name)()
    sim = pybamm.Simulation(model, parameter_values=param_set, solver=solver)

    try:
        sol = sim.solve(initial_soc=float(soc), t_eval=t_eval)

        return {
            "tgt_anode": sol["Negative particle concentration"].entries[:, 0, :],
            "c0_anode": sol["Negative particle concentration"].entries[:, 0, 0],
            "tgt_cathode": sol["Positive particle concentration"].entries[:, 0, :],
            "c0_cathode": sol["Positive particle concentration"].entries[:, 0, 0],
            "current": current,
            "soc": soc,
            **params
        }
    except (pybamm.SolverError, pybamm.ModelError):
        return None
EOF
# --- Create src/datagen/processing.py ---
echo "Creating src/datagen/processing.py..."
cat << 'EOF' > src/datagen/processing.py
import numpy as np
import multiprocessing as mp
from pathlib import Path
from collections import defaultdict
from .simulation import run_single_simulation
from .currents import generate_current_profile

def create_tasks(samples, family, config):
    """Prepares the list of tasks for the multiprocessing pool."""
    tasks = []
    rng = np.random.default_rng()
    t_max = config["pybamm_settings"]["t_max_s"]
    C_nominal = 1.0 # Should be moved to config if it varies

    for i in range(len(samples["soc"])):
        params_for_run = {key: val[i] for key, val in samples.items()}
        soc_for_run = params_for_run.pop("soc")
        current_for_run = generate_current_profile(family, C_nominal, t_max, rng)
        tasks.append((params_for_run, current_for_run, soc_for_run, config))
    return tasks

def run_in_parallel(tasks, n_workers):
    """Manages the multiprocessing pool to run all simulation tasks."""
    # Use 'fork' to ensure memory is shared efficiently where possible (macOS/Linux)
    # This might need adjustment for Windows, which defaults to 'spawn'
    ctx = mp.get_context('fork')

    if n_workers > 1:
        with ctx.Pool(n_workers) as pool:
            results = pool.map(run_single_simulation, tasks)
    else:
        results = [run_single_simulation(task) for task in tasks]

    successful_results = [r for r in results if r is not None]
    if not successful_results:
        return None

    agg_data = defaultdict(list)
    for res_dict in successful_results:
        for key, value in res_dict.items():
            agg_data[key].append(value)

    return {key: np.array(value) for key, value in agg_data.items()}

def save_data(filepath, data):
    """Saves the final aggregated data to a .npz file."""
    p = Path(filepath)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, **data)
    print(f"✅ Saved {len(next(iter(data.values())))} results to {p}")
EOF

echo ""
echo "✅ Project structure created successfully with corrections!"
echo "You can now run the data generation with: ./run_datagen.py"
