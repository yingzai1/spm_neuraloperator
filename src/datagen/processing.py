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
    # TODO: Should be moved to config if it varies
    # TODO: rename C to current
    C_nominal = 2.3

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

def save_data(filepath, data, config):
    """
    Saves the final aggregated data to a .npz file, filtering based on the config.
    """
    p = Path(filepath)
    p.parent.mkdir(parents=True, exist_ok=True)

    param_key_map = config.get('param_key_map', {})
    
    # These are the keys that will always be saved
    essential_keys = {"cn_anode", "c0_anode", "cn_cathode", "c0_cathode", "current", "soc"}
    
    # These are the desired parameter keys from the map
    mapped_param_keys = set(param_key_map.values())
    
    # Filter the data dictionary
    final_data_to_save = {
        key: value for key, value in data.items()
        if key in essential_keys or key in mapped_param_keys
    }

    np.savez(p, **final_data_to_save)
    print(f"✅ Saved {len(final_data_to_save['soc'])} results to {p}")
