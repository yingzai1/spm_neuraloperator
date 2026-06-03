import pathlib
import tqdm

import subprocess, json, textwrap
import numpy as np

def run_idx(idx: int):
    try:
        completed = subprocess.run(
            ["taskset", "-c", "0", "python", "run_one_estimate_dlu.py", str(idx)],
            # ["python", "run_one_estimate_dlu.py", str(idx)],
            text=True,
            capture_output=True,   # keep both stdout **and** stderr
            check=True,
        )
        return json.loads(completed.stdout)

    except subprocess.CalledProcessError as e:
        print(f"\n--- child process stderr for idx={idx} ---\n")
        print(textwrap.indent(e.stderr, prefix="│ "))
        print("\n-----------------------------------------\n")
        raise                     # re-raise so you still see the notebook error


# num_iterations = 1296

# tqdmiter = tqdm.trange(
#     num_iterations,
#     desc="Running simulations",
#     unit="iteration",)

start_idx = 229
num_iterations = 1296

tqdmiter = tqdm.trange(
    start_idx,                      # Start from 229
    num_iterations,                # End at 1296 (exclusive)
    desc="Running simulations",
    unit="iteration"
)

for i in tqdmiter:
    out = run_idx(i)
    print(f"Simulation {i+1}/{num_iterations} completed RMSE: {out['rmse']}")
    desc_str = f"Simulation {i+1}/{num_iterations}: "
    tqdmiter.set_description(desc_str)

def load_the_data(idx):
    # Load the data from the results
    data = np.load(f"param_est_files_dlu/run_{idx}.npz", allow_pickle=True)
    diffs_anode = data["diffs_anode"].item()
    diffs_cathode = data["diffs_cathode"].item()
    concentrations_anode = data["concentrations_anode"].item()
    concentrations_cathode = data["concentrations_cathode"].item()
    voltages = data["voltages"].item()

    return diffs_anode, diffs_cathode, concentrations_anode, concentrations_cathode, voltages

diffs_anode = {
    "log10_Dan_cape_fno": [],
    "log10_Dan_pybamm": [],
    "log10_Dan_true": []
}

diffs_cathode = {
    "log10_Dca_cape_fno": [],
    "log10_Dca_pybamm": [],
    "log10_Dca_true": []
}

concentrations_anode = {
    "c_an_pybamm_from_cape_fno": [],
    "c_an_data": [],
    "c_an_pybamm_from_pybamm": [],
    "c_an_cape_fno_from_cape_fno": [],
    "c_an_cape_fno_from_pybamm": [],
}

concentrations_cathode = {
    "c_ca_pybamm_from_cape_fno": [],
    "c_ca_data": [],
    "c_ca_pybamm_from_pybamm": [],
    "c_ca_cape_fno_from_cape_fno": [],
    "c_ca_cape_fno_from_pybamm": [],
}

voltages = {
    "V_best_cape_fno_from_cape_fno": [],
    "V_best_cape_fno_from_pybamm": [],
    "V_best_pybamm_from_cape_fno": [],
    "V_best_pybamm_from_pybamm": [],
    "V_data": [],
}


def single_data_append(diffs_anode_idx, diffs_cathode_idx, concentrations_anode_idx, concentrations_cathode_idx, voltages_idx):
    diffs_anode["log10_Dan_cape_fno"].append(diffs_anode_idx["log10_Dan_cape_fno"])
    diffs_anode["log10_Dan_pybamm"].append(diffs_anode_idx["log10_Dan_pybamm"])
    diffs_anode["log10_Dan_true"].append(diffs_anode_idx["log10_Dan_true"])

    diffs_cathode["log10_Dca_cape_fno"].append(diffs_cathode_idx["log10_Dca_cape_fno"])
    diffs_cathode["log10_Dca_pybamm"].append(diffs_cathode_idx["log10_Dca_pybamm"])
    diffs_cathode["log10_Dca_true"].append(diffs_cathode_idx["log10_Dca_true"])

    concentrations_anode["c_an_pybamm_from_cape_fno"].append(concentrations_anode_idx["c_an_pybamm_from_cape_fno"])
    concentrations_anode["c_an_data"].append(concentrations_anode_idx["c_an_data"])
    concentrations_anode["c_an_pybamm_from_pybamm"].append(concentrations_anode_idx["c_an_pybamm_from_pybamm"])
    concentrations_anode["c_an_cape_fno_from_cape_fno"].append(concentrations_anode_idx["c_an_cape_fno_from_cape_fno"])
    concentrations_anode["c_an_cape_fno_from_pybamm"].append(concentrations_anode_idx["c_an_cape_fno_from_pybamm"])

    concentrations_cathode["c_ca_pybamm_from_cape_fno"].append(concentrations_cathode_idx["c_ca_pybamm_from_cape_fno"])
    concentrations_cathode["c_ca_data"].append(concentrations_cathode_idx["c_ca_data"])
    concentrations_cathode["c_ca_pybamm_from_pybamm"].append(concentrations_cathode_idx["c_ca_pybamm_from_pybamm"])
    concentrations_cathode["c_ca_cape_fno_from_cape_fno"].append(concentrations_cathode_idx["c_ca_cape_fno_from_cape_fno"])
    concentrations_cathode["c_ca_cape_fno_from_pybamm"].append(concentrations_cathode_idx["c_ca_cape_fno_from_pybamm"])

    voltages["V_best_cape_fno_from_cape_fno"].append(voltages_idx["V_best_cape_fno_from_cape_fno"])
    voltages["V_best_cape_fno_from_pybamm"].append(voltages_idx["V_best_cape_fno_from_pybamm"])
    voltages["V_best_pybamm_from_cape_fno"].append(voltages_idx["V_best_pybamm_from_cape_fno"])
    voltages["V_best_pybamm_from_pybamm"].append(voltages_idx["V_best_pybamm_from_pybamm"])
    voltages["V_data"].append(voltages_idx["V_data"])


def save_as_npz(out_path: str | pathlib.Path,
                diffs_anode: dict[str, list],
                diffs_cathode: dict[str, list],
                concentrations_anode: dict[str, list],
                concentrations_cathode: dict[str, list],
                voltages: dict[str, list]) -> None:
    """
    Flatten the five dictionaries into one `.npz`.
    Each key becomes `<group>/<subkey>` so there are no collisions.
    Lists are turned into object-dtype arrays so differing shapes are OK.
    """
    payload = {}

    def _add_group(prefix: str, group: dict[str, list]):
        for k, v in group.items():
            payload[f"{prefix}/{k}"] = np.asarray(v, dtype=object)

    _add_group("diffs_anode",           diffs_anode)
    _add_group("diffs_cathode",         diffs_cathode)
    _add_group("concentrations_anode",  concentrations_anode)
    _add_group("concentrations_cathode",concentrations_cathode)
    _add_group("voltages",              voltages)

    np.savez_compressed(out_path, **payload)
    print(f"✓ wrote {out_path}")



for i in range(num_iterations):
    
    try:
        diffs_anode_idx, diffs_cathode_idx, concentrations_anode_idx, concentrations_cathode_idx, voltages_idx = load_the_data(i)
    except:
        print(f"File for iteration {i} is corrupted")
        continue

    single_data_append(
        diffs_anode_idx, diffs_cathode_idx, concentrations_anode_idx, concentrations_cathode_idx, voltages_idx)
    

len(diffs_anode["log10_Dan_cape_fno"])


save_as_npz("src/inference/param_est_results.npz",
            diffs_anode, diffs_cathode,
            concentrations_anode, concentrations_cathode,
            voltages)


def load_the_dataset(filename):
    data = np.load(filename, allow_pickle=True)

    # Rebuild the nested dict you started with
    diffs_anode_loaded = {k.split("/",1)[1]: data[k].tolist()
                        for k in data if k.startswith("diffs_anode/")}
    diffs_cathode_loaded = {k.split("/",1)[1]: data[k].tolist()
                            for k in data if k.startswith("diffs_cathode/")}
    concentrations_anode_loaded = {k.split("/",1)[1]: data[k].tolist()
                                for k in data if k.startswith("concentrations_anode/")}
    concentrations_cathode_loaded = {k.split("/",1)[1]: data[k].tolist()
                                    for k in data if k.startswith("concentrations_cathode/")}
    voltages_loaded = {k.split("/",1)[1]: data[k].tolist()
                    for k in data if k.startswith("voltages/")}
    return diffs_anode_loaded, diffs_cathode_loaded, concentrations_anode_loaded, concentrations_cathode_loaded, voltages_loaded

diffs_anode_loaded, diffs_cathode_loaded, concentrations_anode_loaded, concentrations_cathode_loaded, voltages_loaded = load_the_dataset("src/inference/param_est_results.npz")

diffs_cathode_loaded.keys()

np.array(diffs_cathode_loaded["log10_Dca_cape_fno"]).min(), np.array(diffs_cathode_loaded["log10_Dca_cape_fno"]).max()


