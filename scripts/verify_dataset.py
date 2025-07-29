import argparse
import numpy as np
import yaml
import sys
from pathlib import Path
from typing import Dict, Any, Tuple
# Ensure repository root is on PYTHONPATH so that 'src' can be imported even when executed from within scripts/ directory.
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.datagen.simulation import run_single_simulation


# --------------------------------------------------------------------------------------
# Utility helpers
# --------------------------------------------------------------------------------------

def load_yaml_config(path: Path) -> Dict[str, Any]:
    """Loads a YAML configuration file and returns the *datagen* section as a dict."""
    with path.open("r") as f:
        full_cfg = yaml.safe_load(f)

    if "datagen" not in full_cfg:
        raise ValueError(f"Config file '{path}' does not contain a 'datagen' section.")

    return full_cfg["datagen"]


def infer_param_set_from_filename(file_path: Path) -> str:
    """Infers the parameter-set name from a dataset filename like 'Prada2013_CC_100.npz'."""
    return file_path.stem.split("_")[0]


def invert_mapping(mapping: Dict[str, str]) -> Dict[str, str]:
    """Returns a dictionary with keys/values swapped."""
    return {v: k for k, v in mapping.items()}


def build_params_for_sample(index: int, data: Dict[str, np.ndarray], param_map_inv: Dict[str, str]) -> Dict[str, Any]:
    """Extracts the sampled parameters for *index* and converts them to short names."""
    params = {}
    for long_name, short_name in param_map_inv.items():
        if long_name not in data:
            continue  # Parameter not present in this dataset
        params[short_name] = data[long_name][index].item() if data[long_name].ndim == 1 else data[long_name][index]
    return params


def compare_arrays(a: np.ndarray, b: np.ndarray, atol: float = 1e-6) -> Tuple[bool, float]:
    """Returns (is_close, max_abs_err)."""
    if a.shape != b.shape:
        return False, np.inf
    diff = np.abs(a - b)
    return np.all(diff <= atol), diff.max()


# --------------------------------------------------------------------------------------
# Main routine
# --------------------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a generated dataset by re-simulating random samples.")
    parser.add_argument("dataset", type=str, help="Path to the .npz dataset to verify.")
    parser.add_argument("--config", type=str, required=True, help="YAML config that was used to generate the dataset.")
    parser.add_argument("-n", type=int, default=100, help="Number of random samples to verify (default: 100).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling.")

    args = parser.parse_args()

    ds_path = Path(args.dataset)
    cfg_path = Path(args.config)

    # ------------------------------------------------------------------
    # 1. Load configuration and prepare it for a single simulation run
    # ------------------------------------------------------------------
    base_cfg = load_yaml_config(cfg_path)

    # If the config contains multiple parameter sets, attempt to infer which one
    if isinstance(base_cfg.get("parameter_set"), list):
        inferred_set = infer_param_set_from_filename(ds_path)
        if inferred_set not in base_cfg["parameter_set"]:
            raise ValueError(
                "Unable to infer parameter_set. Provide a config with a single parameter_set or adjust filename pattern."
            )
        base_cfg["parameter_set"] = inferred_set

    # ------------------------------------------------------------------
    # 2. Load dataset
    # ------------------------------------------------------------------
    rng = np.random.default_rng(args.seed)

    with np.load(ds_path) as data:
        num_samples = data["soc"].shape[0]
        n_verify = min(args.n, num_samples)
        sample_indices = rng.choice(num_samples, size=n_verify, replace=False)

        print(f"[INFO] Loaded dataset with {num_samples} samples. Verifying {n_verify} random samples …")

        # Prepare inverted parameter map (Dan -> D_n etc.)
        param_key_map = base_cfg.get("param_key_map", {})

        mismatch_count = 0

        for idx in sample_indices:
            params = build_params_for_sample(idx, data, param_key_map)
            print(params)
            current = data["current"][idx]
            soc = float(data["soc"][idx])

            sim_out = run_single_simulation((params, current, soc, base_cfg))
            if sim_out is None:
                print(f"  [WARNING] Simulation failed for sample {idx}. Skipping.")
                continue

            # Compare key outputs
            keys_to_check = ["cn_anode", "c0_anode", "cn_cathode", "c0_cathode"]
            for key in keys_to_check:
                ok, max_err = compare_arrays(sim_out[key], data[key][idx], atol=1e-4)
                if not ok:
                    print(f"  [MISMATCH] Sample {idx} – '{key}' differs (max abs err={max_err:.2e}).")
                    mismatch_count += 1
                    break  # One mismatch is enough for this sample
        if mismatch_count == 0:
            print("\n✅ All checked samples match the simulation output within tolerance.")
        else:
            print(
                f"\n❌ Detected {mismatch_count} mismatching sample(s) out of {n_verify}. See logs above for details."
            )


if __name__ == "__main__":
    main() 
