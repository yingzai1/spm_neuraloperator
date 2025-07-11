# run_one_estimate.py
import json, sys
from diffusion_estimation import estimate_diffusion_parameters
import numpy as np

if __name__ == "__main__":
    idx = int(sys.argv[1])                # e.g. "3"
    result = estimate_diffusion_parameters(idx)

    np.savez(
        f"param_est_files_dlu/run_{idx}.npz",
        diffs_anode        = result[0],
        diffs_cathode      = result[1],
        concentrations_anode   = result[2],
        concentrations_cathode = result[3],
        voltages           = result[4]
    )


    # Stdout must be *machine-readable* so we use JSON
    json.dump({
        "idx": idx,
        "rmse": result[5],
    }, sys.stdout)
    
