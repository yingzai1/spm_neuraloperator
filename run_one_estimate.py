# run_one_estimate.py
import json, sys
from diffusion_estimation import estimate_diffusion_parameters

if __name__ == "__main__":
    idx = int(sys.argv[1])                # e.g. "3"
    result = estimate_diffusion_parameters(idx)

    # Stdout must be *machine-readable* so we use JSON
    json.dump({
        "idx": idx,
        "best_log10_Dan": result[0],
        "best_log10_Dca": result[1],
        "rmse": result[2],
    }, sys.stdout)
