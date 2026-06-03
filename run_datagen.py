#!/usr/bin/env python3
import yaml
import numpy as np
from src.datagen import sampling, processing
import argparse
from datetime import datetime
import copy


def log(message):
    """Prints a message with a timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


def main(config_path: str):
    """Main script to drive the data generation workflow."""
    # 1. Load Configuration
    log(f"Loading configuration from {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    base_cfg = config["datagen"]

    for p_set in base_cfg["parameter_set"]:
        log(f"--- Starting data generation for parameter set: {p_set} ---")
        cfg = copy.deepcopy(base_cfg)
        cfg["parameter_set"] = p_set

        # Loop over the different sample sizes
        for samples_per_soc in cfg["sobol_samples_per_soc"]:
            log(f"--- Generating data for {samples_per_soc} samples per SoC ---")

            # 2. Generate Parameter Samples
            samples = sampling.generate_sobol_samples(samples_per_soc, cfg)

            # 3. Create and Run Simulation Tasks for Each Family
            for family in cfg["current_families"]:
                log(f"--- Preparing tasks for family: {family} ---")

                tasks = processing.create_tasks(samples, family, cfg)

                log(f"--- Running {len(tasks)} simulations for family: {family} ---")
                final_data = processing.run_in_parallel(tasks, cfg["n_workers"])

                # 4. Save Results
                if final_data and len(next(iter(final_data.values()))) > 0:
                    # Get the number of successful samples from the final data
                    num_successful_samples = len(final_data["soc"])

                    # Construct the new filepath with the sample count
                    filepath = (
                        f"{cfg['output_dir']}/"
                        f"{p_set}_{family}_{num_successful_samples}.npz"
                    )

                    processing.save_data(filepath, final_data, cfg)
                else:
                    log(f"No successful simulations for family {family}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate datasets from simulation configs."
    )
    parser.add_argument(
        "--config",
        type=str,
        nargs="+",
        required=True,
        help="Path to the configuration file(s). Can be multiple.",
    )
    args = parser.parse_args()

    for f in args.config:
        main(f)
