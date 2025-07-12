#!/usr/bin/env python3
import yaml
import numpy as np
from src.datagen import sampling, processing


def main():

    """Main script to drive the data generation workflow."""
    # 1. Load Configuration
    with open("configs/datagen_config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    cfg = config['datagen']

    # Loop over the different sample sizes
    for samples_per_soc in cfg["sobol_samples_per_soc"]:
        print(f"--- Generating data for {samples_per_soc} samples per SoC ---")

        # 2. Generate Parameter Samples
        samples = sampling.generate_sobol_samples(samples_per_soc, cfg)

        # 3. Create and Run Simulation Tasks for Each Family
        for family in cfg["current_families"]:
            print(f"--- Preparing tasks for family: {family} ---")
            
            tasks = processing.create_tasks(samples, family, cfg)
            
            print(f"--- Running {len(tasks)} simulations for family: {family} ---")
            final_data = processing.run_in_parallel(tasks, cfg["n_workers"])

            # 4. Save Results
            if final_data and len(next(iter(final_data.values()))) > 0:
                # Get the number of successful samples from the final data
                num_successful_samples = len(final_data['soc'])
                
                # Construct the new filepath with the sample count
                filepath = (
                    f"{cfg['output_dir']}/"
                    f"{cfg['parameter_set']}_{family}_{num_successful_samples}.npz"
                )
                
                processing.save_data(filepath, final_data, cfg)
            else:
                print(f"No successful simulations for family {family}.")

if __name__ == "__main__":
    main()
