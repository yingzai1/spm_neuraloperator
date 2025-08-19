#!/usr/bin/env python3
"""
Main training script for neural network models.
Supports FNO, CAPE-FNO2, and DeepONet models via YAML configuration files.
"""

import argparse
import sys
from pathlib import Path

# Add repo root to path for imports
sys.path.append(str(Path(__file__).resolve().parent))

from src.training import FNOTrainer, CAPEFNO2Trainer, DONTrainer


def get_trainer_class(model_name: str):
    """Get the appropriate trainer class for the model."""
    trainers = {
        "FNO": FNOTrainer,
        "CAPE_FNO2": CAPEFNO2Trainer,
        "DON": DONTrainer
    }
    
    if model_name not in trainers:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(trainers.keys())}")
    
    return trainers[model_name]


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train neural network models for battery simulation")
    parser.add_argument(
        "configs",
        nargs="+",
        type=str,
        help="Path(s) to YAML configuration file(s)"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        help="Model type (overrides config file). Options: FNO, CAPE_FNO2, DON"
    )
    
    args = parser.parse_args()
    
    for config_path_str in args.configs:
        # Validate config file exists
        config_path = Path(config_path_str)
        if not config_path.exists():
            print(f"Error: Configuration file '{config_path}' not found.")
            continue  # Skip to the next config file
        
        # Load config to determine model type if not specified
        if args.model:
            model_name = args.model
        else:
            import yaml
            with config_path.open("r") as f:
                config = yaml.safe_load(f)
            model_name = config["training"]["model_name"]
        
        print(f"\n--- Starting new training session ---")
        print(f"Training {model_name} model with config: {config_path}")
        
        # Get trainer and start training
        try:
            trainer_class = get_trainer_class(model_name)
            trainer = trainer_class(str(config_path))
            trainer.train()
            print(f"\n✅ Training for {config_path} completed successfully!")
            
        except Exception as e:
            print(f"\n❌ Training for {config_path} failed with error: {e}")
            import traceback
            traceback.print_exc()
            # Continue to next config even if one fails


if __name__ == "__main__":
    main() 