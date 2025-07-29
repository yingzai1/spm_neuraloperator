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
        "config", 
        type=str, 
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        help="Model type (overrides config file). Options: FNO, CAPE_FNO2, DON"
    )
    
    args = parser.parse_args()
    
    # Validate config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Configuration file '{config_path}' not found.")
        sys.exit(1)
    
    # Load config to determine model type if not specified
    if args.model:
        model_name = args.model
    else:
        import yaml
        with config_path.open("r") as f:
            config = yaml.safe_load(f)
        model_name = config["training"]["model_name"]
    
    print(f"Training {model_name} model with config: {config_path}")
    
    # Get trainer and start training
    try:
        trainer_class = get_trainer_class(model_name)
        trainer = trainer_class(str(config_path))
        trainer.train()
        print(f"\n✅ Training completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 