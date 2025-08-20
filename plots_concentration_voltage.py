#!/usr/bin/env python3
"""
Concentration and Voltage Metrics Comparison Plotting Script

This script runs error analysis for multiple neural network models and generates
publication-quality comparison plots for concentration and voltage prediction errors.

Usage:
    python plots_concentration_voltage.py
    python plots_concentration_voltage.py --output_dir plots/custom_metrics
    python plots_concentration_voltage.py --configs configs/errors/FNO.yaml configs/errors/CAPE_FNO2.yaml
"""

import argparse
import yaml
import sys
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

from src.analysis import ErrorAnalyzer
from src.plotter import MetricsComparisonOrchestrator


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_model_analysis(config_path: str) -> Dict[str, Any]:
    """
    Run error analysis for a single model architecture across all current profiles.
    
    Args:
        config_path: Path to error analysis configuration
        
    Returns:
        Dictionary containing analysis results for all current profiles
    """
    print(f"🔧 Loading configuration from {config_path}")
    config = load_config(config_path)
    
    model_architecture = config["model"]["architecture"]
    current_profiles = ['CC', 'Triangle', 'PLS', 'GRF']
    results = {}
    
    for profile in current_profiles:
        try:
            print(f"  📋 Analyzing {profile} profile...")
            
            # Create a copy of config for this profile
            profile_config = config.copy()
            profile_config['data'] = config['data'].copy()
            
            # Update the family for this current profile
            profile_config['data']['family'] = profile
            
            # Initialize analyzer with profile-specific config
            analyzer = ErrorAnalyzer(profile_config)
            
            # Run analysis for this profile
            profile_results = analyzer.analyze_model(
                model_architecture=model_architecture,
                anode_model_path=None,  # Use latest model
                cathode_model_path=None  # Ignored in unified model structure
            )
            
            results[profile] = profile_results
            
        except Exception as e:
            print(f"    ⚠️  Warning: Could not analyze {profile} profile: {str(e)}")
            continue
    
    return results


def analyze_multiple_models(config_paths: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Run error analysis for multiple model architectures.
    
    Args:
        config_paths: List of configuration file paths
        
    Returns:
        Dictionary with model names as keys and analysis results as values
    """
    results = {}
    
    for config_path in config_paths:
        print(f"\n📊 Analyzing model from {config_path}")
        try:
            config = load_config(config_path)
            model_name = config["model"]["architecture"]
            
            result = run_model_analysis(config_path)
            results[model_name] = result
            
            print(f"✅ Analysis complete for {model_name}")
            
        except Exception as e:
            print(f"❌ Error analyzing {config_path}: {str(e)}")
            continue
    
    return results


def get_default_configs() -> List[str]:
    """Get list of default configuration files."""
    config_dir = Path("configs/errors")
    
    # Look for available configs
    configs = []
    for config_file in ["FNO.yaml", "CAPE_FNO2.yaml", "DON.yaml"]:
        config_path = config_dir / config_file
        if config_path.exists():
            configs.append(str(config_path))
    
    return configs


def validate_configs(config_paths: List[str]) -> List[str]:
    """Validate that configuration files exist."""
    valid_configs = []
    
    for config_path in config_paths:
        if Path(config_path).exists():
            valid_configs.append(config_path)
        else:
            print(f"⚠️  Warning: Configuration file {config_path} does not exist")
    
    return valid_configs


def main():
    """Main entry point for metrics comparison plotting script."""
    parser = argparse.ArgumentParser(
        description="Create concentration and voltage metrics comparison plots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default configs (FNO, CAPE_FNO2, DON)
  python plots_concentration_voltage.py
  
  # Specify custom configs
  python plots_concentration_voltage.py --configs configs/errors/FNO.yaml configs/errors/CAPE_FNO2.yaml
  
  # Custom output directory
  python plots_concentration_voltage.py --output_dir plots/paper_figures
        """
    )
    
    parser.add_argument(
        "--configs",
        nargs="+",
        help="Paths to error analysis configuration files (default: all configs in configs/errors/)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="plots/metrics_comparison",
        help="Output directory for plots (default: plots/metrics_comparison)"
    )
    
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plots interactively instead of just saving"
    )
    
    args = parser.parse_args()
    
    # Determine which configs to use
    if args.configs:
        config_paths = args.configs
    else:
        config_paths = get_default_configs()
        if not config_paths:
            print("❌ Error: No configuration files found in configs/errors/")
            print("Available configs should be: FNO.yaml, CAPE_FNO2.yaml, DON.yaml")
            sys.exit(1)
    
    # Validate configs exist
    config_paths = validate_configs(config_paths)
    if not config_paths:
        print("❌ Error: No valid configuration files found")
        sys.exit(1)
    
    if len(config_paths) < 2:
        print("❌ Error: At least 2 model configurations are required for comparison")
        sys.exit(1)
    
    print(f"📋 Using configurations: {config_paths}")
    
    try:
        # Run analysis for all models
        print("🔍 Running error analysis for all models...")
        model_results = analyze_multiple_models(config_paths)
        
        if len(model_results) < 2:
            print("❌ Error: Need at least 2 successful model analyses for comparison")
            sys.exit(1)
        
        # Create comparison plots
        print(f"\n📈 Creating metrics comparison plots...")
        orchestrator = MetricsComparisonOrchestrator(output_dir=args.output_dir)
        orchestrator.create_all_comparison_plots(
            model_results, 
            save=True, 
            show_plot=args.show
        )
        
        # Print summary
        print(f"\n📊 Summary of analyzed models:")
        for model_name, model_data in model_results.items():
            print(f"  {model_name}:")
            for profile, profile_data in model_data.items():
                if profile_data:  # Check if data exists for this profile
                    conc_errors = profile_data["concentration_errors_normalized"]["combined"]
                    voltage_errors = profile_data["voltage_errors"]
                    
                    print(f"    {profile}: Conc nL2: {conc_errors['rel_l2'].mean() * 100:.2f}%, "
                          f"Volt RMSE: {np.sqrt(voltage_errors['mse'].mean()) * 1000:.2f} mV")
        
        print(f"\n✅ Metrics comparison plots created successfully!")
        print(f"📁 Plots saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 
