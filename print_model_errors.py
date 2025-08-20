#!/usr/bin/env python3
"""
Model Error Analysis Script

This script runs comprehensive error analysis for trained neural network models
and prints detailed error metrics to the console.

Usage:
    python print_model_errors.py configs/errors/FNO.yaml
    python print_model_errors.py configs/errors/CAPE_FNO2.yaml --model_anode models/FNO/anode_model.msgpack
    python print_model_errors.py configs/errors/DON.yaml --model_cathode models/DON/cathode_model.msgpack
    python print_model_errors.py configs/errors/FNO.yaml configs/errors/CAPE_FNO2.yaml --compare
"""

import argparse
import yaml
import sys
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

from src.analysis import ErrorAnalyzer


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_single_model_analysis(config_path: str, 
                             anode_model_path: Optional[str] = None,
                             cathode_model_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Run error analysis for a single model architecture.
    
    Args:
        config_path: Path to error analysis configuration
        anode_model_path: Optional path to anode model (uses latest if None)
        cathode_model_path: Optional path to cathode model (uses latest if None)
        
    Returns:
        Dictionary containing analysis results
    """
    print(f"🔧 Loading configuration from {config_path}")
    config = load_config(config_path)
    
    # Initialize analyzer
    analyzer = ErrorAnalyzer(config)
    
    # Run analysis
    model_architecture = config["model"]["architecture"]
    results = analyzer.analyze_model(
        model_architecture=model_architecture,
        anode_model_path=anode_model_path,
        cathode_model_path=cathode_model_path
    )
    
    return results


def print_error_summary(results: Dict[str, Any]) -> None:
    """Print a summary of error metrics to console."""
    model_name = results["model_architecture"]
    
    # Check if this is a multi-profile result
    if "profile_results" in results and "families" in results:
        # Multi-profile analysis - print each profile separately
        families = results["families"]
        profile_results = results["profile_results"]
        
        print(f"\n📊 Error Summary for {model_name} - Profile-by-Profile Analysis:")
        print("=" * 70)
        
        for profile in families:
            if profile in profile_results:
                result = profile_results[profile]
                print(f"\n🔬 {profile} Current Profile:")
                print("-" * 50)
                
                # Concentration errors (normalized)
                conc_errors = result["concentration_errors_normalized"]["combined"]
                print("\n🧪 Concentration Errors (Normalized):")
                print(f"  MAE:       {conc_errors['mae'].mean():.6f}")
                print(f"  RMSE:      {np.sqrt(conc_errors['mse'].mean()):.6f}")
                print(f"  Rel L2:    {conc_errors['rel_l2'].mean() * 100:.3f}%")
                print(f"  Rel L∞:    {conc_errors['rel_linf'].mean() * 100:.3f}%")
                
                # Voltage errors
                voltage_errors = result["voltage_errors"]
                print("\n⚡ Voltage Errors:")
                print(f"  MAE:       {voltage_errors['mae'].mean() * 1000:.3f} mV")
                print(f"  RMSE:      {np.sqrt(voltage_errors['mse'].mean()) * 1000:.3f} mV")
                print(f"  Rel L2:    {voltage_errors['rel_l2'].mean() * 100:.3f}%")
                print(f"  Rel L∞:    {voltage_errors['rel_linf'].mean() * 100:.3f}%")
                print("-" * 50)
            else:
                print(f"\n⚠️  {profile} Current Profile: Analysis failed")
        
        # Also print combined summary
        print(f"\n📊 Combined Summary for {model_name} (All Profiles):")
        print("=" * 50)
        
        # Combined concentration errors (normalized)
        conc_errors = results["concentration_errors_normalized"]["combined"]
        print("\n🧪 Concentration Errors (Normalized):")
        print(f"  MAE:       {conc_errors['mae'].mean():.6f}")
        print(f"  RMSE:      {np.sqrt(conc_errors['mse'].mean()):.6f}")
        print(f"  Rel L2:    {conc_errors['rel_l2'].mean() * 100:.3f}%")
        print(f"  Rel L∞:    {conc_errors['rel_linf'].mean() * 100:.3f}%")
        
        # Combined voltage errors
        voltage_errors = results["voltage_errors"]
        print("\n⚡ Voltage Errors:")
        print(f"  MAE:       {voltage_errors['mae'].mean() * 1000:.3f} mV")
        print(f"  RMSE:      {np.sqrt(voltage_errors['mse'].mean()) * 1000:.3f} mV")
        print(f"  Rel L2:    {voltage_errors['rel_l2'].mean() * 100:.3f}%")
        print(f"  Rel L∞:    {voltage_errors['rel_linf'].mean() * 100:.3f}%")
        
        print("=" * 70)
        
    else:
        # Single profile analysis - use original format
        print(f"\n📊 Error Summary for {model_name}:")
        print("=" * 50)
        
        # Concentration errors (normalized)
        conc_errors = results["concentration_errors_normalized"]["combined"]
        print("\n🧪 Concentration Errors (Normalized):")
        print(f"  MAE:       {conc_errors['mae'].mean():.6f}")
        print(f"  RMSE:      {np.sqrt(conc_errors['mse'].mean()):.6f}")
        print(f"  Rel L2:    {conc_errors['rel_l2'].mean() * 100:.3f}%")
        print(f"  Rel L∞:    {conc_errors['rel_linf'].mean() * 100:.3f}%")
        
        # Voltage errors
        voltage_errors = results["voltage_errors"]
        print("\n⚡ Voltage Errors:")
        print(f"  MAE:       {voltage_errors['mae'].mean() * 1000:.3f} mV")
        print(f"  RMSE:      {np.sqrt(voltage_errors['mse'].mean()) * 1000:.3f} mV")
        print(f"  Rel L2:    {voltage_errors['rel_l2'].mean() * 100:.3f}%")
        print(f"  Rel L∞:    {voltage_errors['rel_linf'].mean() * 100:.3f}%")
        
        print("=" * 50)


def print_comparison_table(results: Dict[str, Dict[str, Any]]) -> None:
    """Print a comparison table of error metrics across models."""
    models = list(results.keys())
    
    print("\n" + "=" * 80)
    print("📊 MODEL COMPARISON - ERROR METRICS")
    print("=" * 80)
    
    # Table headers
    print(f"{'Model':<12} {'Conc MAE':<10} {'Conc RMSE':<11} {'Conc L2%':<9} {'Volt MAE':<10} {'Volt RMSE':<11} {'Volt L2%':<9}")
    print("-" * 80)
    
    # Table rows
    for model_name in models:
        result = results[model_name]
        conc_errors = result["concentration_errors_normalized"]["combined"]
        voltage_errors = result["voltage_errors"]
        
        conc_mae = conc_errors['mae'].mean()
        conc_rmse = np.sqrt(conc_errors['mse'].mean())
        conc_l2 = conc_errors['rel_l2'].mean() * 100
        volt_mae = voltage_errors['mae'].mean() * 1000
        volt_rmse = np.sqrt(voltage_errors['mse'].mean()) * 1000
        volt_l2 = voltage_errors['rel_l2'].mean() * 100
        
        print(f"{model_name:<12} {conc_mae:<10.6f} {conc_rmse:<11.6f} {conc_l2:<9.2f} {volt_mae:<10.2f} {volt_rmse:<11.2f} {volt_l2:<9.2f}")
    
    print("=" * 80)
    print("Legend: Conc = Concentration (normalized), Volt = Voltage (mV), L2% = Relative L2 Error (%)")
    print("=" * 80)


def run_comparative_analysis(config_paths: list) -> None:
    """
    Run comparative error analysis across multiple model architectures.
    
    Args:
        config_paths: List of configuration file paths
    """
    print("🔍 Running comparative error analysis...")
    
    results = {}
    
    for config_path in config_paths:
        print(f"\n📊 Analyzing model from {config_path}")
        try:
            config = load_config(config_path)
            model_name = config["model"]["architecture"]
            
            analyzer = ErrorAnalyzer(config)
            result = analyzer.analyze_model(model_architecture=model_name)
            results[model_name] = result
            
        except Exception as e:
            print(f"❌ Error analyzing {config_path}: {str(e)}")
            continue
    
    if len(results) > 1:
        print_comparison_table(results)
    elif len(results) == 1:
        model_name = list(results.keys())[0]
        print_error_summary(results[model_name])
    else:
        print("❌ No models were successfully analyzed for comparison.")
    
    print("✅ Comparative analysis complete!")


def main():
    """Main entry point for error analysis script."""
    parser = argparse.ArgumentParser(
        description="Run error analysis for trained neural network models and print metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze FNO model with latest trained weights
  python print_model_errors.py configs/errors/FNO.yaml
  
  # Analyze CAPE-FNO2 with specific anode model
  python print_model_errors.py configs/errors/CAPE_FNO2.yaml --model_anode models/CAPE_FNO2/anode_model.msgpack
  
  # Analyze DON with specific cathode model
  python print_model_errors.py configs/errors/DON.yaml --model_cathode models/DON/cathode_model.msgpack
  
  # Run comparative analysis across multiple models
  python print_model_errors.py configs/errors/FNO.yaml configs/errors/CAPE_FNO2.yaml --compare
        """
    )
    
    parser.add_argument(
        "config",
        nargs="+",
        help="Path(s) to error analysis configuration file(s)"
    )
    
    parser.add_argument(
        "--model_anode",
        type=str,
        help="Path to specific anode model file (overrides config default)"
    )
    
    parser.add_argument(
        "--model_cathode", 
        type=str,
        help="Path to specific cathode model file (overrides config default)"
    )
    
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run comparative analysis across multiple models (requires multiple configs)"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    for config_path in args.config:
        if not Path(config_path).exists():
            print(f"❌ Error: Configuration file {config_path} does not exist")
            sys.exit(1)
    
    try:
        if args.compare or len(args.config) > 1:
            # Comparative analysis
            if len(args.config) < 2:
                print("❌ Error: Comparative analysis requires at least 2 configuration files")
                sys.exit(1)
                
            if args.model_anode or args.model_cathode:
                print("⚠️  Warning: Model override flags ignored in comparative mode")
                
            run_comparative_analysis(args.config)
            
        else:
            # Single model analysis
            config_path = args.config[0]
            config = load_config(config_path)
            
            # Override model paths if provided
            anode_path = args.model_anode or config.get("models", {}).get("anode_path")
            cathode_path = args.model_cathode or config.get("models", {}).get("cathode_path") 
            
            # Run analysis
            results = run_single_model_analysis(
                config_path=config_path,
                anode_model_path=anode_path,
                cathode_model_path=cathode_path
            )
            
            # Print summary
            print_error_summary(results)
            
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        sys.exit(1)
    
    print("\n✅ Error analysis complete!")


if __name__ == "__main__":
    main() 