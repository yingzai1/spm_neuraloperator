# Error Analysis System

This document provides detailed information about the error analysis and plotting system for neural network models trained on battery simulation data.

## Overview

The error analysis system provides comprehensive evaluation of trained neural network models by:

1. **Loading trained model parameters** for anode and cathode electrodes
2. **Running predictions** on test datasets  
3. **Calculating error metrics** for both concentration and voltage predictions
4. **Generating comparison plots** across different model architectures
5. **Creating summary reports** with detailed error statistics

## Supported Model Architectures

- **FNO (Fourier Neural Operator)**: Vanilla FNO for concentration prediction
- **CAPE-FNO2**: Context-Aware Parameter Encoding FNO with diffusivity parameters
- **DON (DeepONet)**: Deep Operator Network for concentration prediction

## Error Metrics

### Concentration Errors
- **MAE (Mean Absolute Error)**: Average absolute difference between predictions and ground truth
- **RMSE (Root Mean Square Error)**: Square root of mean squared differences
- **Relative L2 Error**: L2 norm of difference divided by L2 norm of ground truth
- **Relative L∞ Error**: Maximum absolute difference divided by maximum ground truth value

### Voltage Errors
Computed from surface concentrations using electrochemical post-processing:
- **MAE**: Reported in millivolts (mV)
- **RMSE**: Reported in millivolts (mV)  
- **Relative L2 Error**: Percentage relative to voltage range
- **Relative L∞ Error**: Percentage relative to voltage range

## Configuration Files

Error analysis configurations are stored in `configs/errors/`:

### `configs/errors/FNO.yaml`
```yaml
error_analysis:
  name: "FNO"
  description: "Error analysis configuration for FNO models"

model:
  architecture: "FNO"
  fno:
    k_modes: 10
    fno_depth: 6
    hidden_channels: 32
    output_channels: 1

data:
  parameter_name: "Prada2013"
  family: "GRF" 
  n_total: 11000
  test_ratio: 0.1
  random_seed: 42
  # ... additional data settings

output:
  plots_dir: "plots/errors/FNO"
  save_formats: ["png", "svg"]
```

## Usage Examples

### Single Model Analysis
```bash
# Analyze FNO with latest trained models
python plot_errors.py configs/errors/FNO.yaml

# Override anode model file
python plot_errors.py configs/errors/FNO.yaml --model_anode models/FNO/specific_anode.msgpack

# Override cathode model file  
python plot_errors.py configs/errors/CAPE_FNO2.yaml --model_cathode models/CAPE_FNO2/specific_cathode.msgpack
```

### Comparative Analysis
```bash
# Compare all three architectures
python plot_errors.py configs/errors/FNO.yaml configs/errors/CAPE_FNO2.yaml configs/errors/DON.yaml --compare

# Compare two architectures with custom output directory
python plot_errors.py configs/errors/FNO.yaml configs/errors/CAPE_FNO2.yaml --compare --output_dir plots/comparison_study
```

## Output Structure

### Single Model Analysis
For each model, plots are saved to the configured directory (e.g., `plots/errors/FNO/`):
- `concentration_error_comparison.png/svg`: Bar charts of concentration errors
- `voltage_error_comparison.png/svg`: Bar charts of voltage errors  
- `error_summary_table.png/svg`: Summary table of all metrics

### Comparative Analysis
When comparing multiple models, plots are saved to `plots/errors/comparison/`:
- `concentration_error_comparison.png/svg`: Side-by-side comparison of concentration errors
- `voltage_error_comparison.png/svg`: Side-by-side comparison of voltage errors
- `error_summary_table.png/svg`: Comprehensive table comparing all models

## Console Output

The script provides detailed console output including:

```
🔍 Analyzing FNO model...
📁 Using anode model: models/FNO/anode_Prada2013_GRF_11000_2025-01-15_14-30-25.msgpack
📁 Using cathode model: models/FNO/cathode_Prada2013_GRF_11000_2025-01-15_14-35-10.msgpack

📊 Error Summary for FNO:
==================================================

🧪 Concentration Errors (Normalized):
  MAE:       0.012345
  RMSE:      0.023456
  Rel L2:    2.345%
  Rel L∞:    5.678%

⚡ Voltage Errors:
  MAE:       12.34 mV
  RMSE:      23.45 mV
  Rel L2:    1.234%
  Rel L∞:    3.456%
==================================================

📊 Creating comprehensive error analysis plots...
📈 Saved plot to plots/errors/FNO/concentration_error_comparison.png
📈 Saved plot to plots/errors/FNO/voltage_error_comparison.png
📈 Saved plot to plots/errors/FNO/error_summary_table.png

✅ Error analysis complete!
```

## Implementation Details

### Error Analyzer (`src/analysis/error_analyzer.py`)
- Loads datasets and applies concentration filtering
- Handles model loading and parameter deserialization
- Preprocesses data according to model architecture requirements
- Runs predictions and calculates comprehensive error metrics
- Performs voltage calculations using electrochemical post-processing

### Error Plotter (`src/plotter/error_plotter.py`)
- Extends the base plotter system for error-specific visualizations
- Creates grouped bar charts for concentration error comparisons
- Generates voltage error comparison plots with appropriate units
- Produces summary tables with formatted error statistics
- Supports both PNG and SVG output formats

### Data Processing Pipeline
1. **Dataset Loading**: Load NPZ files containing simulation data
2. **Data Filtering**: Remove samples with concentrations outside valid ranges
3. **Model Loading**: Deserialize trained model parameters from msgpack files
4. **Preprocessing**: Apply architecture-specific data preprocessing
5. **Prediction**: Run model inference on test data
6. **Error Calculation**: Compute comprehensive error metrics
7. **Visualization**: Generate comparison plots and summary tables

## Model-Specific Considerations

### FNO Models
- Use grid-based data representation with padding
- Require padding removal after prediction
- Support standard FNO preprocessing pipeline

### CAPE-FNO2 Models  
- Require additional diffusivity parameters
- Use context-aware parameter encoding
- Need parameter normalization for stable training

### DeepONet Models
- Use branch-trunk architecture with different data format
- Require trunk point generation for spatial-temporal grid
- Use vectorized model application for batch predictions

## Troubleshooting

### Common Issues

1. **Model files not found**:
   - Ensure models are trained and saved in the expected directories
   - Use `--model_anode` or `--model_cathode` flags to specify exact paths

2. **Dataset missing**:
   - Verify dataset files exist in the `data/` directory
   - Check configuration file has correct dataset parameters

3. **Memory issues**:
   - Reduce batch size in data processing if encountering OOM errors
   - Consider using smaller test datasets for initial testing

4. **Import errors**:
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Verify JAX installation with CUDA support if using GPU

### Performance Optimization

- **GPU Usage**: Error analysis automatically uses JAX's default device (GPU if available)
- **Batch Processing**: Large datasets are processed in batches to manage memory
- **Parallel Processing**: Model predictions use JAX's vectorized operations for efficiency

## Extension Points

The error analysis system is designed to be extensible:

1. **New Error Metrics**: Add custom metrics to `calc_error_metrics` function
2. **Additional Plots**: Extend `ErrorPlotter` with new visualization methods
3. **Model Architectures**: Add support for new models in `ErrorAnalyzer.create_model()`
4. **Post-processing**: Implement custom voltage calculation methods in `calculate_voltage_errors()` 