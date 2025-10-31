## Battery Neural Network Models - Error Analysis System

### Setup
```bash
# Install dependencies
uv venv --python $(grep -oP 'python-version\s*=\s*"\K[^"]+' pyproject.toml)
uv sync
```

### Data Generation
```bash
python run_datagen.py configs/data_generation/FNO.yaml
```
Data also avaiable for download [here](https://zenodo.org/uploads/17280393)

### Training
```bash
python train_model.py configs/training/FNO.yaml
```

### Error Analysis
```bash
# Analyze individual models (uses latest trained models)
python print_model_errors.py configs/errors/FNO.yaml
python print_model_errors.py configs/errors/CAPE_FNO2.yaml
python print_model_errors.py configs/errors/DON.yaml

# Override specific model files
python print_model_errors.py configs/errors/FNO.yaml --model_anode models/FNO/anode_model.msgpack
python print_model_errors.py configs/errors/CAPE_FNO2.yaml --model_cathode models/CAPE_FNO2/cathode_model.msgpack

# Compare multiple models
python print_model_errors.py configs/errors/FNO.yaml configs/errors/CAPE_FNO2.yaml --compare
```

The error analysis system will:
- Calculate concentration and voltage prediction errors
- Print detailed error metrics to console
- Support comparative analysis across model architectures

### Metrics Comparison Plots

Generate publication-quality comparison plots for concentration and voltage prediction errors:

```bash
# Create comparison plots for all available models
python plots_concentration_voltage.py

# Specify custom models
python plots_concentration_voltage.py --configs configs/errors/FNO.yaml configs/errors/CAPE_FNO2.yaml

# Custom output directory
python plots_concentration_voltage.py --output_dir plots/paper_figures

# Show plots interactively
python plots_concentration_voltage.py --show
```

The metrics comparison system will:
- Generate publication-quality grouped bar charts
- Compare concentration errors (nL2, L∞, RMSE, MAE)
- Compare voltage errors (nL2, nL∞, RMSE, MAE)
- Save plots in both SVG and PNG formats
- Support custom output directories
