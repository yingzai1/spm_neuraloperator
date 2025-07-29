# Plotter Module

The `src/plotter` module provides a modular and extensible plotting system for neural network model training and analysis. It's designed to be easily extended for future analysis capabilities.

## Architecture

### Base Classes

#### `BasePlotter`
- Common functionality for all plotters
- Handles file saving, matplotlib configuration, directory management
- Provides utility methods for creating subplots and setting labels
- Supports multiple file formats (PNG, SVG, PDF)

### Specialized Plotters

#### `LossPlotter`
- **Purpose**: Training and validation loss visualization
- **Features**:
  - Training progress plots with epoch-wise loss curves
  - Loss comparison across different models/configurations
  - Final loss value annotations
  - Grid overlays and customizable styling

#### `ConcentrationPlotter`  
- **Purpose**: Battery concentration prediction analysis
- **Features**:
  - Comprehensive concentration comparison plots (prediction vs. ground truth)
  - 2D contour plots with proper scaling and colorbars
  - Current profile visualization
  - Error analysis plots
  - Concentration evolution over time and radial position

#### `TrainingPlotter`
- **Purpose**: High-level training workflow plotting
- **Features**:
  - Integrates loss and concentration plotting
  - Model-agnostic prediction visualization
  - Supports FNO, CAPE-FNO2, and DeepONet models
  - Automatic data preprocessing and scaling

## Usage

### Basic Usage

```python
from src.plotter import TrainingPlotter

# Initialize plotter
plotter = TrainingPlotter("FNO", timestamp="2025-07-29_16-00-00")

# Plot training losses
plotter.plot_training_summary(train_losses, test_losses, electrode="anode")

# Plot model predictions
data_dict = {
    "X_train": X_train, "Y_train": Y_train,
    "X_test": X_test, "Y_test": Y_test,
    "train_cn": train_cn, "test_cn": test_cn,
    "train_I": train_I, "test_I": test_I
}
plotter.plot_model_predictions(model.apply, params, data_dict, config, "anode")
```

### Advanced Usage

```python
from src.plotter import LossPlotter, ConcentrationPlotter

# Specialized loss plotting
loss_plotter = LossPlotter("MyModel")
loss_plotter.plot_loss_comparison({
    "Config_A": (train_losses_a, test_losses_a),
    "Config_B": (train_losses_b, test_losses_b)
})

# Specialized concentration analysis
conc_plotter = ConcentrationPlotter("MyModel")
conc_plotter.plot_concentration_evolution(
    concentrations, times, radial_positions, "anode"
)
```

## Integration with Training

The plotting system is automatically integrated into all training workflows:

### FNO Trainer
```python
# Automatically generates:
# - Loss plots: plots/training/FNO/loss_TIMESTAMP.png
# - Concentration plots: plots/training/FNO/concentration_anode_TIMESTAMP.png
```

### CAPE-FNO2 Trainer
```python
# Handles parameter encoding (D, R) automatically
# Generates same plot types as FNO with proper scaling
```

### DeepONet Trainer
```python
# Special handling for trunk/branch architecture
# Reshapes flat predictions to (r, t) format
```

## File Organization

```
plots/
└── training/
    ├── FNO/
    │   ├── loss_2025-07-29_16-00-00.png
    │   ├── concentration_anode_2025-07-29_16-00-00.png
    │   └── concentration_cathode_2025-07-29_16-00-00.png
    ├── CAPE_FNO2/
    │   └── ...
    └── DON/
        └── ...
```

## Configuration

### Plot Settings
Controlled via training configuration:
```yaml
training:
  output:
    plot_results: true  # Enable/disable plotting
    # Additional plot-specific settings can be added here
```

### Matplotlib Settings
Default styling configured in `BasePlotter`:
- High-resolution output (300 DPI)
- Consistent fonts and sizes
- Scientific notation formatting
- Grid overlays where appropriate

## Extensibility

### Adding New Plot Types

1. **Extend existing plotters**:
```python
class LossPlotter(BasePlotter):
    def plot_learning_rate_schedule(self, lr_values, epochs):
        # New plot type
        pass
```

2. **Create new specialized plotters**:
```python
class PerformancePlotter(BasePlotter):
    def plot_inference_speed(self, batch_sizes, times):
        pass
    
    def plot_memory_usage(self, model_sizes, memory):
        pass
```

3. **Extend TrainingPlotter**:
```python
class TrainingPlotter(BasePlotter):
    def plot_hyperparameter_sweep(self, results_dict):
        # New high-level analysis
        pass
```

### Future Analysis Capabilities

The modular design supports easy addition of:

- **Performance Analysis**: Speed, memory, convergence metrics
- **Model Comparison**: Side-by-side architecture comparisons  
- **Hyperparameter Analysis**: Grid search visualization, sensitivity analysis
- **Error Analysis**: Detailed error breakdowns, statistical analysis
- **Physics Analysis**: Battery-specific metrics, electrochemical validation
- **Interactive Plots**: Plotly/Bokeh integration for dynamic analysis

### Data Pipeline Integration

The plotter system can be extended to work with:
- Real-time training monitoring
- Experiment tracking systems (MLflow, Weights & Biases)
- Automated report generation
- Multi-run experiment analysis

## Best Practices

1. **Use consistent timestamps** across all plots in a training run
2. **Provide meaningful electrode names** for file organization
3. **Use save=True for training**, save=False for interactive analysis
4. **Extend specialized plotters** rather than modifying base classes
5. **Follow naming conventions** for consistency across the codebase 