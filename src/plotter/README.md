# Plotter Module (Refactored)

The `src/plotter` module provides a modular and extensible plotting system for neural network model training and analysis. The system has been refactored into specialized components for better separation of concerns and extensibility.

## Refactored Architecture

### Base Classes

#### `BasePlotter`
- Common functionality for all plotters
- Handles file saving, matplotlib configuration, directory management
- Provides utility methods for creating subplots and setting labels
- Supports multiple file formats (PNG, SVG, PDF)

### Loss Plotting

#### `LossPlotter`
- **Purpose**: Training and validation loss visualization
- **Features**:
  - Training progress plots with epoch-wise loss curves
  - Loss comparison across different models/configurations
  - Final loss value annotations
  - Grid overlays and customizable styling

### Concentration Plotting (Modular Design)

#### `SingleConcentrationPlotter`
- **Purpose**: Creates individual concentration contour plots
- **Features**:
  - Single concentration contour visualization
  - Customizable titles and colorbar labels
  - Standalone plotting capability

#### `ConcentrationComponentsPlotter`
- **Purpose**: Creates individual plot components
- **Features**:
  - Current profile plotting (`plot_current_profile`)
  - Error comparison plotting (`plot_error_comparison`)
  - Contour component creation (`plot_concentration_contour_component`)
  - Error calculation utilities (`calculate_error_profiles`)

#### `ConcentrationSummaryPlotter`
- **Purpose**: Assembles components into comprehensive overview plots
- **Features**:
  - Combines multiple concentration contours
  - Integrates current profiles and error plots
  - Creates the full 2×3 grid layout with colorbars
  - Maintains consistency with original plot format

#### `ConcentrationPlotter` (Updated)
- **Purpose**: High-level interface for concentration plotting
- **Features**:
  - Uses `ConcentrationSummaryPlotter` internally
  - Maintains backward compatibility
  - Provides the same API as before

### Training Integration

#### `TrainingPlotter`
- **Purpose**: High-level training workflow plotting
- **Features**:
  - Integrates loss and concentration plotting
  - Model-agnostic prediction visualization
  - Supports FNO, CAPE-FNO2, and DeepONet models
  - Automatic data preprocessing and scaling

## Usage

### Basic Usage (Unchanged)

```python
from src.plotter import TrainingPlotter

# Initialize plotter
plotter = TrainingPlotter("FNO", timestamp="2025-07-29_19-00-00")

# Plot training losses (separate files for anode/cathode)
plotter.plot_training_summary(train_losses, test_losses, electrode="anode")

# Plot model predictions (comprehensive overview)
data_dict = {
    "X_train": X_train, "Y_train": Y_train,
    "X_test": X_test, "Y_test": Y_test,
    "train_cn": train_cn, "test_cn": test_cn,
    "train_I": train_I, "test_I": test_I
}
plotter.plot_model_predictions(model.apply, params, data_dict, config, "anode")
```

### New Modular Usage

```python
from src.plotter import (
    LossPlotter, 
    SingleConcentrationPlotter,
    ConcentrationComponentsPlotter,
    ConcentrationSummaryPlotter
)

# Individual loss plots
loss_plotter = LossPlotter("MyModel")
loss_plotter.plot_training_losses(train_losses, test_losses, electrode="anode")

# Single concentration plot
single_plotter = SingleConcentrationPlotter("MyModel")
single_plotter.plot_concentration_contour(
    concentration, t_max, particle_radius, 
    "My Concentration Plot", save=True, filename="my_plot"
)

# Component-based plotting
components = ConcentrationComponentsPlotter("MyModel")
current_fig = components.plot_current_profile(current_data, t_max, "Training Current")
error_fig = components.plot_error_comparison(train_error, test_error, t_max)

# Full summary plot
summary = ConcentrationSummaryPlotter("MyModel")
summary.create_concentration_summary(
    c_train_pred, c_train_true, c_test_pred, c_test_true,
    I_train, I_test, t_max, particle_radius, "anode"
)
```

## Key Improvements in Refactoring

### 1. **Separation of Concerns**
- Each plotter class has a single, well-defined responsibility
- Components can be used independently or combined
- Easier to test and maintain individual components

### 2. **Enhanced Extensibility**
- Easy to add new plot types without modifying existing code
- Components can be mixed and matched for custom visualizations
- Clear interfaces for extending functionality

### 3. **Better Code Organization**
- Complex plotting logic broken into manageable pieces
- Reusable components reduce code duplication
- Clear hierarchy from simple components to complex assemblies

### 4. **Maintained Compatibility**
- Existing training code continues to work unchanged
- Same file naming and organization
- Same plot quality and appearance

## File Organization (Unchanged)

```
plots/
└── training/
    ├── FNO/
    │   ├── loss_anode_2025-07-29_19-00-00.png    # Separate loss plots
    │   ├── loss_cathode_2025-07-29_19-00-00.png  # No more overwriting!
    │   ├── concentration_anode_2025-07-29_19-00-00.png
    │   └── concentration_cathode_2025-07-29_19-00-00.png
    ├── CAPE_FNO2/
    └── DON/
```

## Future Extension Examples

### Adding New Component Types
```python
class VoltageComponentsPlotter(ConcentrationComponentsPlotter):
    def plot_voltage_profile(self, voltage, time, title="Voltage"):
        # New voltage plotting component
        pass
        
    def plot_capacity_fade(self, capacity, cycles, title="Capacity Fade"):
        # New capacity analysis component
        pass
```

### Creating Custom Summary Plots
```python
class CustomSummaryPlotter(BasePlotter):
    def __init__(self, model_name, **kwargs):
        super().__init__(model_name, **kwargs)
        self.components = ConcentrationComponentsPlotter(model_name, **kwargs)
        self.voltage_components = VoltageComponentsPlotter(model_name, **kwargs)
    
    def create_full_analysis_plot(self, data):
        # Combine concentration, voltage, and other components
        # into a custom layout
        pass
```

The refactored architecture makes it easy to build complex visualizations from simple, reusable components while maintaining the existing functionality and API. 