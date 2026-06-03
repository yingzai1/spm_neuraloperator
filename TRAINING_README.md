# Refactored Training System

This directory contains the refactored training infrastructure for neural network models used in battery simulation. The system is now configurable via YAML files and organized into modular trainers.

## Overview

The training system has been completely refactored from the original Jupyter notebooks to provide:

- **Configuration-driven training** via YAML files
- **Modular trainer classes** for different model types
- **Consistent training loops** with shared functionality
- **Automatic model saving** to organized directories
- **Loss plotting and visualization**

## Supported Models

### 1. FNO (Fourier Neural Operator)
- **Original file**: `old/src/FNO.py`
- **Trainer**: `src/training/fno_trainer.py`  
- **Config**: `configs/training/FNO.yaml`
- **Description**: Vanilla FNO for concentration prediction

### 2. CAPE-FNO2 (Context-Aware Parameter Encoding FNO)
- **Original file**: `old/src/CAPE_FNO2.py`
- **Trainer**: `src/training/cape_fno2_trainer.py`
- **Config**: `configs/training/CAPE_FNO2.yaml`
- **Description**: FNO with parameter encoding for diffusion coefficients and particle radius

### 3. DON (DeepONet)
- **Original file**: `old/src/DON.py`
- **Trainer**: `src/training/don_trainer.py`
- **Config**: `configs/training/DON.yaml`
- **Description**: Deep Operator Network for concentration prediction

## Usage

### Basic Training
```bash
# Train FNO model
python train_model.py configs/training/FNO.yaml

# Train CAPE-FNO2 model  
python train_model.py configs/training/CAPE_FNO2.yaml

# Train DeepONet model
python train_model.py configs/training/DON.yaml
```

### Override Model Type
```bash
# Force train as FNO regardless of config
python train_model.py configs/training/CAPE_FNO2.yaml --model FNO
```

## Configuration Structure

All config files follow this structure:

```yaml
training:
  model_name: "FNO"  # Model type identifier
  
  dataset:
    parameter_name: "Chen2020"  # PyBaMM parameter set
    family: "GRF"              # Current profile family
    n_total: 11000             # Total samples in dataset
    data_path: "data/dataset/FNO-vanilla/{parameter_name}_{family}_{n_total}.npz"
    test_ratio: 0.1            # Train/test split ratio
    random_seed: 42            # Reproducibility seed
  
  preprocessing:
    padding_t: 5               # Time axis padding
    padding_r: 2               # Radial axis padding  
    num_samples_I: 75          # Time samples
    num_samples_c0: 20         # Radial samples
    
  model:
    # Model-specific hyperparameters
    k_modes: 10
    fno_depth: 6
    hidden_channels: 32
    output_channels: 1
    
  training:
    num_epochs: 150
    batch_size: 20
    
    scheduler:
      warmup_steps_multiplier: 1.0
      peak_lr: 1e-2
      total_steps_multiplier: 25.0  
      end_lr: 1e-4
      
  output:
    model_dir: "models/FNO"    # Where to save trained models
    plot_results: true         # Whether to show training plots
    save_losses: true          # Whether to save loss history

pybamm:
  t_max: 3600                  # Simulation time (seconds)
```

## Model Output

Trained models are saved to:
- `models/FNO/` - FNO models
- `models/CAPE_FNO2/` - CAPE-FNO2 models  
- `models/DON/` - DeepONet models

Each model is saved with metadata including:
- Parameter set name
- Current family
- Dataset size
- Timestamp

## Architecture

### Base Trainer (`src/training/base_trainer.py`)
Common functionality for all trainers:
- Configuration loading
- Dataset loading and splitting
- Optimizer setup with learning rate scheduling
- Loss computation
- Model saving
- Plotting utilities

### Model-Specific Trainers
Each model has its own trainer inheriting from `BaseTrainer`:

- **FNOTrainer**: Handles vanilla FNO training with 4-channel input (I, c0, r, t)
- **CAPEFNO2Trainer**: Handles CAPE-FNO2 with parameter encoding (D, R scaling)
- **DONTrainer**: Handles DeepONet with trunk/branch networks and concentration filtering

## Key Features

### 1. Automatic Data Preprocessing
- Handles electrode-specific data extraction
- Applies model-specific preprocessing (padding, scaling, filtering)
- Manages train/test splits consistently

### 2. Learning Rate Scheduling
- Cosine annealing with warmup
- Configurable peak LR, warmup steps, and decay schedule

### 3. Memory Management
- Automatic cleanup of large arrays after training
- Garbage collection between electrode training

### 4. Error Handling
- Validation of configuration files
- Graceful error reporting
- NaN/infinity checks during training

### 5. Reproducibility
- Fixed random seeds
- Deterministic data splitting
- Consistent initialization

## Migration from Original Code

The key improvements over the original Jupyter notebooks:

1. **Separation of Concerns**: Configuration, data, models, and training are cleanly separated
2. **Reusability**: Common functionality is shared via base classes
3. **Maintainability**: No code duplication across model types
4. **Configurability**: Easy to adjust hyperparameters without code changes
5. **Automation**: No manual cell execution required
6. **Consistency**: All models follow the same training patterns

## Extending the System

To add a new model:

1. Create a trainer class inheriting from `BaseTrainer`
2. Implement the abstract methods:
   - `preprocess_data()`
   - `create_model()`  
   - `train_electrode()`
3. Create a YAML config file
4. Add the trainer to `src/training/__init__.py`
5. Update `train_model.py` trainer registry

This modular design makes it easy to experiment with new architectures while maintaining consistency across the training pipeline. 