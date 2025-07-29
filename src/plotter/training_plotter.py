import numpy as np
import jax.numpy as jnp
from typing import Dict, Any, Optional, Tuple
from .base_plotter import BasePlotter
from .loss_plotter import LossPlotter
from .concentration_plotter import ConcentrationPlotter


class TrainingPlotter(BasePlotter):
    """
    Main plotter class for training workflows.
    Combines loss and concentration plotting capabilities.
    """
    
    def __init__(self, model_name: str, timestamp: Optional[str] = None, **kwargs):
        super().__init__(model_name, timestamp=timestamp, **kwargs)
        
        # Initialize sub-plotters with same configuration
        self.loss_plotter = LossPlotter(model_name, timestamp=self.timestamp, **kwargs)
        self.concentration_plotter = ConcentrationPlotter(model_name, timestamp=self.timestamp, **kwargs)
    
    def plot_training_summary(self,
                            train_losses: list,
                            test_losses: list, 
                            electrode: str = "",
                            save: bool = True,
                            show_plot: bool = False) -> None:
        """
        Plot training loss summary.
        
        Args:
            train_losses: Training loss history
            test_losses: Test loss history
            electrode: Electrode name for saving
            save: Whether to save plots
            show_plot: Whether to display plots
        """
        self.loss_plotter.plot_training_losses(
            train_losses, test_losses, electrode, save, show_plot
        )
    
    def plot_model_predictions(self,
                             model_apply_fn,
                             params: Dict[str, Any],
                             data_dict: Dict[str, Any],
                             config: Dict[str, Any],
                             electrode: str,
                             save: bool = True,
                             show_plot: bool = False) -> None:
        """
        Generate and plot model predictions for concentration analysis.
        
        Args:
            model_apply_fn: Model apply function
            params: Model parameters
            data_dict: Dictionary containing processed data
            config: Training configuration
            electrode: Electrode name (anode/cathode)
            save: Whether to save plots
            show_plot: Whether to display plots
        """
        # Extract data
        X_train = data_dict["X_train"]
        Y_train = data_dict["Y_train"]
        X_test = data_dict["X_test"]
        Y_test = data_dict["Y_test"]
        train_cn = data_dict["train_cn"]
        test_cn = data_dict["test_cn"]
        train_I = data_dict["train_I"]
        test_I = data_dict["test_I"]
        
        # Get configuration
        preprocessing_config = config["training"]["preprocessing"]
        dataset_config = config["training"]["dataset"]
        t_max = config["pybamm"]["t_max"]
        
        # Get battery parameters
        import pybamm
        dataset_config = config["training"]["dataset"]
        
        # Handle array-based parameter names
        parameter_names = dataset_config["parameter_name"]
        if isinstance(parameter_names, list):
            parameter_name = parameter_names[0]
        else:
            parameter_name = parameter_names
            
        param_set = pybamm.ParameterValues(parameter_name)
        
        if electrode == "anode":
            c_max = param_set["Maximum concentration in negative electrode [mol.m-3]"]
            particle_radius = param_set["Negative particle radius [m]"]
        else:
            c_max = param_set["Maximum concentration in positive electrode [mol.m-3]"]
            particle_radius = param_set["Positive particle radius [m]"]
        
        # Pick random samples for visualization
        rng = np.random.default_rng(42)  # Fixed seed for reproducibility
        train_idx = rng.integers(0, X_train.shape[0])
        test_idx = rng.integers(0, X_test.shape[0])
        
        # Generate predictions
        predictions = self._generate_predictions(
            model_apply_fn, params, data_dict, train_idx, test_idx, preprocessing_config
        )
        
        # Scale concentrations to physical units
        scaled_data = self._scale_concentrations(predictions, train_cn, test_cn, train_idx, test_idx, c_max)
        
        # Plot concentration comparison
        self.concentration_plotter.plot_concentration_comparison(
            scaled_data["c_train_pred_scaled"],
            scaled_data["c_train_true_scaled"],
            scaled_data["c_test_pred_scaled"],
            scaled_data["c_test_true_scaled"],
            train_I[train_idx],
            test_I[test_idx],
            t_max,
            particle_radius,
            electrode,
            save,
            show_plot
        )
    
    def _generate_predictions(self,
                            model_apply_fn,
                            params: Dict[str, Any],
                            data_dict: Dict[str, Any],
                            train_idx: int,
                            test_idx: int,
                            preprocessing_config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate model predictions and remove padding."""
        from old.src.util.FNO_util import remove_padding
        
        X_train = data_dict["X_train"]
        X_test = data_dict["X_test"]
        
        # Check if model requires additional parameters (CAPE-FNO2)
        extra_train_args = []
        extra_test_args = []
        
        if "train_D" in data_dict:  # CAPE-FNO2 case
            extra_train_args = [data_dict["train_D"][train_idx][None, ...], 
                              data_dict["train_R"][train_idx][None, ...]]
            extra_test_args = [data_dict["test_D"][test_idx][None, ...],
                             data_dict["test_R"][test_idx][None, ...]]
        
        # Generate predictions
        c_train_pred_full = model_apply_fn(params, X_train[train_idx][None, ...], *extra_train_args)
        c_test_pred_full = model_apply_fn(params, X_test[test_idx][None, ...], *extra_test_args)
        
        # Remove padding
        padding_r = preprocessing_config["padding_r"]
        padding_t = preprocessing_config["padding_t"]
        
        c_train_pred = jnp.squeeze(remove_padding(c_train_pred_full, padding_r, padding_t), axis=(0, -1))
        c_test_pred = jnp.squeeze(remove_padding(c_test_pred_full, padding_r, padding_t), axis=(0, -1))
        
        return {
            "c_train_pred": c_train_pred,
            "c_test_pred": c_test_pred
        }
    
    def _scale_concentrations(self,
                            predictions: Dict[str, Any],
                            train_cn: np.ndarray,
                            test_cn: np.ndarray,
                            train_idx: int,
                            test_idx: int,
                            c_max: float) -> Dict[str, Any]:
        """Scale concentrations to physical units."""
        c_train_pred = predictions["c_train_pred"]
        c_test_pred = predictions["c_test_pred"]
        
        c_train_true = jnp.squeeze(train_cn[train_idx])
        c_test_true = jnp.squeeze(test_cn[test_idx])
        
        # Scale to physical units
        c_train_pred_scaled = c_train_pred * c_max
        c_test_pred_scaled = c_test_pred * c_max
        c_train_true_scaled = c_train_true * c_max
        c_test_true_scaled = c_test_true * c_max
        
        return {
            "c_train_pred_scaled": c_train_pred_scaled,
            "c_test_pred_scaled": c_test_pred_scaled,
            "c_train_true_scaled": c_train_true_scaled,
            "c_test_true_scaled": c_test_true_scaled
        }
    
    def plot_don_predictions(self,
                           model_apply_fn,
                           params: Dict[str, Any],
                           data_dict: Dict[str, Any],
                           config: Dict[str, Any],
                           electrode: str,
                           save: bool = True,
                           show_plot: bool = False) -> None:
        """
        Special plotting function for DeepONet predictions.
        
        Args:
            model_apply_fn: DeepONet model apply function
            params: Model parameters
            data_dict: Dictionary containing DON-specific data
            config: Training configuration
            electrode: Electrode name
            save: Whether to save plots
            show_plot: Whether to display plots
        """
        # Extract DON-specific data
        train_I = data_dict["train_I"]
        test_I = data_dict["test_I"]
        train_c0 = data_dict["train_c0"]
        test_c0 = data_dict["test_c0"]
        train_cn = data_dict["train_cn"]
        test_cn = data_dict["test_cn"]
        trunk_points = data_dict["trunk_points"]
        
        # Configuration
        preprocessing_config = config["training"]["preprocessing"]
        dataset_config = config["training"]["dataset"]
        t_max = config["pybamm"]["t_max"]
        
        # Battery parameters
        import pybamm
        dataset_config = config["training"]["dataset"]
        
        # Handle array-based parameter names
        parameter_names = dataset_config["parameter_name"]
        if isinstance(parameter_names, list):
            parameter_name = parameter_names[0]
        else:
            parameter_name = parameter_names
            
        param_set = pybamm.ParameterValues(parameter_name)
        
        if electrode == "anode":
            c_max = param_set["Maximum concentration in negative electrode [mol.m-3]"]
            particle_radius = param_set["Negative particle radius [m]"]
        else:
            c_max = param_set["Maximum concentration in positive electrode [mol.m-3]"]
            particle_radius = param_set["Positive particle radius [m]"]
        
        # Pick random samples
        rng = np.random.default_rng(42)
        train_idx = rng.integers(0, len(train_I))
        test_idx = rng.integers(0, len(test_I))
        
        # Generate predictions
        c_train_pred_flat = model_apply_fn(params, train_I[train_idx], train_c0[train_idx], trunk_points)
        c_test_pred_flat = model_apply_fn(params, test_I[test_idx], test_c0[test_idx], trunk_points)
        
        # Reshape to (r, t) format
        num_samples_c0 = preprocessing_config["num_samples_c0"]
        num_samples_I = preprocessing_config["num_samples_I"]
        
        c_train_pred = c_train_pred_flat.reshape(num_samples_c0, num_samples_I)
        c_test_pred = c_test_pred_flat.reshape(num_samples_c0, num_samples_I)
        c_train_true = train_cn[train_idx].reshape(num_samples_c0, num_samples_I)
        c_test_true = test_cn[test_idx].reshape(num_samples_c0, num_samples_I)
        
        # Scale to physical units
        c_train_pred_scaled = c_train_pred * c_max
        c_test_pred_scaled = c_test_pred * c_max
        c_train_true_scaled = c_train_true * c_max
        c_test_true_scaled = c_test_true * c_max
        
        # Plot
        self.concentration_plotter.plot_concentration_comparison(
            c_train_pred_scaled,
            c_train_true_scaled,
            c_test_pred_scaled,
            c_test_true_scaled,
            train_I[train_idx],
            test_I[test_idx],
            t_max,
            particle_radius,
            electrode,
            save,
            show_plot
        ) 