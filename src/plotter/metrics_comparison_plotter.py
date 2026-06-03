import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns
from typing import Dict, List, Optional, Any
from pathlib import Path

from .base_plotter import BasePlotter


class ConcentrationMetricsPlotter:
    """Creates concentration error metrics comparison plots."""
    
    def __init__(self, output_dir: str = "plots/metrics_comparison", **kwargs):
        self.plots_dir = Path(output_dir)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
    def create_concentration_metrics_plot(self, 
                                        metric_data: Dict[str, np.ndarray],
                                        models: List[str],
                                        datasets: List[str],
                                        save: bool = True,
                                        show_plot: bool = False) -> Optional[plt.Figure]:
        """
        Create concentration error metrics comparison plot.
        
        Args:
            metric_data: Dictionary with metric names as keys and data arrays as values
            models: List of model names
            datasets: List of dataset names
            save: Whether to save the plot
            show_plot: Whether to display the plot
            
        Returns:
            Figure object if save=False, None otherwise
        """
        # Setup colors and appearance
        pred_cols = sns.color_palette("colorblind", len(datasets))
        label_col = dict(zip(datasets, pred_cols))
        
        # Figure size setup (190mm width as in original)
        mm_to_in = 1 / 25.4
        fig_w = 190 * mm_to_in
        fig_h = fig_w * 0.22  # Slightly taller to accommodate 4 datasets
        
        fig, axs = plt.subplots(1, 4, figsize=(fig_w, fig_h), constrained_layout=True)
        axs = axs.ravel()
        
        # Bar width settings - adjusted for better appearance with 4 datasets
        cluster_frac = 0.8  # Reduced from 0.9 to make bars thinner
        bar_width = cluster_frac / len(datasets)
        offsets = (np.arange(len(datasets)) - (len(datasets) - 1) / 2) * bar_width
        
        for ax, (metric_name, values) in zip(axs, metric_data.items()):
            x = np.arange(len(models))
            for i, ds in enumerate(datasets):
                # Only plot if data is not NaN
                if not np.isnan(values[i]).all():
                    ax.bar(x + offsets[i], values[i],
                           width=bar_width, color=label_col[ds],
                           edgecolor='black', linewidth=0.1,
                           label=ds if ax is axs[0] else "_nolegend_")
            
            ax.set_xticks(x)
            ax.set_xticklabels(models, fontsize=7)  # Slightly smaller
            ax.tick_params(axis='y', labelsize=7)
            
            if metric_name in ['nL2', 'L∞']:
                ax.set_ylabel(r'Error [%]', fontsize=7.5)
            else:  # RMSE & MAE
                ax.set_ylabel(r'Error [$\mathrm{mol\,m^{-3}}$]', fontsize=7.5)
            
            ax.set_yscale('log')
            ax.set_title(metric_name, fontsize=8)  # Smaller title
            ax.margins(x=0.07)
            ax.grid(True, alpha=0.3)  # Add grid
        
        # Shared legend
        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center',
                   ncol=len(datasets), frameon=False, fontsize=7,
                   bbox_to_anchor=(0.5, -0.1))
        
        fig.tight_layout()
        
        if save:
            self.save_figure(fig, "concentration_error_metrics_comparison", format="png")
        
        if show_plot:
            plt.show()
        elif save:
            plt.close(fig)
            return None
        else:
            return fig
    
    def save_figure(self, fig: plt.Figure, name: str, electrode: str = "", 
                   format: str = "png", close_after_save: bool = True) -> Path:
        """Save figure to the metrics comparison directory."""
        filename = f"{name}.{format}"
        filepath = self.plots_dir / filename
        
        fig.savefig(filepath, format=format, bbox_inches="tight", dpi=1200)
        
        if close_after_save:
            plt.close(fig)
            
        print(f"📈 Saved plot to {filepath}")
        return filepath


class VoltageMetricsPlotter:
    """Creates voltage error metrics comparison plots."""
    
    def __init__(self, output_dir: str = "plots/metrics_comparison", **kwargs):
        self.plots_dir = Path(output_dir)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
    def create_voltage_metrics_plot(self, 
                                   metric_data: Dict[str, np.ndarray],
                                   models: List[str],
                                   datasets: List[str],
                                   save: bool = True,
                                   show_plot: bool = False) -> Optional[plt.Figure]:
        """
        Create voltage error metrics comparison plot.
        
        Args:
            metric_data: Dictionary with metric names as keys and data arrays as values
            models: List of model names
            datasets: List of dataset names
            save: Whether to save the plot
            show_plot: Whether to display the plot
            
        Returns:
            Figure object if save=False, None otherwise
        """
        # Setup colors and appearance
        pred_cols = sns.color_palette("colorblind", len(datasets))
        label_col = dict(zip(datasets, pred_cols))
        
        # Figure size setup (190mm width as in original)
        mm_to_in = 1 / 25.4
        fig_w = 190 * mm_to_in
        fig_h = fig_w * 0.22  # Slightly taller to accommodate 4 datasets
        
        fig, axs = plt.subplots(1, 4, figsize=(fig_w, fig_h), constrained_layout=True)
        axs = axs.ravel()
        
        # Bar width settings - adjusted for better appearance with 4 datasets
        cluster_frac = 0.8  # Reduced from 0.9 to make bars thinner
        bar_width = cluster_frac / len(datasets)
        offsets = (np.arange(len(datasets)) - (len(datasets) - 1) / 2) * bar_width
        
        for ax, (metric_name, values) in zip(axs, metric_data.items()):
            x = np.arange(len(models))
            
            # Grouped bars
            for i, ds in enumerate(datasets):
                # Only plot if data is not NaN
                if not np.isnan(values[i]).all():
                    ax.bar(x + offsets[i], values[i],
                           width=bar_width,
                           color=label_col[ds],
                           edgecolor='black', linewidth=0.1,
                           label=ds if ax is axs[0] else "_nolegend_")
            
            # Cosmetics
            ax.set_xticks(x)
            ax.set_xticklabels(models, fontsize=7)  # Slightly smaller
            ax.tick_params(axis='y', labelsize=7)
            
            if metric_name in ['nL2', 'nL∞']:
                ax.set_ylabel(r'Error [%]', fontsize=7.5)
                ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
            else:  # RMSE & MAE
                ax.set_ylabel(r'Error [$mV$]', fontsize=7.5)
                ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
            
            ax.set_title(metric_name, fontsize=8)  # Smaller title
            ax.margins(x=0.07)
            ax.grid(True, alpha=0.3)  # Add grid
        
        # Shared legend
        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels,
                   loc='lower center',
                   bbox_to_anchor=(0.5, -0.1),
                   ncol=len(datasets),
                   frameon=False,
                   fontsize=7)
        
        fig.tight_layout()
        fig.align_ylabels()
        
        if save:
            self.save_figure(fig, "voltage_error_metrics_comparison", format="png")
        
        if show_plot:
            plt.show()
        elif save:
            plt.close(fig)
            return None
        else:
            return fig
    
    def save_figure(self, fig: plt.Figure, name: str, electrode: str = "", 
                   format: str = "png", close_after_save: bool = True) -> Path:
        """Save figure to the metrics comparison directory."""
        filename = f"{name}.{format}"
        filepath = self.plots_dir / filename
        
        fig.savefig(filepath, format=format, bbox_inches="tight", dpi=1200)
        
        if close_after_save:
            plt.close(fig)
            
        print(f"📈 Saved plot to {filepath}")
        return filepath


class MetricsComparisonOrchestrator:
    """Orchestrates the creation of all metrics comparison plots."""
    
    def __init__(self, output_dir: str = "plots/metrics_comparison", **kwargs):
        self.plots_dir = Path(output_dir)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize individual plotters
        self.concentration_plotter = ConcentrationMetricsPlotter(output_dir=output_dir, **kwargs)
        self.voltage_plotter = VoltageMetricsPlotter(output_dir=output_dir, **kwargs)
    
    def prepare_concentration_data(self, model_results: Dict[str, Dict]) -> tuple:
        """
        Prepare concentration error data from model results.
        
        Args:
            model_results: Dictionary with model names as keys and analysis results as values
            
        Returns:
            Tuple of (metric_data, models, datasets)
        """
        # Enforce requested model order for concentration plot
        present_models = list(model_results.keys())
        desired_order_keys = ["DON", "FNO", "CAPE_FNO2"]
        ordered_keys = [k for k in desired_order_keys if k in present_models] + [
            k for k in present_models if k not in desired_order_keys
        ]
        display_map = {"CAPE_FNO2": "PE-FNO"}
        models = [display_map.get(k, k) for k in ordered_keys]
        datasets = ['CC', 'Triangle', 'PLS', 'GRF']  # All current profiles
        
        # Initialize metric arrays
        metric_data = {
            'nL2': np.zeros((len(datasets), len(models))),
            'L∞': np.zeros((len(datasets), len(models))),
            'RMSE': np.zeros((len(datasets), len(models))),
            'MAE': np.zeros((len(datasets), len(models)))
        }
        
        for j, model_key in enumerate(ordered_keys):
            result = model_results[model_key]
            
            for i, dataset in enumerate(datasets):
                if dataset in result:
                    # Get concentration errors for this dataset
                    conc_errors_norm = result[dataset]["concentration_errors_normalized"]["combined"]
                    conc_errors_scaled = result[dataset]["concentration_errors"]["combined"]
                    
                    # Normalized L2 error in percentage
                    metric_data['nL2'][i, j] = conc_errors_norm['rel_l2'].mean() * 100
                    
                    # Normalized L∞ error in percentage
                    metric_data['L∞'][i, j] = conc_errors_norm['rel_linf'].mean() * 100
                    
                    # RMSE in mol/m³
                    metric_data['RMSE'][i, j] = np.sqrt(conc_errors_scaled['mse'].mean())
                    
                    # MAE in mol/m³
                    metric_data['MAE'][i, j] = conc_errors_scaled['mae'].mean()
                else:
                    # Fill with NaN if dataset not available for this model
                    metric_data['nL2'][i, j] = np.nan
                    metric_data['L∞'][i, j] = np.nan
                    metric_data['RMSE'][i, j] = np.nan
                    metric_data['MAE'][i, j] = np.nan
        
        # Use display labels for plotting
        dataset_labels = ['CC', 'TRI', 'PLS', 'GRF']  # Display labels for plots
        return metric_data, models, dataset_labels
    
    def prepare_voltage_data(self, model_results: Dict[str, Dict]) -> tuple:
        """
        Prepare voltage error data from model results.
        
        Args:
            model_results: Dictionary with model names as keys and analysis results as values
            
        Returns:
            Tuple of (metric_data, models, datasets)
        """
        # Enforce requested model order and exclude DON for voltage plot
        present_models = list(model_results.keys())
        # Support both canonical and display names for PE-FNO
        preferred_order = ["FNO", "CAPE_FNO2", "PE-FNO"]  # DON excluded
        ordered_keys = [k for k in preferred_order if k in present_models] + [
            k for k in present_models if k not in set(preferred_order + ["DON"])
        ]
        display_map = {"CAPE_FNO2": "PE-FNO"}
        models = [display_map.get(k, k) for k in ordered_keys]
        datasets = ['CC', 'Triangle', 'PLS', 'GRF']  # All current profiles
        
        # Initialize metric arrays
        metric_data = {
            'nL2': np.zeros((len(datasets), len(models))),
            'nL∞': np.zeros((len(datasets), len(models))),
            'RMSE': np.zeros((len(datasets), len(models))),
            'MAE': np.zeros((len(datasets), len(models)))
        }
        
        for j, model_key in enumerate(ordered_keys):
            result = model_results[model_key]
            
            for i, dataset in enumerate(datasets):
                if dataset in result:
                    voltage_errors = result[dataset]["voltage_errors"]
                    
                    # Normalized L2 error in percentage
                    metric_data['nL2'][i, j] = voltage_errors['rel_l2'].mean() * 100
                    
                    # Normalized L∞ error in percentage
                    metric_data['nL∞'][i, j] = voltage_errors['rel_linf'].mean() * 100
                    
                    # RMSE in mV
                    metric_data['RMSE'][i, j] = np.sqrt(voltage_errors['mse'].mean()) * 1000
                    
                    # MAE in mV
                    metric_data['MAE'][i, j] = voltage_errors['mae'].mean() * 1000
                else:
                    # Fill with NaN if dataset not available for this model
                    metric_data['nL2'][i, j] = np.nan
                    metric_data['nL∞'][i, j] = np.nan
                    metric_data['RMSE'][i, j] = np.nan
                    metric_data['MAE'][i, j] = np.nan
        
        # Use display labels for plotting
        dataset_labels = ['CC', 'TRI', 'PLS', 'GRF']  # Display labels for plots
        return metric_data, models, dataset_labels
    
    def create_all_comparison_plots(self, 
                                   model_results: Dict[str, Dict],
                                   save: bool = True,
                                   show_plot: bool = False) -> None:
        """
        Create all metrics comparison plots.
        
        Args:
            model_results: Dictionary with model names as keys and analysis results as values
            save: Whether to save the plots
            show_plot: Whether to display the plots
        """
        print("📊 Creating metrics comparison plots...")
        
        # Create concentration metrics comparison
        conc_data, models, datasets = self.prepare_concentration_data(model_results)
        self.concentration_plotter.create_concentration_metrics_plot(
            conc_data, models, datasets, save=save, show_plot=show_plot
        )
        
        # Create voltage metrics comparison
        volt_data, models, datasets = self.prepare_voltage_data(model_results)
        self.voltage_plotter.create_voltage_metrics_plot(
            volt_data, models, datasets, save=save, show_plot=show_plot
        )
        
        if save:
            print(f"📈 All metrics comparison plots saved to {self.plots_dir}")
    
    def save_figure(self, fig: plt.Figure, name: str, electrode: str = "", 
                   format: str = "png", close_after_save: bool = True) -> Path:
        """Override to use metrics comparison specific directory structure."""
        filename = f"{name}.{format}"
        filepath = self.plots_dir / filename
        
        fig.savefig(filepath, format=format, bbox_inches="tight", dpi=1200)
        
        if close_after_save:
            plt.close(fig)
            
        print(f"📈 Saved plot to {filepath}")
        return filepath 