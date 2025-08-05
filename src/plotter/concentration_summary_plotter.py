import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import jax.numpy as jnp
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from typing import Optional
from .base_plotter import BasePlotter
from .concentration_components_plotter import ConcentrationComponentsPlotter


class ConcentrationSummaryPlotter(BasePlotter):
    """Assembles concentration components into comprehensive overview plots."""
    
    def __init__(self, model_name: str, timestamp: Optional[str] = None, **kwargs):
        super().__init__(model_name, timestamp=timestamp, **kwargs)
        self.components_plotter = ConcentrationComponentsPlotter(model_name, timestamp=timestamp, **kwargs)
    
    def create_concentration_summary(self,
                                   c_train_pred: np.ndarray,
                                   c_train_true: np.ndarray,
                                   c_test_pred: np.ndarray,
                                   c_test_true: np.ndarray,
                                   I_train: np.ndarray,
                                   I_test: np.ndarray,
                                   t_max: float,
                                   particle_radius: float,
                                   electrode: str,
                                   save: bool = True,
                                   show_plot: bool = False) -> Optional[plt.Figure]:
        """
        Create comprehensive concentration comparison summary plot.
        
        Args:
            c_train_pred: Predicted training concentrations (r, t)
            c_train_true: True training concentrations (r, t)
            c_test_pred: Predicted test concentrations (r, t)
            c_test_true: True test concentrations (r, t)
            I_train: Training current profile
            I_test: Test current profile
            t_max: Maximum time in seconds
            particle_radius: Particle radius in meters
            electrode: Electrode name (anode/cathode)
            save: Whether to save the plot
            show_plot: Whether to display the plot
            
        Returns:
            Figure object if save=False, None otherwise
        """
        # Calculate error profiles using components plotter
        train_error_line, test_error_line = self.components_plotter.calculate_error_profiles(
            c_train_pred, c_train_true, c_test_pred, c_test_true
        )
        
        # Data for normalization (ensure positive scales)
        train_data = np.concatenate([c_train_pred.ravel(), c_train_true.ravel()])
        test_data = np.concatenate([c_test_pred.ravel(), c_test_true.ravel()])
        
        train_min = max(train_data.min(), 0.0)
        train_max = train_data.max()
        test_min = max(test_data.min(), 0.0)
        test_max = test_data.max()
        
        # Create normalization objects
        train_norm = Normalize(vmin=train_min, vmax=train_max)
        test_norm = Normalize(vmin=test_min, vmax=test_max)
        
        # Create main figure and layout
        fig = plt.figure(figsize=(20, 15))
        gs = gridspec.GridSpec(2, 3, figure=fig)
        
        # Training prediction contour
        ax1 = fig.add_subplot(gs[0, 0])
        contour1 = self._create_contour_subplot(
            ax1, c_train_pred, t_max, particle_radius, train_norm,
            f'Predicted Lithium Concentration in\n{electrode.title()} from Training Sample [mol/m³]'
        )
        
        # Training ground truth contour
        ax2 = fig.add_subplot(gs[1, 0])
        contour2 = self._create_contour_subplot(
            ax2, c_train_true, t_max, particle_radius, train_norm,
            f'True Lithium Concentration in\n{electrode.title()} from Training Sample [mol/m³]'
        )
        
        # Test prediction contour
        ax3 = fig.add_subplot(gs[0, 1])
        contour3 = self._create_contour_subplot(
            ax3, c_test_pred, t_max, particle_radius, test_norm,
            f'Predicted Lithium Concentration in\n{electrode.title()} from Test Sample [mol/m³]'
        )
        
        # Test ground truth contour
        ax4 = fig.add_subplot(gs[1, 1])
        contour4 = self._create_contour_subplot(
            ax4, c_test_true, t_max, particle_radius, test_norm,
            f'True Lithium Concentration in\n{electrode.title()} from Test Sample [mol/m³]'
        )
        
        # Right column with current and error plots
        self._create_right_column_subplots(
            fig, gs, I_train, I_test, train_error_line, test_error_line, t_max
        )
        
        plt.tight_layout(rect=[0, 0.125, 1, 1])
        
        # Add colorbars
        self._add_colorbars(fig, train_norm, test_norm)
        
        if show_plot:
            plt.show()
            
        if save:
            self.save_figure(fig, "concentration", electrode)
            return None
        else:
            return fig
    
    def _create_contour_subplot(self, ax, concentration, t_max, particle_radius, norm, title):
        """Create a single contour subplot."""
        # Create coordinate grids
        t = np.linspace(0, 1, concentration.shape[1])
        r = np.linspace(0, 1, concentration.shape[0])
        T_plot, R_plot = jnp.meshgrid(t, r)
        
        # Create contour plot
        contour = ax.contourf(T_plot * t_max, R_plot * particle_radius * 1e6,
                             concentration, levels=50, cmap='viridis', norm=norm)
        
        self.set_common_labels(ax, 'Time [s]', 'Radial position [µm]', title)
        return contour
    
    def _create_right_column_subplots(self, fig, gs, I_train, I_test, train_error_line, test_error_line, t_max):
        """Create the right column subplots (current and error plots)."""
        right_gs = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[:, 2], hspace=0.4)
        
        # Training current
        ax_curr = fig.add_subplot(right_gs[0, 0])
        t = np.linspace(0, 1, len(I_train))
        ax_curr.plot(t * t_max, I_train)
        self.set_common_labels(ax_curr, 'Time [s]', 'Current [A]', 'Training Current')
        
        # Test current
        ax_volt = fig.add_subplot(right_gs[1, 0])
        t = np.linspace(0, 1, len(I_test))
        ax_volt.plot(t * t_max, I_test, linestyle='-')
        self.set_common_labels(ax_volt, 'Time [s]', 'Current [A]', 'Test Current')
        
        # Error comparison
        ax_err = fig.add_subplot(right_gs[2, 0])
        t = np.linspace(0, 1, len(train_error_line))
        ax_err.plot(t * t_max, train_error_line, color='grey', label='Training Error', linewidth=2)
        ax_err.plot(t * t_max, test_error_line, color='black', label='Test Error', linewidth=2)
        self.set_common_labels(ax_err, 'Time [s]', 'Absolute Error [mol/m³]', 'Absolute Error')
        ax_err.legend()
    
    def _add_colorbars(self, fig, train_norm, test_norm):
        """Add colorbars to the concentration plot."""
        # Create ScalarMappables for colorbars
        sm_train = ScalarMappable(norm=train_norm, cmap='viridis')
        sm_train.set_array([])
        sm_test = ScalarMappable(norm=test_norm, cmap='viridis')
        sm_test.set_array([])
        
        # Add colorbars
        cbar_width = 0.35
        cbar_height = 0.02
        cbar_train_ax = fig.add_axes([0.15, 0.05, cbar_width, cbar_height])
        cbar_test_ax = fig.add_axes([0.55, 0.05, cbar_width, cbar_height])
        
        cbar_train = fig.colorbar(sm_train, cax=cbar_train_ax, orientation='horizontal')
        cbar_train.set_label('Training Scale [mol/m³]', labelpad=12)
        cbar_train.ax.xaxis.set_label_position('top')
        cbar_train.ax.xaxis.set_ticks_position('top')
        
        cbar_test = fig.colorbar(sm_test, cax=cbar_test_ax, orientation='horizontal')
        cbar_test.set_label('Test Scale [mol/m³]') 