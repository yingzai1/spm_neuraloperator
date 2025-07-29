import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import jax.numpy as jnp
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from typing import Optional, Tuple
from .base_plotter import BasePlotter


class ConcentrationPlotter(BasePlotter):
    """Handles plotting of concentration predictions and comparisons."""
    
    def plot_concentration_comparison(self,
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
        Create comprehensive concentration comparison plots.
        
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
        # Calculate error profiles
        train_error_line = jnp.mean(jnp.abs(c_train_pred - c_train_true), axis=0)
        test_error_line = jnp.mean(jnp.abs(c_test_pred - c_test_true), axis=0)
        
        # Create coordinate grids
        t = np.linspace(0, 1, c_train_pred.shape[1])
        r = np.linspace(0, 1, c_train_pred.shape[0])
        T_plot, R_plot = jnp.meshgrid(t, r)
        
        # Data for normalization
        train_data = np.concatenate([c_train_pred.ravel(), c_train_true.ravel()])
        test_data = np.concatenate([c_test_pred.ravel(), c_test_true.ravel()])
        
        train_min, train_max = train_data.min(), train_data.max()
        test_min, test_max = test_data.min(), test_data.max()
        
        # Create normalization objects
        train_norm = Normalize(vmin=train_min, vmax=train_max)
        test_norm = Normalize(vmin=test_min, vmax=test_max)
        
        # Create figure and layout
        fig = plt.figure(figsize=(20, 15))
        gs = gridspec.GridSpec(2, 3, figure=fig)
        
        # Training prediction
        ax1 = fig.add_subplot(gs[0, 0])
        contour1 = ax1.contourf(T_plot * t_max, R_plot * particle_radius * 1e6, 
                               c_train_pred, levels=50, cmap='viridis', norm=train_norm)
        self.set_common_labels(ax1, 
                              'Time [s]', 
                              'Radial position [µm]',
                              f'Predicted Lithium Concentration in\n{electrode.title()} from Training Sample [mol/m³]')
        
        # Training ground truth
        ax2 = fig.add_subplot(gs[1, 0])
        contour2 = ax2.contourf(T_plot * t_max, R_plot * particle_radius * 1e6,
                               c_train_true, levels=50, cmap='viridis', norm=train_norm)
        self.set_common_labels(ax2,
                              'Time [s]',
                              'Radial position [µm]', 
                              f'True Lithium Concentration in\n{electrode.title()} from Training Sample [mol/m³]')
        
        # Test prediction
        ax3 = fig.add_subplot(gs[0, 1])
        contour3 = ax3.contourf(T_plot * t_max, R_plot * particle_radius * 1e6,
                               c_test_pred, levels=50, cmap='viridis', norm=test_norm)
        self.set_common_labels(ax3,
                              'Time [s]',
                              'Radial position [µm]',
                              f'Predicted Lithium Concentration in\n{electrode.title()} from Test Sample [mol/m³]')
        
        # Test ground truth
        ax4 = fig.add_subplot(gs[1, 1])
        contour4 = ax4.contourf(T_plot * t_max, R_plot * particle_radius * 1e6,
                               c_test_true, levels=50, cmap='viridis', norm=test_norm)
        self.set_common_labels(ax4,
                              'Time [s]',
                              'Radial position [µm]',
                              f'True Lithium Concentration in\n{electrode.title()} from Test Sample [mol/m³]')
        
        # Right column with current and error plots
        right_gs = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[:, 2], hspace=0.4)
        
        # Training current
        ax_curr = fig.add_subplot(right_gs[0, 0])
        ax_curr.plot(t * t_max, I_train)
        self.set_common_labels(ax_curr, 'Time [s]', 'Current [A]', 'Training Current')
        
        # Test current
        ax_volt = fig.add_subplot(right_gs[1, 0])
        ax_volt.plot(t * t_max, I_test, linestyle='-')
        self.set_common_labels(ax_volt, 'Time [s]', 'Current [A]', 'Test Current')
        
        # Error comparison
        ax_err = fig.add_subplot(right_gs[2, 0])
        ax_err.plot(t * t_max, train_error_line, color='grey', label='Training Error', linewidth=2)
        ax_err.plot(t * t_max, test_error_line, color='black', label='Test Error', linewidth=2)
        self.set_common_labels(ax_err, 'Time [s]', 'Absolute Error [mol/m³]', 'Absolute Error')
        ax_err.legend()
        
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
    
    def plot_concentration_evolution(self,
                                   concentrations: np.ndarray,
                                   times: np.ndarray,
                                   radial_positions: np.ndarray,
                                   electrode: str,
                                   save: bool = True,
                                   show_plot: bool = False) -> Optional[plt.Figure]:
        """
        Plot concentration evolution over time at different radial positions.
        
        Args:
            concentrations: Concentration data (r, t)
            times: Time array
            radial_positions: Radial position array
            electrode: Electrode name
            save: Whether to save the plot
            show_plot: Whether to display the plot
            
        Returns:
            Figure object if save=False, None otherwise
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Time evolution at different radial positions
        n_positions = min(5, len(radial_positions))
        position_indices = np.linspace(0, len(radial_positions)-1, n_positions, dtype=int)
        
        for i, pos_idx in enumerate(position_indices):
            r_val = radial_positions[pos_idx]
            ax1.plot(times, concentrations[pos_idx, :], 
                    label=f'r = {r_val:.3f}', linewidth=2)
        
        ax1.set_xlabel('Time [s]')
        ax1.set_ylabel('Concentration [mol/m³]')
        ax1.set_title(f'{electrode.title()} Concentration vs Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Radial profiles at different times
        n_times = min(5, len(times))
        time_indices = np.linspace(0, len(times)-1, n_times, dtype=int)
        
        for i, time_idx in enumerate(time_indices):
            t_val = times[time_idx]
            ax2.plot(radial_positions, concentrations[:, time_idx],
                    label=f't = {t_val:.0f}s', linewidth=2)
        
        ax2.set_xlabel('Radial Position [normalized]')
        ax2.set_ylabel('Concentration [mol/m³]')
        ax2.set_title(f'{electrode.title()} Radial Concentration Profiles')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if show_plot:
            plt.show()
            
        if save:
            self.save_figure(fig, "concentration_evolution", electrode)
            return None
        else:
            return fig 