import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
from typing import Optional
from .base_plotter import BasePlotter


class SingleConcentrationPlotter(BasePlotter):
    """Creates individual concentration contour plots."""
    
    def plot_concentration_contour(self,
                                 concentration: np.ndarray,
                                 t_max: float,
                                 particle_radius: float,
                                 title: str,
                                 colorbar_label: str = "Concentration [mol/m³]",
                                 save: bool = False,
                                 filename: str = None,
                                 show_plot: bool = False) -> Optional[plt.Figure]:
        """
        Create a single concentration contour plot.
        
        Args:
            concentration: 2D concentration array (r, t)
            t_max: Maximum time in seconds
            particle_radius: Particle radius in meters
            title: Plot title
            colorbar_label: Label for colorbar
            save: Whether to save the plot
            filename: Filename if saving
            show_plot: Whether to display the plot
            
        Returns:
            Figure object if save=False, None otherwise
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create coordinate grids
        t = np.linspace(0, 1, concentration.shape[1])
        r = np.linspace(0, 1, concentration.shape[0])
        T_plot, R_plot = jnp.meshgrid(t, r)
        
        # Create contour plot
        contour = ax.contourf(T_plot * t_max, R_plot * particle_radius * 1e6,
                             concentration, levels=50, cmap='viridis')
        
        # Add colorbar
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label(colorbar_label)
        
        # Set labels and title
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Radial position [µm]')
        ax.set_title(title, pad=14)
        
        plt.tight_layout()
        
        if show_plot:
            plt.show()
            
        if save and filename:
            self.save_figure(fig, filename)
            return None
        else:
            return fig 