import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
from typing import Optional
from .base_plotter import BasePlotter


class ConcentrationComponentsPlotter(BasePlotter):
    """Creates individual components for concentration overview plots."""
    
    def plot_current_profile(self,
                           current: np.ndarray,
                           t_max: float,
                           title: str = "Current Profile") -> plt.Figure:
        """
        Plot current vs time.
        
        Args:
            current: Current array
            t_max: Maximum time in seconds
            title: Plot title
            
        Returns:
            Figure with current plot
        """
        fig, ax = plt.subplots(figsize=(8, 4))
        
        t = np.linspace(0, 1, len(current))
        ax.plot(t * t_max, current, linewidth=2)
        
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Current [A]')
        ax.set_title(title, pad=14)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_error_comparison(self,
                            train_error: np.ndarray,
                            test_error: np.ndarray,
                            t_max: float,
                            title: str = "Absolute Error") -> plt.Figure:
        """
        Plot training and test error comparison.
        
        Args:
            train_error: Training error array
            test_error: Test error array
            t_max: Maximum time in seconds
            title: Plot title
            
        Returns:
            Figure with error comparison
        """
        fig, ax = plt.subplots(figsize=(8, 4))
        
        t = np.linspace(0, 1, len(train_error))
        
        ax.plot(t * t_max, train_error, color='grey', label='Training Error', linewidth=2)
        ax.plot(t * t_max, test_error, color='black', label='Test Error', linewidth=2)
        
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Absolute Error [mol/m³]')
        ax.set_title(title, pad=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_concentration_contour_component(self,
                                           concentration: np.ndarray,
                                           t_max: float,
                                           particle_radius: float,
                                           title: str,
                                           norm=None) -> tuple:
        """
        Create a concentration contour plot component for assembly.
        
        Args:
            concentration: 2D concentration array (r, t)
            t_max: Maximum time in seconds
            particle_radius: Particle radius in meters
            title: Plot title
            norm: Matplotlib normalizer for colorbar consistency
            
        Returns:
            Tuple of (figure, axes, contour_object)
        """
        fig, ax = plt.subplots(figsize=(6, 5))
        
        # Create coordinate grids
        t = np.linspace(0, 1, concentration.shape[1])
        r = np.linspace(0, 1, concentration.shape[0])
        T_plot, R_plot = jnp.meshgrid(t, r)
        
        # Create contour plot
        contour = ax.contourf(T_plot * t_max, R_plot * particle_radius * 1e6,
                             concentration, levels=50, cmap='viridis', norm=norm)
        
        # Set labels and title
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Radial position [µm]')
        ax.set_title(title, pad=14)
        
        return fig, ax, contour
    
    def calculate_error_profiles(self,
                               c_train_pred: np.ndarray,
                               c_train_true: np.ndarray,
                               c_test_pred: np.ndarray,
                               c_test_true: np.ndarray) -> tuple:
        """
        Calculate error profiles for training and test data.
        
        Args:
            c_train_pred: Predicted training concentrations
            c_train_true: True training concentrations
            c_test_pred: Predicted test concentrations
            c_test_true: True test concentrations
            
        Returns:
            Tuple of (train_error_line, test_error_line)
        """
        train_error_line = jnp.mean(jnp.abs(c_train_pred - c_train_true), axis=0)
        test_error_line = jnp.mean(jnp.abs(c_test_pred - c_test_true), axis=0)
        
        return train_error_line, test_error_line 