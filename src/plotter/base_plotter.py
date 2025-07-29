import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import numpy as np


class BasePlotter:
    """Base class for all plotting functionality."""
    
    def __init__(self, 
                 model_name: str,
                 output_dir: str = "plots",
                 timestamp: Optional[str] = None,
                 use_agg_backend: bool = True):
        """
        Initialize plotter.
        
        Args:
            model_name: Name of the model (FNO, CAPE_FNO2, DON)
            output_dir: Base directory for saving plots
            timestamp: Optional timestamp string, auto-generated if None
            use_agg_backend: Whether to use Agg backend for headless plotting
        """
        if use_agg_backend:
            matplotlib.use('Agg')
            
        self.model_name = model_name
        self.timestamp = timestamp or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # Setup output directories
        self.plots_dir = Path(output_dir) / "training" / model_name
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Set default style
        self._setup_matplotlib_style()
    
    def _setup_matplotlib_style(self):
        """Configure matplotlib style for consistent plots."""
        plt.rcParams.update({
            'font.size': 14.5,
            'axes.titlesize': 18,
            'axes.labelsize': 16,
            'xtick.labelsize': 14.5,
            'ytick.labelsize': 14.5,
            'legend.fontsize': 14.5,
            'figure.titlesize': 20,
            'figure.dpi': 100,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight'
        })
    
    def save_figure(self, fig: plt.Figure, name: str, electrode: str = "", 
                   format: str = "png", close_after_save: bool = True) -> Path:
        """
        Save a matplotlib figure.
        
        Args:
            fig: Matplotlib figure to save
            name: Base name for the file
            electrode: Optional electrode suffix (anode/cathode)
            format: File format (png, svg, pdf)
            close_after_save: Whether to close figure after saving
            
        Returns:
            Path to saved file
        """
        electrode_suffix = f"_{electrode}" if electrode else ""
        filename = f"{name}{electrode_suffix}_{self.timestamp}.{format}"
        filepath = self.plots_dir / filename
        
        fig.savefig(filepath, format=format, bbox_inches="tight")
        
        if close_after_save:
            plt.close(fig)
            
        print(f"📈 Saved plot to {filepath}")
        return filepath
    
    def create_subplot_grid(self, rows: int, cols: int, 
                           figsize: tuple = (20, 15)) -> tuple:
        """Create a standardized subplot grid."""
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        return fig, axes
    
    def set_common_labels(self, ax, xlabel: str = None, ylabel: str = None, 
                         title: str = None):
        """Set common axis labels and title."""
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title, pad=14) 