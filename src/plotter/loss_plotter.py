import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional
from .base_plotter import BasePlotter


class LossPlotter(BasePlotter):
    """Handles plotting of training and validation losses."""
    
    def plot_training_losses(self, 
                           train_losses: List[float], 
                           test_losses: List[float],
                           electrode: str = "",
                           save: bool = True,
                           show_plot: bool = False) -> Optional[plt.Figure]:
        """
        Plot training and test losses over epochs.
        
        Args:
            train_losses: Training loss values per epoch
            test_losses: Test/validation loss values per epoch  
            electrode: Electrode name for filename
            save: Whether to save the plot
            show_plot: Whether to display the plot
            
        Returns:
            Figure object if save=False, None otherwise
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        epochs = range(1, len(train_losses) + 1)
        
        ax.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        ax.plot(epochs, test_losses, 'r-', label='Test Loss', linewidth=2)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f'{self.model_name} Training Progress')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add final loss values as text
        final_train = train_losses[-1]
        final_test = test_losses[-1]
        ax.text(0.02, 0.98, f'Final Train Loss: {final_train:.6f}\nFinal Test Loss: {final_test:.6f}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if show_plot:
            plt.show()
            
        if save:
            self.save_figure(fig, "loss", electrode)
            return None
        else:
            return fig
    
    def plot_loss_comparison(self, 
                           loss_data: dict,
                           save: bool = True,
                           show_plot: bool = False) -> Optional[plt.Figure]:
        """
        Compare losses across different models or configurations.
        
        Args:
            loss_data: Dict with keys as labels and values as (train_losses, test_losses) tuples
            save: Whether to save the plot
            show_plot: Whether to display the plot
            
        Returns:
            Figure object if save=False, None otherwise
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(loss_data)))
        
        for i, (label, (train_losses, test_losses)) in enumerate(loss_data.items()):
            epochs = range(1, len(train_losses) + 1)
            color = colors[i]
            
            ax1.plot(epochs, train_losses, color=color, label=label, linewidth=2)
            ax2.plot(epochs, test_losses, color=color, label=label, linewidth=2)
        
        # Training losses subplot
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Training Loss')
        ax1.set_title('Training Loss Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Test losses subplot  
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Test Loss')
        ax2.set_title('Test Loss Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if show_plot:
            plt.show()
            
        if save:
            self.save_figure(fig, "loss_comparison")
            return None
        else:
            return fig 