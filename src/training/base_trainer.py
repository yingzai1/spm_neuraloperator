import yaml
import numpy as np
import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from tqdm import trange

from .scheduler import cosine_schedule_with_warmup


class BaseTrainer(ABC):
    """Base class for all model trainers."""
    
    def __init__(self, config_path: str):
        """Initialize trainer with configuration."""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        from datetime import datetime
        from ..plotter import TrainingPlotter
        self.model_name = self.config["training"]["model_name"]
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # Initialize plotter
        self.plotter = TrainingPlotter(self.model_name, timestamp=self.timestamp)
        self.setup_paths()
        self.setup_device()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load YAML configuration file."""
        with self.config_path.open("r") as f:
            return yaml.safe_load(f)
    
    def setup_paths(self):
        """Setup output directories."""
        self.model_dir = Path(self.config["training"]["output"]["model_dir"])
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def setup_device(self):
        """Setup JAX device and print info."""
        print("All devices:", jax.devices())
        print("Default backend:", jax.default_backend())
    
    def load_dataset(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Load and split dataset into train/test."""
        dataset_config = self.config["training"]["dataset"]
        data_path = dataset_config["data_path"].format(**dataset_config)
        
        print(f"Loading dataset from: {data_path}")
        data = np.load(data_path)
        
        # Train/test split
        from old.src.util.FNO_util import train_test_split
        train_data, test_data = train_test_split(
            data, 
            N_total=dataset_config["n_total"],
            test_ratio=dataset_config["test_ratio"],
            seed=dataset_config["random_seed"]
        )
        
        return train_data, test_data
    
    def setup_optimizer(self, params, n_total: int) -> Tuple[optax.GradientTransformation, Any]:
        """Setup optimizer with learning rate schedule."""
        scheduler_config = self.config["training"]["training"]["scheduler"]
        batch_size = self.config["training"]["training"]["batch_size"]
        
        warmup_steps = int(n_total / batch_size * scheduler_config["warmup_steps_multiplier"])
        total_steps = int(warmup_steps * scheduler_config["total_steps_multiplier"])
        
        scheduler = cosine_schedule_with_warmup(
            warmup_steps=warmup_steps,
            peak_lr=scheduler_config["peak_lr"],
            total_steps=total_steps,
            end_lr=scheduler_config["end_lr"]
        )
        
        optimizer = optax.adam(scheduler)
        opt_state = optimizer.init(params)
        
        return optimizer, opt_state
    
    def relative_l2_loss(self, pred: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
        """Compute relative L2 loss."""
        norm_target = jnp.linalg.norm(target, 2)
        norm_diff = jnp.linalg.norm(pred - target, 2)
        norm_target = jnp.where(norm_target == 0, 1e-12, norm_target)
        return norm_diff / norm_target
    
    def plot_losses(self, train_losses, test_losses, electrode: str = ""):
        """Plot training and test losses using the plotter system.

        Each electrode now gets its own plot file so results are not overwritten.
        """
        if self.config["training"]["output"].get("plot_results", True):
            self.plotter.plot_training_summary(train_losses, test_losses, electrode=electrode)
    
    def save_model(self, params, electrode: str = ""):
        """Save trained model parameters."""
        from old.src.util import functions
        dataset_config = self.config["training"]["dataset"]
        
        prefix = f"{electrode}_" if electrode else ""
        filename = functions.save_model_params(
            params, 
            directory=str(self.model_dir),
            prefix=prefix,
            family=dataset_config["family"],
            parameter_name=dataset_config["parameter_name"],
            N_total=dataset_config["n_total"]
        )
        return filename
    
    def save_figure(self, fig, name: str):
        """Save a matplotlib figure to the plots directory."""
        import matplotlib.pyplot as plt
        filepath = self.plots_dir / f"{name}_{self.timestamp}.png"
        fig.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"📈 Saved plot to {filepath}")
    
    @abstractmethod
    def preprocess_data(self, train_data: Dict, test_data: Dict) -> Tuple[Any, Any, Any, Any]:
        """Preprocess training and test data. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def create_model(self) -> Any:
        """Create model instance. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def train_electrode(self, electrode: str) -> Tuple[Any, list, list]:
        """Train model for specific electrode. Must be implemented by subclasses."""
        pass
    
    def train(self):
        """Main training loop."""
        print(f"Starting training for {self.config['training']['model_name']}")
        
        # Load dataset
        train_data, test_data = self.load_dataset()
        
        # Train for both electrodes
        for electrode in ["anode", "cathode"]:
            print(f"\n--- Training {electrode} model ---")
            params, train_losses, test_losses = self.train_electrode(electrode)
            
            # Save results
            self.save_model(params, electrode)
            self.plot_losses(train_losses, test_losses, electrode)
            
            # Clean up memory
            import gc
            gc.collect() 