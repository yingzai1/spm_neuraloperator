import numpy as np
import jax
import jax.numpy as jnp
import optax
from typing import Dict, Any, Tuple
from tqdm import trange

from .base_trainer import BaseTrainer


class FNOTrainer(BaseTrainer):
    """Trainer for vanilla FNO models."""
    
    def __init__(self, config_path: str):
        super().__init__(config_path)
        self.train_data = None
        self.test_data = None
        
    def preprocess_data(self, train_data: Dict, test_data: Dict):
        """Preprocess data for FNO training."""
        from old.src.util.FNO_util import preprocess_data
        
        preprocessing_config = self.config["training"]["preprocessing"]
        
        # Extract current and concentration data
        train_I = np.array(train_data["current"])
        test_I = np.array(test_data["current"])
        
        # Store for later use
        self.train_data = train_data
        self.test_data = test_data
        self.train_I = train_I
        self.test_I = test_I
        
        return train_I, test_I, train_data, test_data
    
    def create_model(self):
        """Create FNO model."""
        from ..models import FNO
        
        model_config = self.config["training"]["model"]
        return FNO(
            k_modes=model_config["k_modes"],
            fno_depth=model_config["fno_depth"],
            hidden_channels=model_config["hidden_channels"],
            output_channels=model_config["output_channels"]
        )
    
    def train_electrode(self, electrode: str) -> Tuple[Any, list, list]:
        """Train FNO for specific electrode."""
        if self.train_data is None:
            self.preprocess_data(*self.load_dataset())
        
        from old.src.util.FNO_util import preprocess_data, data_loader_noD, remove_padding
        
        preprocessing_config = self.config["training"]["preprocessing"]
        training_config = self.config["training"]["training"]
        
        # Get electrode-specific data
        if electrode == "anode":
            train_c0 = np.array(self.train_data["c0_anode"])
            test_c0 = np.array(self.test_data["c0_anode"])
            train_cn = np.array(self.train_data["cn_anode"])
            test_cn = np.array(self.test_data["cn_anode"])
        else:  # cathode
            train_c0 = np.array(self.train_data["c0_cathode"])
            test_c0 = np.array(self.test_data["c0_cathode"])
            train_cn = np.array(self.train_data["cn_cathode"])
            test_cn = np.array(self.test_data["cn_cathode"])
        
        # Preprocess data
        X_train, Y_train = preprocess_data(
            self.train_I, train_c0, train_cn,
            preprocessing_config["num_samples_I"],
            preprocessing_config["num_samples_c0"],
            preprocessing_config["padding_r"],
            preprocessing_config["padding_t"]
        )
        
        X_test, Y_test = preprocess_data(
            self.test_I, test_c0, test_cn,
            preprocessing_config["num_samples_I"],
            preprocessing_config["num_samples_c0"],
            preprocessing_config["padding_r"],
            preprocessing_config["padding_t"]
        )
        
        print(f"X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")
        
        # Create model
        model = self.create_model()
        
        # Initialize parameters
        init_key = jax.random.PRNGKey(42)
        params = model.init(init_key, X_train[:1, ...])
        
        # Setup optimizer
        n_total = self.config["training"]["dataset"]["n_total"]
        optimizer, opt_state = self.setup_optimizer(params, n_total)
        
        # Define loss function
        def loss_fn(params, X_batch, Y_batch):
            preds = model.apply(params, X_batch)
            return self.relative_l2_loss(preds.flatten(), Y_batch.flatten())
        
        # Define training step
        @jax.jit
        def train_step(params, opt_state, X_batch, Y_batch):
            loss_value, grads = jax.value_and_grad(loss_fn)(params, X_batch, Y_batch)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss_value
        
        # Training loop
        train_losses = []
        test_losses = []
        num_epochs = training_config["num_epochs"]
        batch_size = training_config["batch_size"]
        
        pbar = trange(num_epochs, desc=f"Training {electrode}")
        
        for epoch in pbar:
            total_train_loss = 0.0
            count = 0
            
            for X_batch, Y_batch in data_loader_noD(X_train, Y_train, batch_size):
                params, opt_state, loss_value = train_step(
                    params, opt_state, jnp.array(X_batch), jnp.array(Y_batch)
                )
                total_train_loss += float(loss_value)
                count += 1
            
            total_test_loss = 0.0
            count2 = 0
            
            for X_batch, Y_batch in data_loader_noD(X_test, Y_test, 200):
                loss_value = loss_fn(params, jnp.array(X_batch), jnp.array(Y_batch))
                total_test_loss += float(loss_value)
                count2 += 1
            
            avg_train_loss = total_train_loss / max(1, count)
            avg_test_loss = total_test_loss / max(1, count2)
            
            train_losses.append(avg_train_loss)
            test_losses.append(avg_test_loss)
            
            desc_str = f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f}"
            pbar.set_description(desc_str)
        
        # ── Generate concentration plots using plotter system ──────────────────
        if self.config["training"]["output"].get("plot_results", True):
            data_dict = {
                "X_train": X_train,
                "Y_train": Y_train,
                "X_test": X_test,
                "Y_test": Y_test,
                "train_cn": train_cn,
                "test_cn": test_cn,
                "train_I": self.train_I,
                "test_I": self.test_I
            }
            
            self.plotter.plot_model_predictions(
                model.apply, params, data_dict, self.config, electrode
            )

        # Clean up large arrays
        del X_train, Y_train, X_test, Y_test
        
        return params, train_losses, test_losses 