import numpy as np
import jax
import jax.numpy as jnp
import optax
from typing import Dict, Any, Tuple
from tqdm import trange

from .base_trainer import BaseTrainer


class FNOTrainer(BaseTrainer):
    """Trainer for vanilla FNO models."""
    
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
    
    def train_electrode(self, electrode: str, family: str) -> Tuple[Any, list, list]:
        """Train FNO for a specific electrode and data family."""
        
        from old.src.util.FNO_util import preprocess_data, data_loader_noD, remove_padding
        
        preprocessing_config = self.config["training"]["preprocessing"]
        training_config = self.config["training"]["training"]
        
        # Use the data for the current family
        train_data = self.current_train_data
        test_data = self.current_test_data
        
        train_I = np.array(train_data["current"])
        test_I = np.array(test_data["current"])

        # Get electrode-specific data
        if electrode == "anode":
            train_c0 = np.array(train_data["c0_anode"])
            test_c0 = np.array(test_data["c0_anode"])
            train_cn = np.array(train_data["cn_anode"])
            test_cn = np.array(test_data["cn_anode"])
        else:  # cathode
            train_c0 = np.array(train_data["c0_cathode"])
            test_c0 = np.array(test_data["c0_cathode"])
            train_cn = np.array(train_data["cn_cathode"])
            test_cn = np.array(test_data["cn_cathode"])
        
        # Preprocess data
        X_train, Y_train = preprocess_data(
            train_I, train_c0, train_cn,
            preprocessing_config["num_samples_I"],
            preprocessing_config["num_samples_c0"],
            preprocessing_config["padding_r"],
            preprocessing_config["padding_t"]
        )
        
        X_test, Y_test = preprocess_data(
            test_I, test_c0, test_cn,
            preprocessing_config["num_samples_I"],
            preprocessing_config["num_samples_c0"],
            preprocessing_config["padding_r"],
            preprocessing_config["padding_t"]
        )
        
        print(f"X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")
        
        # Create model
        model = self.create_model()
        
        # Set random keys (matching old FNO implementation)
        main_key_seed = self.config["training"]["dataset"]["main_key_seed"]
        random_seed = self.config["training"]["dataset"]["random_seed"]
        main_key = jax.random.PRNGKey(main_key_seed)
        key_train, key_test = jax.random.split(main_key)
        
        # Initialize parameters (matching old FNO implementation)
        init_key = jax.random.PRNGKey(random_seed)
        params = model.init(init_key, X_train[:1, ...])
        
        # Setup optimizer
        n_total = X_train.shape[0]  # Use the actual size of the training data for the family
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
                "train_I": train_I,
                "test_I": test_I
            }
            
            self.plotter.plot_model_predictions(
                model.apply, params, data_dict, self.config, electrode, family=family
            )

        # Clean up large arrays
        del X_train, Y_train, X_test, Y_test
        
        return params, train_losses, test_losses 