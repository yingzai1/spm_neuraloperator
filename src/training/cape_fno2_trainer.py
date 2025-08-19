import numpy as np
import jax
import jax.numpy as jnp
import optax
from typing import Dict, Any, Tuple
from tqdm import trange

from .base_trainer import BaseTrainer


class CAPEFNO2Trainer(BaseTrainer):
    """Trainer for CAPE-FNO2 models with parameter encoding."""
    
    def setup_parameter_bounds(self):
        """Setup parameter bounds for normalization."""
        bounds_config = self.config["training"]["preprocessing"]["parameter_scaling"]["param_bounds"]
        
        # Create full parameter bounds matching original CAPE-FNO2 format
        # Ensure bounds are converted to float tuples
        self.param_bounds = {
            "D_n|Negative particle diffusivity [m2.s-1]": (float(bounds_config["D_n"][0]), float(bounds_config["D_n"][1])),
            "D_p|Positive particle diffusivity [m2.s-1]": (float(bounds_config["D_p"][0]), float(bounds_config["D_p"][1])),
            "R_n|Negative particle radius [m]": (float(bounds_config["R_n"][0]), float(bounds_config["R_n"][1])),
            "R_p|Positive particle radius [m]": (float(bounds_config["R_p"][0]), float(bounds_config["R_p"][1])),
        }
    
    def scale_parameters(self, train_param_anode, test_param_anode, train_param_cathode, test_param_cathode, key_an, key_ca):
        """Scale parameters for CAPE-FNO2."""
        from old.src.util.FNO_util import normalise_diffusion
        
        log_scale = self.config["training"]["preprocessing"]["parameter_scaling"]["log_scale"]
        
        if log_scale:
            train_param_anode = np.log10(train_param_anode)
            test_param_anode = np.log10(test_param_anode)
            train_param_cathode = np.log10(train_param_cathode)
            test_param_cathode = np.log10(test_param_cathode)
            upper_an = np.log10(self.param_bounds[key_an][1])
            lower_an = np.log10(self.param_bounds[key_an][0])
            upper_ca = np.log10(self.param_bounds[key_ca][1])
            lower_ca = np.log10(self.param_bounds[key_ca][0])
        else:
            upper_an = self.param_bounds[key_an][1]
            lower_an = self.param_bounds[key_an][0]
            upper_ca = self.param_bounds[key_ca][1]
            lower_ca = self.param_bounds[key_ca][0]
            
        train_param_anode = normalise_diffusion(train_param_anode, lower=lower_an, upper=upper_an).reshape(-1, 1)
        test_param_anode = normalise_diffusion(test_param_anode, lower=lower_an, upper=upper_an).reshape(-1, 1)
        train_param_cathode = normalise_diffusion(train_param_cathode, lower=lower_ca, upper=upper_ca).reshape(-1, 1)
        test_param_cathode = normalise_diffusion(test_param_cathode, lower=lower_ca, upper=upper_ca).reshape(-1, 1)
        
        return train_param_anode, test_param_anode, train_param_cathode, test_param_cathode
    
    def create_model(self):
        """Create CAPE-FNO2 model."""
        from ..models import CAPEFNO2
        
        model_config = self.config["training"]["model"]
        
        # Input channels: (I, c0, r, t) = 4 channels  
        input_channels = 4
        
        return CAPEFNO2(
            k_modes=tuple(model_config["k_modes"]),
            input_channels=input_channels,
            fno_depth=model_config["fno_depth"],
            cape_hidden_size=model_config["cape_hidden_size"],
            hidden_channels=model_config["hidden_channels"],
            output_channels=model_config["output_channels"]
        )
    
    def train_electrode(self, electrode: str, family: str) -> Tuple[Any, list, list]:
        """Train CAPE-FNO2 for a specific electrode and data family."""
        
        # Setup parameter bounds for scaling
        self.setup_parameter_bounds()

        from old.src.util.FNO_util import preprocess_data, data_loader_pe
        
        preprocessing_config = self.config["training"]["preprocessing"]
        training_config = self.config["training"]["training"]

        # Use the data for the current family
        train_data = self.current_train_data
        test_data = self.current_test_data

        # Extract and scale parameters
        train_I = np.array(train_data["current"])
        test_I = np.array(test_data["current"])
        
        train_D_anode, test_D_anode, train_D_cathode, test_D_cathode = self.scale_parameters(
            np.array(train_data["D_n"]), np.array(test_data["D_n"]),
            np.array(train_data["D_p"]), np.array(test_data["D_p"]),
            "D_n|Negative particle diffusivity [m2.s-1]", 
            "D_p|Positive particle diffusivity [m2.s-1]"
        )
        
        train_R_anode, test_R_anode, train_R_cathode, test_R_cathode = self.scale_parameters(
            np.array(train_data["R_n"]), np.array(test_data["R_n"]),
            np.array(train_data["R_p"]), np.array(test_data["R_p"]),
            "R_n|Negative particle radius [m]",
            "R_p|Positive particle radius [m]"
        )
        
        # Get electrode-specific data
        if electrode == "anode":
            train_c0 = np.array(train_data["c0_anode"])
            test_c0 = np.array(test_data["c0_anode"])
            train_cn = np.array(train_data["cn_anode"])
            test_cn = np.array(test_data["cn_anode"])
            train_D, test_D = train_D_anode, test_D_anode
            train_R, test_R = train_R_anode, test_R_anode
        else:  # cathode
            train_c0 = np.array(train_data["c0_cathode"])
            test_c0 = np.array(test_data["c0_cathode"])
            train_cn = np.array(train_data["cn_cathode"])
            test_cn = np.array(test_data["cn_cathode"])
            train_D, test_D = train_D_cathode, test_D_cathode
            train_R, test_R = train_R_cathode, test_R_cathode
        
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
        
        # Initialize parameters
        init_key = jax.random.PRNGKey(42)
        dummy_D = jax.random.normal(init_key, (1, 1))
        dummy_R = jax.random.normal(init_key, (1, 1))
        params = model.init(init_key, X_train[:1, ...], dummy_D, dummy_R)
        
        # Setup optimizer
        n_total = X_train.shape[0] # Use the actual size of the training data for the family
        optimizer, opt_state = self.setup_optimizer(params, n_total)
        
        # Define loss function
        def loss_fn(params, X_batch, D_batch, R_batch, Y_batch):
            preds = model.apply(params, X_batch, D_batch, R_batch)
            return self.relative_l2_loss(preds.flatten(), Y_batch.flatten())
        
        # Define training step
        @jax.jit
        def train_step(params, opt_state, X_batch, D_batch, R_batch, Y_batch):
            loss_value, grads = jax.value_and_grad(loss_fn)(params, X_batch, D_batch, R_batch, Y_batch)
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
            
            for X_batch, D_batch, R_batch, Y_batch in data_loader_pe(X_train, train_D, train_R, Y_train, batch_size):
                params, opt_state, loss_value = train_step(
                    params, opt_state, 
                    jnp.array(X_batch), jnp.array(D_batch), jnp.array(R_batch), jnp.array(Y_batch)
                )
                total_train_loss += float(loss_value)
                count += 1
            
            total_test_loss = 0.0
            count2 = 0
            
            for X_batch, D_batch, R_batch, Y_batch in data_loader_pe(X_test, test_D, test_R, Y_test, 200):
                loss_value = loss_fn(params, jnp.array(X_batch), jnp.array(D_batch), jnp.array(R_batch), jnp.array(Y_batch))
                total_test_loss += float(loss_value)
                count2 += 1
            
            avg_train_loss = total_train_loss / max(1, count)
            avg_test_loss = total_test_loss / max(1, count2)
            
            train_losses.append(avg_train_loss)
            test_losses.append(avg_test_loss)
            
            desc_str = f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f}"
            pbar.set_description(desc_str)

        # ── Concentration plots using plotter system ─────────────────────────────────────
        if self.config["training"]["output"].get("plot_results", True):
            data_dict = {
                "X_train": X_train,
                "Y_train": Y_train,
                "X_test": X_test,
                "Y_test": Y_test,
                "train_cn": train_cn,
                "test_cn": test_cn,
                "train_I": train_I,
                "test_I": test_I,
                "train_D": train_D,
                "test_D": test_D,
                "train_R": train_R,
                "test_R": test_R
            }
            
            self.plotter.plot_model_predictions(
                model.apply, params, data_dict, self.config, electrode
            )
        
        # Clean up large arrays
        del X_train, Y_train, X_test, Y_test
        
        return params, train_losses, test_losses 