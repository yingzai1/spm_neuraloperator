import numpy as np
import jax
import jax.numpy as jnp
import optax
from typing import Dict, Any, Tuple
from tqdm import trange

from .base_trainer import BaseTrainer


class DONTrainer(BaseTrainer):
    """Trainer for DeepONet models."""
    
    def filter_concentration_data(self, train_cn_anode, train_cn_cathode, test_cn_anode, test_cn_cathode):
        """Filter concentration data based on bounds."""
        from old.src.util.postprocess import filter_anode_cathode
        
        bounds_config = self.config["training"]["preprocessing"]["concentration_bounds"]
        
        train_cn_anode, train_cn_cathode, train_mask = filter_anode_cathode(
            train_cn_anode, train_cn_cathode,
            anode_lo=bounds_config["cs_min_a_norm"], 
            anode_hi=bounds_config["cs_max_a_norm"],
            cathode_lo=bounds_config["cs_min_c_norm"], 
            cathode_hi=bounds_config["cs_max_c_norm"]
        )
        
        test_cn_anode, test_cn_cathode, test_mask = filter_anode_cathode(
            test_cn_anode, test_cn_cathode,
            anode_lo=bounds_config["cs_min_a_norm"], 
            anode_hi=bounds_config["cs_max_a_norm"],
            cathode_lo=bounds_config["cs_min_c_norm"], 
            cathode_hi=bounds_config["cs_max_c_norm"]
        )
        
        return (train_cn_anode, train_cn_cathode, train_mask), (test_cn_anode, test_cn_cathode, test_mask)
    
    def create_data_loader(self, train_I, train_c0, train_cn, trunk_points, batch_size):
        """Create data loader for DON training."""
        def data_loader(train_I, train_c0, train_cn, trunk_points, batch_size):
            num_samples = train_I.shape[0]
            permutation = np.random.permutation(num_samples)
            
            for start_idx in range(0, num_samples, batch_size):
                batch_indices = permutation[start_idx:start_idx+batch_size]
                
                I_batch = train_I[batch_indices]
                c0_batch = train_c0[batch_indices]
                cn_batch = train_cn[batch_indices]
                
                yield I_batch, c0_batch, trunk_points, cn_batch
        
        return data_loader(train_I, train_c0, train_cn, trunk_points, batch_size)
    
    def create_model(self):
        """Create DeepONet model."""
        from ..models import DeepONet
        
        model_config = self.config["training"]["model"]
        
        # Create branch and trunk layers
        branch_layers = [model_config["width"]] * model_config["depth"] + [model_config["amount_basis"]]
        trunk_layers = branch_layers.copy()
        
        return DeepONet(branch_layers=branch_layers, trunk_layers=trunk_layers)
    
    def setup_optimizer_with_clipping(self, params, n_total: int):
        """Setup optimizer with gradient clipping."""
        scheduler_config = self.config["training"]["training"]["scheduler"]
        training_config = self.config["training"]["training"]
        batch_size = training_config["batch_size"]
        
        warmup_steps = int(n_total / batch_size * scheduler_config["warmup_steps_multiplier"])
        total_steps = int(warmup_steps * scheduler_config["total_steps_multiplier"])
        
        from .scheduler import cosine_schedule_with_warmup
        scheduler = cosine_schedule_with_warmup(
            warmup_steps=warmup_steps,
            peak_lr=scheduler_config["peak_lr"],
            total_steps=total_steps,
            end_lr=scheduler_config["end_lr"]
        )
        
        # Create optimizer with gradient clipping
        max_grad_norm = training_config["max_grad_norm"]
        optimizer = optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adam(scheduler)
        )
        
        opt_state = optimizer.init(params)
        return optimizer, opt_state
    
    def train_electrode(self, electrode: str, family: str) -> Tuple[Any, list, list]:
        """Train DON for a specific electrode and data family."""
        
        from ..models import generate_trunk_points
        
        preprocessing_config = self.config["training"]["preprocessing"]
        training_config = self.config["training"]["training"]
        
        # Use the data for the current family
        train_data = self.current_train_data
        test_data = self.current_test_data

        # Preprocess data for DON
        train_I, test_I = np.array(train_data["current"]), np.array(test_data["current"])
        train_cn_anode, test_cn_anode = np.array(train_data["cn_anode"]), np.array(test_data["cn_anode"])
        train_c0_anode, test_c0_anode = np.array(train_data["c0_anode"]), np.array(test_data["c0_anode"])
        train_cn_cathode, test_cn_cathode = np.array(train_data["cn_cathode"]), np.array(test_data["cn_cathode"])
        train_c0_cathode, test_c0_cathode = np.array(train_data["c0_cathode"]), np.array(test_data["c0_cathode"])
        train_soc, test_soc = np.array(train_data["soc"]), np.array(test_data["soc"])

        (train_cn_anode, train_cn_cathode, train_mask), (test_cn_anode, test_cn_cathode, test_mask) = \
            self.filter_concentration_data(train_cn_anode, train_cn_cathode, test_cn_anode, test_cn_cathode)
        
        train_I, test_I = train_I[train_mask], test_I[test_mask]
        train_c0_anode, test_c0_anode = train_c0_anode[train_mask], test_c0_anode[test_mask]
        train_c0_cathode, test_c0_cathode = train_c0_cathode[train_mask], test_c0_cathode[test_mask]

        # Get electrode-specific data
        if electrode == "anode":
            train_c0, test_c0 = train_c0_anode, test_c0_anode
            train_cn, test_cn = train_cn_anode, test_cn_anode
        else:  # cathode
            train_c0, test_c0 = train_c0_cathode, test_c0_cathode
            train_cn, test_cn = train_cn_cathode, test_cn_cathode
        
        # Setup trunk points
        t_max = self.config["pybamm"]["t_max"]
        num_samples_I = preprocessing_config["num_samples_I"]
        num_samples_c0 = preprocessing_config["num_samples_c0"]
        
        t = np.linspace(0, t_max, num_samples_I)
        r = np.linspace(0, 1, num_samples_c0)
        trunk_points = generate_trunk_points(r, t/t_max)
        
        print(f"Training {electrode} with {train_cn.shape[0]} samples")
        
        # Create model
        model = self.create_model()
        
        # Initialize parameters
        random_seed = self.config["training"]["dataset"]["random_seed"]
        key1, key2 = jax.random.split(jax.random.PRNGKey(random_seed))
        dummy_I = jax.random.normal(key1, (num_samples_I,))
        dummy_c0 = jax.random.normal(key1, (num_samples_c0,))
        dummy_trunk_input = jax.random.normal(key2, (num_samples_I*num_samples_c0, 2))
        
        params = model.init(jax.random.PRNGKey(42), dummy_I, dummy_c0, dummy_trunk_input)
        
        # Setup optimizer with clipping
        n_total = train_cn.shape[0]  # Use filtered dataset size
        optimizer, opt_state = self.setup_optimizer_with_clipping(params, n_total)
        
        # Define loss functions
        def single_forward(params, I_single, c0_single, trunk_points):
            return model.apply(params, I_single, c0_single, trunk_points)
        
        batch_forward = jax.vmap(
            single_forward,
            in_axes=(None, 0, 0, None),
            out_axes=0
        )
        
        def relative_l2_loss(pred, target):
            norm_target = jnp.mean(target.flatten() ** 2)
            norm_diff = jnp.mean((pred.flatten() - target.flatten()) ** 2)
            norm_target = jnp.where(norm_target == 0, 1e-3, norm_target)
            return norm_diff / norm_target
        
        def loss_fn(params, I_batch, c0_batch, trunk_pts, cn_batch):
            pred = batch_forward(params, I_batch, c0_batch, trunk_pts)
            return relative_l2_loss(pred, cn_batch)
        
        # Define training step
        @jax.jit
        def train_step(params, opt_state, I_batch, c0_batch, trunk_pts, cn_batch):
            loss_value, grads = jax.value_and_grad(loss_fn)(params, I_batch, c0_batch, trunk_pts, cn_batch)
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
            
            for I_batch, c0_batch, trunk_pts, cn_batch in self.create_data_loader(
                train_I, train_c0, train_cn, trunk_points, batch_size
            ):
                params, opt_state, loss_value = train_step(
                    params, opt_state, I_batch, c0_batch, trunk_pts, cn_batch
                )
                
                # Validation checks
                assert jnp.all(jnp.isfinite(I_batch)), "I_batch has NaNs"
                assert jnp.all(jnp.isfinite(c0_batch)), "c0_batch has NaNs"
                assert jnp.all(jnp.isfinite(cn_batch)), "cn_batch has NaNs"
                
                if loss_value > 1e3:
                    print(f"Warning: High loss value detected: {loss_value:.4f} at epoch {epoch+1}")
                
                total_train_loss += float(loss_value)
                count += 1
            
            total_test_loss = 0.0
            count2 = 0
            
            for I_batch, c0_batch, trunk_pts, cn_batch in self.create_data_loader(
                test_I, test_c0, test_cn, trunk_points, batch_size
            ):
                loss_value_test = loss_fn(params, I_batch, c0_batch, trunk_pts, cn_batch)
                total_test_loss += float(loss_value_test)
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
                "train_I": train_I,
                "test_I": test_I,
                "train_c0": train_c0,
                "test_c0": test_c0,
                "train_cn": train_cn,
                "test_cn": test_cn,
                "trunk_points": trunk_points
            }
            
            self.plotter.plot_don_predictions(
                model.apply, params, data_dict, self.config, electrode, family=family
            )
        
        return params, train_losses, test_losses 