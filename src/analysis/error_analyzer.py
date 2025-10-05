import numpy as np
import jax
import jax.numpy as jnp
import pybamm
import flax
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union
from glob import glob
import yaml

from ..models import FNO, CAPEFNO2, DeepONet, generate_trunk_points


class ErrorAnalyzer:
    """Comprehensive error analysis for trained models."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize error analyzer.
        
        Args:
            config: Configuration dictionary containing model and data settings
        """
        self.config = config
        self.model_config = config["model"]
        self.data_config = config["data"]
        self.error_config = config.get("error_analysis", {})
        # Optional per-architecture model directory overrides, e.g., {"FNO": "models-00-ali/FNO"}
        self.model_dirs = self.model_config.get("model_dirs", {})
        
        # Import utility functions from old structure
        from old.src.util.FNO_util import preprocess_data, train_test_split, remove_padding, normalise_diffusion
        from old.src.util.postprocess import filter_anode_cathode, calc_error_metrics, calc_error_metrics_all
        from old.src.util import functions
        
        self.preprocess_data = preprocess_data
        self.train_test_split = train_test_split
        self.remove_padding = remove_padding
        self.normalise_diffusion = normalise_diffusion
        self.filter_anode_cathode = filter_anode_cathode
        self.calc_error_metrics = calc_error_metrics
        self.calc_error_metrics_all = calc_error_metrics_all
        self.functions = functions
        
    def _canonical_architecture(self, name: str) -> str:
        """Map various architecture aliases to a canonical internal name."""
        alias = (name or "").strip().lower()
        mapping = {
            "fno": "FNO",
            "don": "DON",
            "cape_fno2": "CAPE_FNO2",
            "pe-fno": "CAPE_FNO2",
            "pe_fno": "CAPE_FNO2",
            "pefno": "CAPE_FNO2",
        }
        return mapping.get(alias, name)

    def _config_key_for_architecture(self, canonical_arch: str) -> str:
        """Return the config subkey under model for a canonical architecture name."""
        config_keys = {
            "FNO": "fno",
            "DON": "don",
            "CAPE_FNO2": "cape_fno2",
        }
        if canonical_arch not in config_keys:
            raise ValueError(f"Unknown model architecture: {canonical_arch}")
        return config_keys[canonical_arch]

    def load_dataset(self) -> Tuple[Dict, Dict]:
        """Load and split dataset."""
        parameter_name = self.data_config["parameter_name"]
        family = self.data_config["family"]
        n_total = self.data_config["n_total"]
        dataset_dir = self.data_config.get("dataset_dir", "data")  # Default to "data" for backward compatibility
        
        # Handle single family (the multiple family logic is now in analyze_model)
        if isinstance(family, list):
            raise ValueError("load_dataset should only be called with a single family. Use analyze_model for multiple families.")
        
        # Construct the dataset path
        data_path = Path(dataset_dir) / f"{parameter_name}_{family}_{n_total}.npz"
        
        if not data_path.exists():
            # Fallback for old data structure if needed
            fallback_path = Path("data") / f"{parameter_name}_{family}_{n_total}.npz"
            if fallback_path.exists():
                data_path = fallback_path
            else:
                raise FileNotFoundError(f"Dataset not found: {data_path}")
            
        print(f"📂 Loading dataset from {data_path}")
        data = np.load(data_path)
        
        train_data, test_data = self.train_test_split(
            data, 
            N_total=n_total, 
            test_ratio=self.data_config["test_ratio"], 
            seed=self.data_config["random_seed"]
        )
        
        return train_data, test_data
    
    def filter_data(self, train_data: Dict, test_data: Dict) -> Tuple[Dict, Dict]:
        """Filter data to remove samples outside valid concentration ranges."""
        # Extract concentration data
        train_cn_anode = np.array(train_data["cn_anode"])
        train_cn_cathode = np.array(train_data["cn_cathode"])
        test_cn_anode = np.array(test_data["cn_anode"])
        test_cn_cathode = np.array(test_data["cn_cathode"])
        
        # Battery parameters for normalization
        parameter_name = self.data_config["parameter_name"]
        params_bat = pybamm.ParameterValues(parameter_name)
        cs_max_a = params_bat["Maximum concentration in negative electrode [mol.m-3]"]
        cs_max_c = params_bat["Maximum concentration in positive electrode [mol.m-3]"]
        
        # Normalized concentration bounds
        cs_max_a_norm = 1.0
        cs_max_c_norm = 1.0
        cs_min_a_norm = 0.0
        cs_min_c_norm = 0.0
        
        # Filter data
        train_cn_anode_filt, train_cn_cathode_filt, train_mask = self.filter_anode_cathode(
            train_cn_anode, train_cn_cathode,
            anode_lo=cs_min_a_norm, anode_hi=cs_max_a_norm,
            cathode_lo=cs_min_c_norm, cathode_hi=cs_max_c_norm
        )
        
        test_cn_anode_filt, test_cn_cathode_filt, test_mask = self.filter_anode_cathode(
            test_cn_anode, test_cn_cathode,
            anode_lo=cs_min_a_norm, anode_hi=cs_max_a_norm,
            cathode_lo=cs_min_c_norm, cathode_hi=cs_max_c_norm
        )
        
        # Apply masks to all data
        filtered_train_data = {}
        filtered_test_data = {}
        
        for key, values in train_data.items():
            filtered_train_data[key] = np.array(values)[train_mask]
            
        for key, values in test_data.items():
            filtered_test_data[key] = np.array(values)[test_mask]
            
        return filtered_train_data, filtered_test_data
    
    def find_latest_model(self, model_type: str, electrode: str, current_profile: Optional[str] = None) -> Optional[str]:
        """
        Find the most recently trained model for a given type, electrode, and current profile.
        This function strictly searches for electrode-specific models matching the given profile.
        """
        # Honor override directory if provided for this architecture
        override_dir = self.model_dirs.get(model_type)
        model_dir = Path(override_dir) if override_dir else Path(f"models/{model_type}")
        if not model_dir.exists():
            return None
        
        # If a current profile is specified, we ONLY look for models matching it.
        if current_profile:
            profile_mapping = {
                'CC': 'CC', 'Triangle': 'Triangle', 'PLS': 'PLS', 'GRF': 'GRF'
            }
            profile_id = profile_mapping.get(current_profile, current_profile)
            
            # Pattern for electrode-specific models with profile
            pattern = f"{electrode}_*_{profile_id}_*.msgpack"
            model_files = list(model_dir.glob(pattern))
            
            if model_files:
                # Sort by modification time and return the latest
                latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
                return str(latest_model)
            else:
                # If no model is found for this specific profile, return None.
                return None
        else:
            # If no profile is specified, find the latest model for the electrode.
            pattern = f"{electrode}_*.msgpack"
            model_files = list(model_dir.glob(pattern))
            
            if model_files:
                latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
                return str(latest_model)

        return None
    
    def load_model_params(self, model_path: str, model_architecture: str) -> Any:
        """Load model parameters from file."""
        param_bytes = self.functions.load_model_params(model_path)

        # Create dummy model to get the parameter structure
        canonical_arch = self._canonical_architecture(model_architecture)
        model = self.create_model(canonical_arch)

        if canonical_arch == "FNO":
            init_key = jax.random.PRNGKey(42)
            dummy_input = jax.random.normal(init_key, (1, 24, 85, 4))
            dummy_params = model.init(init_key, dummy_input)
            params = flax.serialization.from_bytes(dummy_params, param_bytes)
            return params
        elif canonical_arch == "CAPE_FNO2":
            init_key = jax.random.PRNGKey(42)
            dummy_input = jax.random.normal(init_key, (1, 24, 85, 4))
            dummy_D = jax.random.normal(init_key, (1, 1))
            dummy_R = jax.random.normal(init_key, (1, 1))
            dummy_params = model.init(init_key, dummy_input, dummy_D, dummy_R)
            params = flax.serialization.from_bytes(dummy_params, param_bytes)
            return params
        elif canonical_arch == "DON":
            # Autodetect basis size to avoid shape mismatches across families
            preprocessing = self.data_config["preprocessing"]
            num_samples_I = preprocessing["num_samples_I"]
            num_samples_c0 = preprocessing["num_samples_c0"]

            init_key = jax.random.PRNGKey(42)
            dummy_I = jax.random.normal(init_key, (num_samples_I,))
            dummy_c0 = jax.random.normal(init_key, (num_samples_c0,))
            dummy_trunk = jax.random.normal(init_key, (num_samples_I * num_samples_c0, 2))

            # Try a few common basis sizes
            candidate_basis = [self.model_config["don"].get("amount_basis", 16), 16, 32, 64, 128]
            tried = set()
            last_err = None
            for M in candidate_basis:
                if M in tried:
                    continue
                tried.add(M)
                try:
                    # Build a model with this basis size
                    branch_layers = [self.model_config["don"]["width"]] * self.model_config["don"]["depth"] + [M]
                    trunk_layers = list(branch_layers)
                    from ..models.deeponet import DeepONet as DONModel  # local import
                    don_model = DONModel(branch_layers=branch_layers, trunk_layers=trunk_layers)
                    dummy_params = don_model.init(init_key, dummy_I, dummy_c0, dummy_trunk)
                    params = flax.serialization.from_bytes(dummy_params, param_bytes)
                    # Dry run apply
                    _ = don_model.apply(params, dummy_I, dummy_c0, dummy_trunk)
                    # Cache detected basis for downstream use by returning tuple
                    return params
                except Exception as e:
                    last_err = e
                    continue
            raise RuntimeError(f"Failed to load DON params from {model_path} with autodetect: {last_err}")
        else:
            raise ValueError(f"Unknown model architecture: {canonical_arch}")
    
    def create_model(self, model_architecture: str) -> Any:
        """Create model instance based on architecture."""
        canonical_arch = self._canonical_architecture(model_architecture)
        config_key = self._config_key_for_architecture(canonical_arch)
        model_params = self.model_config[config_key]
        
        if canonical_arch == "FNO":
            return FNO(
                k_modes=model_params["k_modes"],
                fno_depth=model_params["fno_depth"],
                hidden_channels=model_params["hidden_channels"],
                output_channels=model_params["output_channels"]
            )
        elif canonical_arch == "CAPE_FNO2":
            k_modes = model_params["k_modes"]
            if isinstance(k_modes, list):
                k_modes_tuple = tuple(k_modes)
            else:
                k_modes_tuple = (k_modes, k_modes)
                
            return CAPEFNO2(
                k_modes=k_modes_tuple,
                fno_depth=model_params["fno_depth"],
                cape_hidden_size=model_params["cape_hidden_size"],
                hidden_channels=model_params["hidden_channels"],
                input_channels=model_params["input_channels"],
                output_channels=model_params["output_channels"]
            )
        elif canonical_arch == "DON":
            branch_layers = [model_params["width"]] * model_params["depth"] + [model_params["amount_basis"]]
            trunk_layers = branch_layers.copy()
            return DeepONet(
                branch_layers=branch_layers,
                trunk_layers=trunk_layers
            )
        else:
            raise ValueError(f"Unknown model architecture: {canonical_arch}")
    
    def preprocess_model_data(self, train_data: Dict, test_data: Dict, 
                            model_architecture: str, electrode: str) -> Tuple[Any, Any, Any, Any]:
        """Preprocess data for specific model architecture."""
        preprocessing = self.data_config["preprocessing"]
        canonical_arch = self._canonical_architecture(model_architecture)
        
        # Extract current and concentration data
        train_I = train_data["current"]
        test_I = test_data["current"]
        
        if electrode == "anode":
            train_c0 = train_data["c0_anode"]
            test_c0 = test_data["c0_anode"]
            train_cn = train_data["cn_anode"]
            test_cn = test_data["cn_anode"]
        else:  # cathode
            train_c0 = train_data["c0_cathode"]
            test_c0 = test_data["c0_cathode"]
            train_cn = train_data["cn_cathode"]
            test_cn = test_data["cn_cathode"]
        
        if canonical_arch in ["FNO", "CAPE_FNO2"]:
            # Grid-based preprocessing
            X_train, Y_train = self.preprocess_data(
                train_I, train_c0, train_cn,
                preprocessing["num_samples_I"],
                preprocessing["num_samples_c0"],
                preprocessing["padding_r"],
                preprocessing["padding_t"]
            )
            
            X_test, Y_test = self.preprocess_data(
                test_I, test_c0, test_cn,
                preprocessing["num_samples_I"],
                preprocessing["num_samples_c0"],
                preprocessing["padding_r"],
                preprocessing["padding_t"]
            )
            
            return X_train, Y_train, X_test, Y_test
            
        elif canonical_arch == "DON":
            # DeepONet uses different data format
            num_samples_I = preprocessing["num_samples_I"]
            num_samples_c0 = preprocessing["num_samples_c0"]
            
            # Generate trunk points
            t_max = self.data_config["t_max"]
            t = np.linspace(0, 1, num_samples_I)
            r = np.linspace(0, 1, num_samples_c0)
            trunk_points = generate_trunk_points(r, t)
            
            return train_I, train_c0, test_I, test_c0, train_cn, test_cn, trunk_points
        
        else:
            raise ValueError(f"Unknown model architecture: {canonical_arch}")
    
    def run_predictions(self, model: Any, params: Any, model_architecture: str,
                       model_data: Tuple, electrode: str) -> Tuple[np.ndarray, np.ndarray]:
        """Run model predictions on test data."""
        canonical_arch = self._canonical_architecture(model_architecture)
        
        if canonical_arch == "FNO":
            X_train, Y_train, X_test, Y_test = model_data
            
            # Run predictions
            c_test_pred = model.apply(params, X_test)
            
            # Remove padding from both predictions and targets
            preprocessing = self.data_config["preprocessing"]
            c_test_pred_unpadded = self.remove_padding(
                c_test_pred, 
                preprocessing["padding_r"], 
                preprocessing["padding_t"]
            )
            c_test_true_unpadded = self.remove_padding(
                Y_test,
                preprocessing["padding_r"], 
                preprocessing["padding_t"]
            )
            
            pred = c_test_pred_unpadded.squeeze()
            true = c_test_true_unpadded.squeeze()
            print(f"    ▶ {model_architecture} {electrode} shapes (pred/true): {pred.shape} / {true.shape}")
            return pred, true
        
        elif canonical_arch == "CAPE_FNO2":
            X_train, Y_train, X_test, Y_test = model_data
            
            # Need additional parameter data for CAPE-FNO2
            # This requires loading parameter data from the dataset
            train_data, test_data = self.load_dataset()
            train_data, test_data = self.filter_data(train_data, test_data)
            
            # Get diffusivity parameters
            if electrode == "anode":
                param_key = self.data_config["param_keys"]["anode"]
                test_D = test_data[param_key]
                # Get radius parameter - try different possible keys
                test_R_key = None
                for r_key in ["R_n", "R_anode", "radius_anode"]:
                    if r_key in test_data:
                        test_R_key = r_key
                        break
                test_R = test_data[test_R_key] if test_R_key else np.ones_like(test_D) * 5e-6
            else:
                param_key = self.data_config["param_keys"]["cathode"] 
                test_D = test_data[param_key]
                # Get radius parameter - try different possible keys
                test_R_key = None
                for r_key in ["R_p", "R_cathode", "radius_cathode"]:
                    if r_key in test_data:
                        test_R_key = r_key
                        break
                test_R = test_data[test_R_key] if test_R_key else np.ones_like(test_D) * 2e-6
                
            # Normalize parameters using the scaling function from old code
            # Log scale diffusivity and radius
            test_D_log = np.log10(test_D)
            test_R_log = np.log10(test_R)
            
            # Normalize to [-1, 1] range
            if electrode == "anode":
                # Diffusivity bounds
                d_lower, d_upper = np.log10(1e-18), np.log10(1e-14)
                # Radius bounds  
                r_lower, r_upper = np.log10(4e-6), np.log10(1.5e-5)
            else:
                # Diffusivity bounds
                d_lower, d_upper = np.log10(1e-18), np.log10(1e-14)
                # Radius bounds
                r_lower, r_upper = np.log10(1e-8), np.log10(1.5e-5)
            
            test_D_norm = (2 * (test_D_log - d_lower) / (d_upper - d_lower) - 1).reshape(-1, 1)
            test_R_norm = (2 * (test_R_log - r_lower) / (r_upper - r_lower) - 1).reshape(-1, 1)
            
            # Run predictions with batch processing to handle memory constraints
            batch_size = 256  # Adjust this if needed
            num_samples = X_test.shape[0]
            
            if num_samples <= batch_size:
                # Small dataset, process all at once
                c_test_pred = model.apply(params, X_test, test_D_norm, test_R_norm)
            else:
                # Large dataset, process in batches
                print(f"    🔄 Processing {num_samples} samples in batches of {batch_size}...")
                predictions = []
                
                for i in range(0, num_samples, batch_size):
                    end_idx = min(i + batch_size, num_samples)
                    batch_X = X_test[i:end_idx]
                    batch_D = test_D_norm[i:end_idx]
                    batch_R = test_R_norm[i:end_idx]
                    
                    try:
                        batch_pred = model.apply(params, batch_X, batch_D, batch_R)
                        predictions.append(batch_pred)
                        print(f"      ✓ Processed batch {i//batch_size + 1}/{(num_samples + batch_size - 1)//batch_size}")
                    except Exception as e:
                        print(f"      ⚠️ Batch {i//batch_size + 1} failed: {str(e)}")
                        # Try with smaller batch size
                        smaller_batch_size = batch_size // 2
                        if smaller_batch_size >= 1:
                            print(f"      🔄 Retrying with smaller batch size: {smaller_batch_size}")
                            for j in range(i, end_idx, smaller_batch_size):
                                small_end = min(j + smaller_batch_size, end_idx)
                                small_batch_X = X_test[j:small_end]
                                small_batch_D = test_D_norm[j:small_end]
                                small_batch_R = test_R_norm[j:small_end]
                                
                                small_batch_pred = model.apply(params, small_batch_X, small_batch_D, small_batch_R)
                                predictions.append(small_batch_pred)
                        else:
                            raise e
                
                c_test_pred = np.concatenate(predictions, axis=0)
            
            # Remove padding from both predictions and targets
            preprocessing = self.data_config["preprocessing"]
            c_test_pred_unpadded = self.remove_padding(
                c_test_pred,
                preprocessing["padding_r"],
                preprocessing["padding_t"]
            )
            c_test_true_unpadded = self.remove_padding(
                Y_test,
                preprocessing["padding_r"],
                preprocessing["padding_t"]
            )
            
            pred = c_test_pred_unpadded.squeeze()
            true = c_test_true_unpadded.squeeze()
            print(f"    ▶ {canonical_arch} {electrode} shapes (pred/true): {pred.shape} / {true.shape}")
            return pred, true
            
        elif canonical_arch == "DON":
            train_I, train_c0, test_I, test_c0, train_cn, test_cn, trunk_points = model_data
            
            # Run predictions using vectorized model application
            c_test_pred = jax.vmap(model.apply, in_axes=(None, 0, 0, None))(
                params, test_I, test_c0, trunk_points
            )
            
            # Reshape predictions to match ground truth format
            num_samples_I = self.data_config["preprocessing"]["num_samples_I"]
            num_samples_c0 = self.data_config["preprocessing"]["num_samples_c0"]
            c_test_pred_reshaped = c_test_pred.reshape(-1, num_samples_c0, num_samples_I)
            
            print(f"    ▶ {canonical_arch} {electrode} shapes (pred/true): {c_test_pred_reshaped.shape} / {test_cn.shape}")
            return c_test_pred_reshaped, test_cn
            
        else:
            raise ValueError(f"Unknown model architecture: {canonical_arch}")
    
    def calculate_concentration_errors(self, c_pred_anode: np.ndarray, c_true_anode: np.ndarray,
                                     c_pred_cathode: np.ndarray, c_true_cathode: np.ndarray) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, np.ndarray]]]:
        """Calculate concentration prediction errors for both electrodes."""
        # Battery parameters for scaling
        parameter_name = self.data_config["parameter_name"]
        params_bat = pybamm.ParameterValues(parameter_name)
        cs_max_a = params_bat["Maximum concentration in negative electrode [mol.m-3]"]
        cs_max_c = params_bat["Maximum concentration in positive electrode [mol.m-3]"]
        
        # Scale concentrations
        c_pred_anode_scaled = c_pred_anode * cs_max_a
        c_true_anode_scaled = c_true_anode * cs_max_a
        c_pred_cathode_scaled = c_pred_cathode * cs_max_c
        c_true_cathode_scaled = c_true_cathode * cs_max_c
        
        # Calculate concentration errors
        concentration_errors_anode = self.calc_error_metrics(c_pred_anode_scaled, c_true_anode_scaled)
        concentration_errors_cathode = self.calc_error_metrics(c_pred_cathode_scaled, c_true_cathode_scaled)
        concentration_errors_all = self.calc_error_metrics_all(
            concentration_errors_anode, concentration_errors_cathode
        )
        
        # Calculate normalized concentration errors
        concentration_errors_anode_norm = self.calc_error_metrics(c_pred_anode, c_true_anode)
        concentration_errors_cathode_norm = self.calc_error_metrics(c_pred_cathode, c_true_cathode)
        concentration_errors_all_norm = self.calc_error_metrics_all(
            concentration_errors_anode_norm, concentration_errors_cathode_norm
        )
        
        errors_normalized = {
            "anode": concentration_errors_anode_norm,
            "cathode": concentration_errors_cathode_norm,
            "combined": concentration_errors_all_norm
        }
        errors_scaled = {
            "anode": concentration_errors_anode,
            "cathode": concentration_errors_cathode,
            "combined": concentration_errors_all
        }
        return errors_normalized, errors_scaled

    def calculate_voltage_errors(self, c_pred_anode: np.ndarray, c_true_anode: np.ndarray,
                               c_pred_cathode: np.ndarray, c_true_cathode: np.ndarray,
                               test_I: np.ndarray) -> Dict[str, float]:
        """Calculate voltage prediction errors."""
        # Extract surface concentrations (last radial point)
        c_pred_an_surf = c_pred_anode[:, -1, :].squeeze()
        c_true_an_surf = c_true_anode[:, -1, :].squeeze()
        c_pred_ca_surf = c_pred_cathode[:, -1, :].squeeze()
        c_true_ca_surf = c_true_cathode[:, -1, :].squeeze()
        
        # Battery parameters
        parameter_name = self.data_config["parameter_name"]
        params_bat = pybamm.ParameterValues(parameter_name)
        
        # Physical parameters
        Ran = params_bat["Negative particle radius [m]"]
        Rca = params_bat["Positive particle radius [m]"]
        epsan = params_bat["Negative electrode active material volume fraction"]
        epsca = params_bat["Positive electrode active material volume fraction"]
        Lan = params_bat["Negative electrode thickness [m]"]
        Lca = params_bat["Positive electrode thickness [m]"]
        A = params_bat["Electrode height [m]"] * params_bat["Electrode width [m]"]
        
        # Calculate voltages using post-processing function
        V_pred, V_true = self.functions.post_proc(
            params_bat, test_I, c_pred_an_surf, c_true_an_surf, 
            c_pred_ca_surf, c_true_ca_surf, Ran, Rca, epsan, epsca, Lan, Lca, A
        )
        
        # Calculate voltage errors
        voltage_errors = self.calc_error_metrics(V_pred, V_true, axis=(1,))
        
        return voltage_errors
    
    def analyze_model(self, model_architecture: str, 
                     anode_model_path: Optional[str] = None,
                     cathode_model_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Run complete error analysis for a model architecture.
        
        Args:
            model_architecture: Architecture name (FNO, CAPE_FNO2, DON)
            anode_model_path: Optional path to anode model (uses latest if None)
            cathode_model_path: Optional path to cathode model (uses latest if None)
            
        Returns:
            Dictionary containing all error metrics
        """
        # Get current profile(s) from data config
        families = self.data_config.get('family', None)
        
        # Handle both single family (string) and multiple families (list)
        if isinstance(families, str):
            families = [families]
        
        if len(families) == 1:
            # Single profile analysis
            return self._analyze_single_profile(
                model_architecture, families[0], anode_model_path, cathode_model_path
            )
        else:
            # Multiple profiles - analyze each separately and combine results
            print(f"🔍 Analyzing {model_architecture} model for {families} current profiles...")
            print("📊 Running separate analysis for each profile to match trained models with corresponding datasets...")
            
            profile_results = {}
            combined_concentration_errors_normalized = {"anode": {}, "cathode": {}, "combined": {}}
            combined_concentration_errors_scaled = {"anode": {}, "cathode": {}, "combined": {}}
            combined_voltage_errors = {}
            
            for profile in families:
                print(f"\n🔬 Analyzing profile: {profile}")
                
                try:
                    # Run analysis for this specific profile
                    result = self._analyze_single_profile(
                        model_architecture, profile, None, None  # Let it find profile-specific models
                    )
                    profile_results[profile] = result
                    
                    # Accumulate results for combined metrics (normalized and scaled)
                    for electrode in ["anode", "cathode", "combined"]:
                        if electrode not in combined_concentration_errors_normalized:
                            combined_concentration_errors_normalized[electrode] = {}
                        if electrode not in combined_concentration_errors_scaled:
                            combined_concentration_errors_scaled[electrode] = {}

                        conc_errors_norm = result["concentration_errors_normalized"][electrode]
                        for metric, values in conc_errors_norm.items():
                            if metric not in combined_concentration_errors_normalized[electrode]:
                                combined_concentration_errors_normalized[electrode][metric] = []
                            combined_concentration_errors_normalized[electrode][metric].extend(values)

                        conc_errors_scaled = result["concentration_errors"][electrode]
                        for metric, values in conc_errors_scaled.items():
                            if metric not in combined_concentration_errors_scaled[electrode]:
                                combined_concentration_errors_scaled[electrode][metric] = []
                            combined_concentration_errors_scaled[electrode][metric].extend(values)
                    
                    # Accumulate voltage errors
                    volt_errors = result["voltage_errors"]
                    for metric, values in volt_errors.items():
                        if metric not in combined_voltage_errors:
                            combined_voltage_errors[metric] = []
                        combined_voltage_errors[metric].extend(values)
                        
                except Exception as e:
                    print(f"⚠️  Warning: Failed to analyze profile {profile}: {str(e)}")
                    continue
            
            # Convert accumulated lists to numpy arrays
            for electrode in combined_concentration_errors_normalized:
                for metric in combined_concentration_errors_normalized[electrode]:
                    combined_concentration_errors_normalized[electrode][metric] = np.array(
                        combined_concentration_errors_normalized[electrode][metric]
                    )

            for electrode in combined_concentration_errors_scaled:
                for metric in combined_concentration_errors_scaled[electrode]:
                    combined_concentration_errors_scaled[electrode][metric] = np.array(
                        combined_concentration_errors_scaled[electrode][metric]
                    )
            
            for metric in combined_voltage_errors:
                combined_voltage_errors[metric] = np.array(combined_voltage_errors[metric])
            
            # Return combined results
            return {
                "model_architecture": model_architecture,
                "families": families,
                "profile_results": profile_results,
                "concentration_errors_normalized": combined_concentration_errors_normalized,
                "concentration_errors": combined_concentration_errors_scaled,
                "voltage_errors": combined_voltage_errors
            }

    def _analyze_single_profile(self, model_architecture: str, profile: str,
                               anode_model_path: Optional[str] = None,
                               cathode_model_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Run error analysis for a single current profile.
        
        Args:
            model_architecture: Architecture name (FNO, CAPE_FNO2, DON)  
            profile: Current profile name (CC, Triangle, PLS, GRF)
            anode_model_path: Optional path to anode model (uses latest if None)
            cathode_model_path: Optional path to cathode model (uses latest if None)
            
        Returns:
            Dictionary containing error metrics for this profile
        """
        print(f"🔍 Analyzing {model_architecture} model for {profile} current profile...")
        
        # Temporarily set family to single profile for dataset loading
        original_family = self.data_config['family']
        self.data_config['family'] = profile
        
        try:
            # Load dataset for this specific profile
            train_data, test_data = self.load_dataset()
            train_data, test_data = self.filter_data(train_data, test_data)
            
            # Find model paths if not provided - look for models trained on this specific profile
            if anode_model_path is None:
                anode_model_path = self.find_latest_model(model_architecture, "anode", profile)
                if anode_model_path is None:
                    raise FileNotFoundError(f"No anode model found for {model_architecture} with profile {profile}")
                    
            if cathode_model_path is None:
                cathode_model_path = self.find_latest_model(model_architecture, "cathode", profile)
                if cathode_model_path is None:
                    raise FileNotFoundError(f"No cathode model found for {model_architecture} with profile {profile}")
            
            print(f"📁 Using anode model for {profile}: {anode_model_path}")
            print(f"📁 Using cathode model for {profile}: {cathode_model_path}")
            
            # Load models
            if model_architecture == "DON":
                # Build models with autodetected basis sizes separately per electrode
                preprocessing = self.data_config["preprocessing"]
                num_samples_I = preprocessing["num_samples_I"]
                num_samples_c0 = preprocessing["num_samples_c0"]
                # Reuse the autodetection logic to find matching model structure
                def build_don(model_path: str):
                    init_key = jax.random.PRNGKey(0)
                    dummy_I = jax.random.normal(init_key, (num_samples_I,))
                    dummy_c0 = jax.random.normal(init_key, (num_samples_c0,))
                    dummy_trunk = jax.random.normal(init_key, (num_samples_I * num_samples_c0, 2))
                    param_bytes = self.functions.load_model_params(model_path)
                    candidate_basis = [self.model_config["don"].get("amount_basis", 16), 16, 32, 64, 128]
                    tried = set()
                    last_err = None
                    for M in candidate_basis:
                        if M in tried:
                            continue
                        tried.add(M)
                        try:
                            branch_layers = [self.model_config["don"]["width"]] * self.model_config["don"]["depth"] + [M]
                            trunk_layers = list(branch_layers)
                            from ..models.deeponet import DeepONet as DONModel
                            don_model = DONModel(branch_layers=branch_layers, trunk_layers=trunk_layers)
                            dummy_params = don_model.init(init_key, dummy_I, dummy_c0, dummy_trunk)
                            params = flax.serialization.from_bytes(dummy_params, param_bytes)
                            _ = don_model.apply(params, dummy_I, dummy_c0, dummy_trunk)
                            return don_model, params
                        except Exception as e:
                            last_err = e
                            continue
                    raise RuntimeError(f"Failed to build DON model from {model_path} with autodetect: {last_err}")

                anode_model, anode_params = build_don(anode_model_path)
                cathode_model, cathode_params = build_don(cathode_model_path)
            else:
                anode_model = self.create_model(model_architecture)
                cathode_model = self.create_model(model_architecture)
                anode_params = self.load_model_params(anode_model_path, model_architecture)
                cathode_params = self.load_model_params(cathode_model_path, model_architecture)
            
            # Preprocess data for both electrodes
            anode_data = self.preprocess_model_data(train_data, test_data, model_architecture, "anode")
            cathode_data = self.preprocess_model_data(train_data, test_data, model_architecture, "cathode")
            
            # Run predictions
            c_pred_anode, c_true_anode = self.run_predictions(
                anode_model, anode_params, model_architecture, anode_data, "anode"
            )
            
            c_pred_cathode, c_true_cathode = self.run_predictions(
                cathode_model, cathode_params, model_architecture, cathode_data, "cathode"
            )
            
            # Calculate concentration errors
            concentration_errors_normalized, concentration_errors_scaled = self.calculate_concentration_errors(
                c_pred_anode, c_true_anode, c_pred_cathode, c_true_cathode
            )
            
            # Calculate voltage errors
            voltage_errors = self.calculate_voltage_errors(
                c_pred_anode, c_true_anode, c_pred_cathode, c_true_cathode, test_data["current"]
            )
            
            print(f"✅ Analysis complete for {model_architecture} on {profile}")
            
            return {
                "model_architecture": model_architecture,
                "current_profile": profile,
                "concentration_errors_normalized": concentration_errors_normalized,
                "concentration_errors": concentration_errors_scaled,
                "voltage_errors": voltage_errors
            }
            
        finally:
            # Restore original family configuration
            self.data_config['family'] = original_family 