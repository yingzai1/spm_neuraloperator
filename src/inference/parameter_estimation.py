"""Fast parameter estimation with a pretrained CAPE-FNO model."""

from dataclasses import dataclass
from functools import partial
import jax
import jax.numpy as jnp
import pybamm
from flax import serialization

from models.FNO import CAPE_FNO
from util.FNO_util import remove_padding
import util.functions as functions


def load_cape_fno(anode_file: str, cathode_file: str,
                   k_modes: int = 10,
                   fno_depth: int = 8,
                   cape_hidden_size: int = 32,
                   hidden_channels: int = 64,
                   input_channels: int = 4,
                   output_channels: int = 1,
                   rng_key: jax.Array | None = None):
    """Load pretrained CAPE-FNO models for anode and cathode."""
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)
    model = CAPE_FNO(
        k_modes=k_modes,
        fno_depth=fno_depth,
        cape_hidden_size=cape_hidden_size,
        hidden_channels=hidden_channels,
        input_channels=input_channels,
        output_channels=output_channels,
    )
    dummy_x = jnp.zeros((1, 24, 85, input_channels))
    dummy_d = jnp.zeros((1, 1))
    params = model.init(rng_key, dummy_x, dummy_d)
    with open(anode_file, "rb") as f:
        params_anode = serialization.from_bytes(params, f.read())
    with open(cathode_file, "rb") as f:
        params_cathode = serialization.from_bytes(params, f.read())
    return model, params_anode, params_cathode


@dataclass
class BatteryConstants:
    """Collection of constants required for voltage computation."""

    C: float
    Ran: float
    Rca: float
    epsan: float
    epsca: float
    Lan: float
    Lca: float
    A: float
    R_gas: float
    F: float
    Temp: float


def load_pybamm_params(parameter_name: str = "Prada2013") -> tuple[pybamm.ParameterValues, BatteryConstants]:
    """Return PyBaMM parameters and commonly used constants."""
    params = pybamm.ParameterValues(parameter_name)
    const = BatteryConstants(
        C=params["Nominal cell capacity [A.h]"],
        Ran=params["Negative particle radius [m]"],
        Rca=params["Positive particle radius [m]"],
        epsan=params["Negative electrode active material volume fraction"],
        epsca=params["Positive electrode active material volume fraction"],
        Lan=params["Negative electrode thickness [m]"],
        Lca=params["Positive electrode thickness [m]"],
        A=params["Electrode height [m]"] * params["Electrode width [m]"],
        R_gas=params["Ideal gas constant [J.K-1.mol-1]"],
        F=params["Faraday constant [C.mol-1]"],
        Temp=params["Ambient temperature [K]"],
    )
    return params, const


# ---- OCP helper functions ----------------------------------------------------

def U_OCP_an(sto: jnp.ndarray) -> jnp.ndarray:
    """Open circuit potential for the anode."""
    return (
        1.9793 * jnp.exp(-39.3631 * sto)
        + 0.2482
        - 0.0909 * jnp.tanh(29.8538 * (sto - 0.1234))
        - 0.04478 * jnp.tanh(14.9159 * (sto - 0.2769))
        - 0.0205 * jnp.tanh(30.4444 * (sto - 0.6103))
    )


def U_OCP_ca(sto: jnp.ndarray) -> jnp.ndarray:
    """Open circuit potential for the cathode."""
    c1 = -150 * sto
    c2 = -30 * (1 - sto)
    return 3.4077 - 0.020269 * sto + 0.5 * jnp.exp(c1) - 0.9 * jnp.exp(c2)


# ---- estimator --------------------------------------------------------------

class CAPEFNOEstimator:
    """Voltage prediction and parameter estimation using CAPE-FNO."""

    def __init__(self, model: CAPE_FNO, params_anode, params_cathode,
                 const: BatteryConstants, padding_r: int = 2, padding_t: int = 5):
        self.model = model
        self.params_anode = params_anode
        self.params_cathode = params_cathode
        self.const = const
        self.padding_r = padding_r
        self.padding_t = padding_t

    @partial(jax.jit, static_argnums=0)
    def _predict_conc(self, X_anode: jnp.ndarray, X_cathode: jnp.ndarray,
                       Dan: jnp.ndarray, Dca: jnp.ndarray):
        cn_anode = self.model.apply(self.params_anode, X_anode, Dan)
        cn_cathode = self.model.apply(self.params_cathode, X_cathode, Dca)
        cn_anode = remove_padding(cn_anode, self.padding_r, self.padding_t)
        cn_cathode = remove_padding(cn_cathode, self.padding_r, self.padding_t)
        cn_anode_surf = cn_anode[:, -1, :].squeeze()
        cn_cathode_surf = cn_cathode[:, -1, :].squeeze()
        eps = 1e-12
        cn_anode_surf = jnp.clip(cn_anode_surf, eps, 1.0 - eps)
        cn_cathode_surf = jnp.clip(cn_cathode_surf, eps, 1.0 - eps)
        return cn_anode_surf, cn_cathode_surf

    @partial(jax.jit, static_argnums=0)
    def _voltage(self, func_I: jnp.ndarray,
                 c_an: jnp.ndarray, c_ca: jnp.ndarray) -> jnp.ndarray:
        Ran = self.const.Ran
        Rca = self.const.Rca
        epsan = self.const.epsan
        epsca = self.const.epsca
        Lan = self.const.Lan
        Lca = self.const.Lca
        A = self.const.A
        R_gas = self.const.R_gas
        Temp = self.const.Temp
        F = self.const.F

        j_anode = jnp.sqrt(c_an * (1.0 - c_an))
        j_cathode = jnp.sqrt(c_ca * (1.0 - c_ca))
        xan = functions.in_arcsinh(-func_I, Ran, epsan, Lan, A)
        xca = functions.in_arcsinh(-func_I, Rca, epsca, Lca, A)
        return (
            U_OCP_ca(c_ca)
            - U_OCP_an(c_an)
            - 2 * R_gas * Temp / F * jnp.arcsinh(0.5 * xan / j_anode)
            - 2 * R_gas * Temp / F * jnp.arcsinh(0.5 * xca / j_cathode)
        )

    @partial(jax.jit, static_argnums=0)
    def predict_voltage(self, func_I: jnp.ndarray,
                        X_anode: jnp.ndarray,
                        X_cathode: jnp.ndarray,
                        Dan: jnp.ndarray,
                        Dca: jnp.ndarray) -> jnp.ndarray:
        """Predict voltage for batches of candidate diffusivities."""
        func_I = jnp.asarray(func_I)
        X_anode = jnp.asarray(X_anode)
        X_cathode = jnp.asarray(X_cathode)
        Dan = jnp.asarray(Dan)
        Dca = jnp.asarray(Dca)

        def single(dan, dca):
            c_an, c_ca = self._predict_conc(X_anode, X_cathode, dan[None, :], dca[None, :])
            return self._voltage(func_I, c_an, c_ca)

        return jax.vmap(single)(Dan, Dca)


__all__ = [
    "load_cape_fno",
    "load_pybamm_params",
    "BatteryConstants",
    "CAPEFNOEstimator",
]
