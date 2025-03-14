import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy.stats import qmc

def GaussianRFCurrent(seed, value, t_max):
    """
    Generates a Gaussian random field current function.
    
    Parameters:
      seed  : integer seed for reproducibility
      value : scaling factor for the current
      t_max : maximum time value
    
    Returns:
      A function f(t) that interpolates the generated field.
    """
    # Set up sample points for the field
    t_samples = np.linspace(0, t_max, 75)
    L = 1.0
    n = t_samples.shape[0]
    t_i = t_samples.reshape(-1, 1)
    t_j = t_samples.reshape(1, -1)

    # Compute periodic distance
    delta_t = np.pi * (t_i - t_j) / t_max
    sin_term = np.sin(delta_t)

    # Periodic covariance function
    cov = np.exp(-2 * (sin_term / L) ** 2)

    # Add small jitter to ensure positive definiteness
    jitter = 1e-6 * np.eye(n)
    cov += jitter

    mean = np.zeros(n)
    
    # Use NumPy's random generator
    np.random.seed(seed)
    field = np.random.multivariate_normal(mean, cov)
    
    # Truncate field values to within [-1.5, 1.5]
    field = np.clip(field, -1.5, 1.5)
    
    def f(t):
        t_np = np.array(t, ndmin=1)
        interpolated = np.interp(t_np, t_samples, field) * value
        if np.isscalar(t):
            return float(interpolated[0])
        return interpolated

    return f

# -------------------------------
# Other Current Functions (using NumPy)
# -------------------------------

def ConstantCurrent(value):
    def f(t):
        return np.ones_like(t) * value
    return f

def TriangleCurrent(value):
    def f(t):
        t1 = 900
        t2 = 1800
        return np.where(t <= t1, t / t1, (t2 - t) / t1) * value
    return f


# def GaussianRFCurrent(key, value, t_max):
#     # Set up sample points for the field
#     t_samples = np.linspace(0, t_max, 75)

#     L = 1.0
#     n = t_samples.shape[0]
#     t_i = t_samples.reshape(-1, 1)
#     t_j = t_samples.reshape(1, -1)

#     # Compute periodic distance
#     delta_t = np.pi * (t_i - t_j) / t_max
#     sin_term = np.sin(delta_t)

#     # Periodic covariance function
#     cov = np.exp(-2 * (sin_term / L) ** 2)

#     # Add small jitter to ensure positive definiteness
#     jitter = 1e-6 * np.eye(n)
#     cov += jitter

#     mean = np.zeros(n)

#     seed = int(jax.random.randint(key, shape=(), minval=0, maxval=2**31-1))
#     np.random.seed(seed)

#     # Generate the Gaussian random field samples at the chosen sample points
#     field = np.random.multivariate_normal(mean, cov)

#     # Truncate field values to within [-1.5, 1.5] as per your example
#     field = np.clip(field, -1.5, 1.5)

#     def f(t):
#         # Convert input t to a numpy array for interpolation
#         t_np = np.array(t, ndmin=1)  # Ensures we have at least 1D array
#         # Interpolate the field values at the given time points
#         interpolated = np.interp(t_np, t_samples, field) * value

#         # If the original t was scalar, return a scalar
#         if np.isscalar(t):
#             return float(interpolated[0])
#         return interpolated

#     return f


# def ConstantCurrent(value):

#     def f(t):
#         return jnp.ones_like(t) * value
#     return f

# def TriangleCurrent(value):

#     def f(t):

#         t1 = 900
#         t2 = 1800

#         return jnp.where(t <= t1, t / t1, (t2 - t) / t1) * value

#     return f

# def func(k, value):

#     def f(t):

#         tmax = t.max()

#         return jnp.tanh(k * (t - tmax/2) / (tmax/2)) * value

#     return f


def gen_data(m_inner, m_init, m_bc0, m_bcR, epsilon=1e-6, sort_axis = None, ub_time = 1):

    sampler_res = qmc.Sobol(d=2, scramble=False)
    sampler_bc = qmc.Sobol(d=1, scramble=False)

    X_inner = sampler_res.random_base2(m=m_inner)
    X_inner = qmc.scale(X_inner, [epsilon, epsilon], [ub_time-epsilon, 1-epsilon])
    if sort_axis is not None:
        indices = np.argsort(X_inner[:, sort_axis])
        X_inner = X_inner[indices]

    r_initial = sampler_bc.random_base2(m=m_init)
    t_initial = np.zeros_like(r_initial)
    X_initial = np.concatenate((t_initial, r_initial), axis=1)

    sampler_bc.reset()
    t_boundary0 = sampler_bc.random_base2(m=m_bc0)
    t_boundary0 = qmc.scale(t_boundary0, epsilon, ub_time)
    r_boundary0 = np.zeros_like(t_boundary0)
    X_boundary_0 = np.concatenate((t_boundary0, r_boundary0), axis=1)
    if sort_axis is not None:
        indices = np.argsort(X_boundary_0[:, sort_axis])
        X_boundary_0 = X_boundary_0[indices]

    sampler_bc.reset()
    t_boundaryR = sampler_bc.random_base2(m=m_bcR)
    t_boundaryR = qmc.scale(t_boundaryR, epsilon, ub_time)
    r_boundaryR = np.ones_like(t_boundaryR)
    X_boundary_R = np.concatenate((t_boundaryR, r_boundaryR), axis=1)
    if sort_axis is not None:
        indices = np.argsort(X_boundary_R[:, sort_axis])
        X_boundary_R = X_boundary_R[indices]

    return X_inner, X_initial, X_boundary_0, X_boundary_R


def plot_col_points(X_inner, X_initial, X_boundary_0, X_boundary_R):

    # Create the scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(X_initial[:,0], X_initial[:,1], color='blue', label='Initial Points')
    plt.scatter(X_inner[:,0], X_inner[:,1], color='green', label='Inner Points')
    plt.scatter(X_boundary_0[:,0], X_boundary_0[:,1], color='red', label='Boundary 0 Points')
    plt.scatter(X_boundary_R[:,0], X_boundary_R[:,1], color='purple', label='Boundary R Points')

    # Adding labels and legend
    plt.xlabel('t-axis Label')  # Update with your specific label
    plt.ylabel('r-axis Label')  # Update with your specific label
    plt.title('Collocation Points')  # Optional: add a title if desired
    plt.legend()

    # Show the plot
    plt.grid(True)  # Optional: add a grid for easier visualization
    plt.show()


# my_functions.py
import pybamm

def simulate_single(I_func,t, soc=0.5, params = pybamm.ParameterValues("Chen2020")):
    spm = pybamm.lithium_ion.SPM()
    sim = pybamm.Simulation(spm, parameter_values=params)
    sim.parameter_values["Current function [A]"] = pybamm.Interpolant(t, -1.0 * I_func, pybamm.t)
    sol = sim.solve(initial_soc=soc, t_eval=t)
    c0 = sol["Positive particle concentration"].entries[:, 0, 0]
    cn_target = sol["Negative particle concentration"].entries[:, 0, :]
    return cn_target, c0


import os
import flax.serialization
from datetime import datetime

def save_model_params(params, directory="./trained_models", prefix = "anode", family = "CC"):
    """Serialize and save model params with a timestamped filename."""
    os.makedirs(directory, exist_ok=True)
    # Create a unique timestamp string, e.g. "2025-03-06_14-20-59"
    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Construct filename, e.g. "./checkpoints/model_params_2025-03-06_14-20-59.msgpack"
    filename = os.path.join(directory, f"{prefix}_{family}_{timestamp_str}.msgpack")

    # Convert params PyTree to a bytes object
    param_bytes = flax.serialization.to_bytes(params)

    # Write to disk
    with open(filename, "wb") as f:
        f.write(param_bytes)

    print(f"Saved model parameters to {filename}")
    return filename

def load_model_params(filename):
    # Read the saved bytes
    with open(filename, "rb") as f:
        param_bytes = f.read()

    return param_bytes

# Usage:
# model = FNO(...)  # or whatever your model class
# params = load_model_params(model, "./checkpoints/model_params_2025-03-06_14-20-59.msgpack")


def post_proc(params, I_c, c_pred_an, c_true_an, c_pred_ca, c_true_ca, Ran, Rca, epsan, epsca, Lan, Lca, A):
    #params = pybamm.ParameterValues("Ecker2015") #OKane und Chen teilen sich SPM params
    U_OCP_an = params["Negative electrode OCP [V]"]
    U_OCP_ca = params["Positive electrode OCP [V]"]
    R = params['Ideal gas constant [J.K-1.mol-1]']
    F = params['Faraday constant [C.mol-1]']
    T = params["Ambient temperature [K]"]

    def in_arcsinh(I, R, epsilon, L, A):

        x = I * R / (3 * epsilon * L * A)

        return x
    
    #sim_length = c_true_an.shape[0]

    j_pred_an = c_pred_an**0.5 * (1-c_pred_an)**0.5
    j_true_an = c_true_an**0.5 * (1-c_true_an)**0.5

    j_pred_ca = c_pred_ca**0.5 * (1-c_pred_ca)**0.5
    j_true_ca = c_true_ca**0.5 * (1-c_true_ca)**0.5

    xan = in_arcsinh(-I_c, Ran, epsan, Lan, A)
    xca = in_arcsinh(-I_c, Rca, epsca, Lca, A)

    V_pred = U_OCP_ca(c_pred_ca) - U_OCP_an(c_pred_an) - 2 * R*T/F * jnp.arcsinh(0.5*xan/(j_pred_an)) - 2 * R*T/F * jnp.arcsinh(0.5*xca/(j_pred_ca))
    V_true = U_OCP_ca(c_true_ca) - U_OCP_an(c_true_an) - 2 * R*T/F * jnp.arcsinh(0.5*xan/(j_true_an)) - 2 * R*T/F * jnp.arcsinh(0.5*xca/(j_true_ca))

    return V_pred, V_true

def post_proc2(params, I_c, c_pred_an, c_true_an, c_pred_ca, c_true_ca, Ran, Rca, epsan, epsca, Lan, Lca, A):
    #params = pybamm.ParameterValues("Ecker2015") #OKane und Chen teilen sich SPM params
    U_OCP_an = params["Negative electrode OCP [V]"]
    U_OCP_ca = params["Positive electrode OCP [V]"]
    R = params['Ideal gas constant [J.K-1.mol-1]']
    F = params['Faraday constant [C.mol-1]']
    T = params["Ambient temperature [K]"]

    # k_an = params["Negative electrode reaction rate constant [m.s-1]"]
    # k_ca = params["Positive electrode reaction rate constant [m.s-1]"]

    m_an = 6.48e-7
    m_ca  = 3.42e-6
    c_e = 1000

    def in_arcsinh(I, Ri, epsilon, L, A):

        x = I * Ri / (3 * epsilon * L * A)

        return x
    
    #sim_length = c_true_an.shape[0]

    c_an_max = params["Maximum concentration in negative electrode [mol.m-3]"]
    c_ca_max = params["Maximum concentration in positive electrode [mol.m-3]"]
    c_e = 1000

    j_pred_an = m_an * jnp.sqrt(c_pred_an * (1-c_pred_an))/jnp.sqrt(c_an_max) * jnp.sqrt(c_e)
    j_true_an = m_an * jnp.sqrt(c_true_an * (1-c_true_an))/jnp.sqrt(c_an_max) * jnp.sqrt(c_e)

    j_pred_ca = m_ca * jnp.sqrt(c_pred_ca * (1-c_pred_ca))/jnp.sqrt(c_ca_max) * jnp.sqrt(c_e)
    j_true_ca = m_ca * jnp.sqrt(c_true_ca * (1-c_true_ca))/jnp.sqrt(c_ca_max) * jnp.sqrt(c_e)

    xan = in_arcsinh(-I_c, Ran, epsan, Lan, A)
    xca = in_arcsinh(-I_c, Rca, epsca, Lca, A)

    V_pred = U_OCP_ca(c_pred_ca) - U_OCP_an(c_pred_an) - 2 * R*T/F * jnp.arcsinh(0.5*xan/(j_pred_an)) - 2 * R*T/F * jnp.arcsinh(0.5*xca/(j_pred_ca))
    V_true = U_OCP_ca(c_true_ca) - U_OCP_an(c_true_an) - 2 * R*T/F * jnp.arcsinh(0.5*xan/(j_true_an)) - 2 * R*T/F * jnp.arcsinh(0.5*xca/(j_true_ca))

    return V_pred, V_true