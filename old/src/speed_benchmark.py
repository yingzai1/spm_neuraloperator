#!/usr/bin/env python
# coding: utf-8

# In[24]:


import numpy as np
import jax
import jax.numpy as jnp
import util.functions as functions
from models.FNO import CAPE_FNO, FNO
from models.DON import DeepONet, generate_trunk_points
import pybamm
import flax


# In[25]:


from util.FNO_util import remove_padding, normalise_diffusion
# from util.plotting import create_plot_FNO_results, create_plot_voltage
from util.postprocess import filter_anode_cathode
from inference.parameter_estimation import get_jnp_U_OCP_anode, get_jnp_U_OCP_cathode


# In[26]:


parameter_name = "Prada2013"


# In[27]:


family = "CC"
N_total = 11000
data = np.load(f"../data/{parameter_name}_{family}_{N_total}.npz")
random_seed = 42


# In[28]:


params_bat = pybamm.ParameterValues(parameter_name)
U_OCP_an = get_jnp_U_OCP_anode(parameter_name)
U_OCP_ca = get_jnp_U_OCP_cathode(parameter_name)


C = params_bat["Nominal cell capacity [A.h]"]
Dan = params_bat["Negative particle diffusivity [m2.s-1]"]
Dca = params_bat["Positive particle diffusivity [m2.s-1]"]
Ran = params_bat["Negative particle radius [m]"]
Rca = params_bat["Positive particle radius [m]"]
epsan = params_bat["Negative electrode active material volume fraction"]
epsca = params_bat["Positive electrode active material volume fraction"]
cs_max_a = params_bat["Maximum concentration in negative electrode [mol.m-3]"]
cs_max_c = params_bat["Maximum concentration in positive electrode [mol.m-3]"]
Lan = params_bat["Negative electrode thickness [m]"]
Lca = params_bat["Positive electrode thickness [m]"]
A = params_bat["Electrode height [m]"] * params_bat["Electrode width [m]"]

EPS = 1e-12
t_max = 3600


# In[29]:


# Here we only look at speed so we dont split the data


# In[30]:


func_I = jnp.array(data["current"])

### Anode data ###
cn_anode = jnp.array(data["cn_anode"])
c0_anode = jnp.array(data["c0_anode"])
D_anode = jnp.array(data["Dan"])

### Cathode data ###
cn_cathode = jnp.array(data["cn_cathode"])
c0_cathode = jnp.array(data["c0_cathode"])
D_cathode = jnp.array(data["Dca"])

soc = jnp.array(data["soc"])


# In[31]:


D_anode = normalise_diffusion(D_anode)
D_cathode = normalise_diffusion(D_cathode)


# In[32]:


parameter_name = "Prada2013"
params_bat = pybamm.ParameterValues(parameter_name)
cs_max_a_norm = 1 #params_bat["Maximum concentration in negative electrode [mol.m-3]"]
cs_max_c_norm = 1 #params_bat["Maximum concentration in positive electrode [mol.m-3]"]
cs_min_a_norm = 0.0
cs_min_c_norm = 0.0


# In[33]:


cn_anode, cn_cathode, mask = filter_anode_cathode(cn_anode, cn_cathode,
                                                    anode_lo=cs_min_a_norm, anode_hi=cs_max_a_norm, 
                                                    cathode_lo=cs_min_c_norm, cathode_hi=cs_max_c_norm)


# In[34]:


func_I = func_I[mask]
c0_anode = c0_anode[mask]
c0_cathode = c0_cathode[mask]
D_anode = D_anode[mask]
D_cathode = D_cathode[mask]
soc = soc[mask]


# In[35]:


soc.shape


# In[36]:


# def filter_for_soc(cn_anode, cn_cathode, func_I, c0_anode, c0_cathode, D_anode, D_cathode, soc, soc_value=0.2):
#     mask = (np.isclose(soc, soc_value))
#     return (cn_anode[mask], cn_cathode[mask], func_I[mask], c0_anode[mask], c0_cathode[mask], D_anode[mask], D_cathode[mask])


# In[37]:


# initial_soc = 0.3
# cn_anode_soc, cn_cathode_soc, func_I_soc, c0_anode_soc, c0_cathode_soc, D_anode_soc, D_cathode_soc = filter_for_soc(
#     cn_anode, cn_cathode, func_I, c0_anode, c0_cathode, D_anode, D_cathode, soc, initial_soc)


# In[38]:


padding_r = 2
padding_t = 5


# In[39]:


def preprocess_data_fast(train_I, train_c0):
    """
    Pre-process the dataset into channels and padded shapes suitable for the FNO.

    Parameters
    ----------
    train_I  : array (N, 75)      – current (time) history
    train_c0 : array (N, 20)      – initial concentration
    train_cn : array (N, 20, 75)  – target concentration
    padding_r, padding_t          – radial / temporal padding

    Returns
    -------
    X : array (N, 24, 85, 4)  – 4-channel input tensor
    Y : array (N, 24, 85, 1)  – 1-channel target tensor
    """
    # ---------- constants ----------
    N, W_orig   = train_I.shape          # 75
    H_orig      = train_c0.shape[1]      # 20
    H, W        = H_orig + 2*padding_r, W_orig + 2*padding_t   # 24, 85

    # ---------- coordinate grid (shared by all samples) ----------
    t = jnp.linspace(0.0, 1.0, W_orig)
    r = jnp.linspace(0.0, 1.0, H_orig)
    R, T = jnp.meshgrid(r, t, indexing='ij')                   # (20, 75)
    R = jnp.pad(R, ((padding_r, padding_r), (padding_t, padding_t)))
    T = jnp.pad(T, ((padding_r, padding_r), (padding_t, padding_t)))
    # broadcast cheaply – no data copy!
    R = jnp.broadcast_to(R, (N, H, W))
    T = jnp.broadcast_to(T, (N, H, W))

    # ---------- pad & broadcast the inputs ----------
    I_pad  = jnp.pad(train_I,  ((0, 0), (padding_t, padding_t)))          # (N, 85)
    c0_pad = jnp.pad(train_c0, ((0, 0), (padding_r, padding_r)))          # (N, 24)

    I_2D  = jnp.broadcast_to(I_pad[:,  None, :], (N, H, W))               # (N,24,85)
    c0_2D = jnp.broadcast_to(c0_pad[:, :, None], (N, H, W))               # (N,24,85)

    # ---------- assemble channels ----------
    X = jnp.stack((I_2D, c0_2D, R, T), axis=-1)                           # (N,24,85,4)

    return X.astype(jnp.float32)


# In[40]:


# Original sample counts
num_samples_I = 75
num_samples_c0 = 20


# In[41]:


X_anode = preprocess_data_fast(func_I, c0_anode)
X_cathode = preprocess_data_fast(func_I, c0_cathode)


# In[42]:


# Assume these hyperparameters
k_modes = 10
fno_depth = 8
hidden_channels = 16
input_channels = 4 # We should automate this TODO
output_channels = 1
cape_hidden_size = 32


# In[43]:


model = CAPE_FNO(k_modes=k_modes, input_channels= input_channels, 
                 fno_depth=fno_depth, cape_hidden_size = cape_hidden_size, 
                 hidden_channels=hidden_channels, output_channels=output_channels)

main_key = jax.random.PRNGKey(random_seed)
# Initialize parameters
dummy_D = jax.random.normal(main_key, (1,1))
params = model.init(main_key, X_anode[:1,...], dummy_D)

# Forward pass
out = model.apply(params, X_anode[:1,...], dummy_D)


# In[58]:


width = 500
depth = 11
amount_basis = 64
num_samples_I = 75
num_samples_c0 = 20
t_max = 3600
K = num_samples_I * num_samples_c0
t = np.linspace(0, t_max, num_samples_I)
r = np.linspace(0, 1, num_samples_c0)


# In[59]:


trunk_points = generate_trunk_points(r, t/t_max)


# In[60]:


# Setup model hyperparameters
branch_layers = [width] * depth + [amount_basis]  # M = 200 at the end
trunk_layers = branch_layers

model_don = DeepONet(branch_layers=branch_layers, trunk_layers=trunk_layers)

# Create dummy inputs
key1, key2 = jax.random.split(jax.random.PRNGKey(random_seed))
dummy_I = jax.random.normal(key1, (num_samples_I,))
dummy_c0 = jax.random.normal(key1, (num_samples_c0,))      # 75 current samples + 10 initial conditions
dummy_trunk_input = jax.random.normal(key2, (num_samples_I*num_samples_c0, 2))   # K=750 points in the (t,r)-space

# Initialize parameters
params_don = model_don.init(jax.random.PRNGKey(42), dummy_I, dummy_c0, dummy_trunk_input)

# Forward pass
output = model_don.apply(params_don, dummy_I, dummy_c0, dummy_trunk_input)

print("Output shape:", output.shape)


# In[46]:


anode_file_don = "../trained_models/DON/anode_Prada2013_CC_2200_2025-06-19_15-52-50.msgpack"
cathode_file_don = "../trained_models/DON/cathode_Prada2013_CC_2200_2025-06-19_15-54-05.msgpack"

params_anode_don = functions.load_model_params(anode_file_don)
params_cathode_don = functions.load_model_params(cathode_file_don)

params_anode_don = flax.serialization.from_bytes(params_don, params_anode_don)
params_cathode_don = flax.serialization.from_bytes(params_don, params_cathode_don)


# In[47]:


# Assume these hyperparameters
k_modes = 10
fno_depth = 6
hidden_channels = 32
input_channels = X_anode.shape[-1]  # should be 4
output_channels = 1


# In[48]:


model_fno = FNO(k_modes=k_modes, input_channels= input_channels, 
                 fno_depth=fno_depth, 
                 hidden_channels=hidden_channels, output_channels=output_channels)

main_key = jax.random.PRNGKey(random_seed)
# Initialize parameters
dummy_D = jax.random.normal(main_key, (1,1))
params_fno = model_fno.init(main_key, X_anode[:1,...])

# Forward pass
out = model_fno.apply(params_fno, X_anode[:1,...])


# In[49]:


anode_file = "../trained_models/cape_fno/anode_CC_2025-06-10_17-57-42.msgpack"
cathode_file = "../trained_models/cape_fno/cathode_CC_2025-06-10_18-21-51.msgpack"

params_anode = functions.load_model_params(anode_file)
params_cathode = functions.load_model_params(cathode_file)

params_anode = flax.serialization.from_bytes(params, params_anode)
params_cathode = flax.serialization.from_bytes(params, params_cathode)


# In[50]:


anode_file_fno = "../trained_models/fno/anode_Prada2013_GRF_11000_2025-06-18_22-26-27.msgpack"
cathode_file_fno = "../trained_models/fno/cathode_Prada2013_GRF_11000_2025-06-18_22-34-38.msgpack"

params_anode_fno = functions.load_model_params(anode_file_fno)
params_cathode_fno = functions.load_model_params(cathode_file_fno)

params_anode_fno = flax.serialization.from_bytes(params_fno, params_anode_fno)
params_cathode_fno = flax.serialization.from_bytes(params_fno, params_cathode_fno)


# In[51]:


R_gas = params_bat['Ideal gas constant [J.K-1.mol-1]']
F = params_bat['Faraday constant [C.mol-1]']
Temp = params_bat["Ambient temperature [K]"]


# In[77]:


kan = Ran / (3 * epsan * Lan * A)
kca = Rca / (3 * epsca * Lca * A)
RTF = 2 * R_gas*Temp/F


# In[78]:


def calc_voltage(c_an_surf, c_ca_surf, func_I,):
    # Calculate the anode and cathode OCPs
    j_anode   = jnp.sqrt(c_an_surf * (1.0 - c_an_surf))
    j_cathode = jnp.sqrt(c_ca_surf * (1.0 - c_ca_surf))

    xan = functions.in_arcsinh(-func_I, Ran, epsan, Lan, A)
    xca = functions.in_arcsinh(-func_I, Rca, epsca, Lca, A)

    V_pred = U_OCP_ca(c_ca_surf) - U_OCP_an(c_an_surf) - 2 * R_gas*Temp/F * jnp.arcsinh(0.5*xan/(j_anode)) - 2 * R_gas*Temp/F * jnp.arcsinh(0.5*xca/(j_cathode))

    return V_pred


# In[79]:


@jax.jit
def inference_step(func_I_truth, X_anode_truth, X_cathode_truth, D_anode_candidate, D_cathode_candidate):
    cn_anode = model.apply(params_anode,X_anode_truth, D_anode_candidate)
    cn_cathode = model.apply(params_cathode,X_cathode_truth, D_cathode_candidate)

    cn_anode_reshape = remove_padding(cn_anode, padding_r = padding_r, padding_t = padding_t)
    cn_cathode_reshape = remove_padding(cn_cathode, padding_r = padding_r, padding_t = padding_t)

    cn_anode_surf = cn_anode_reshape[:, -1, :, 0].clip(EPS, 1.0 - EPS)
    cn_cathode_surf = cn_cathode_reshape[:, -1, :, 0].clip(EPS, 1.0 - EPS)

    j_anode   = jnp.sqrt(cn_anode_surf * (1.0 - cn_anode_surf))
    j_cathode = jnp.sqrt(cn_cathode_surf * (1.0 - cn_cathode_surf))

    xan = -func_I_truth * kan
    xca = -func_I_truth * kca

    V_pred = U_OCP_ca(cn_cathode_surf) - U_OCP_an(cn_anode_surf) - RTF * jnp.arcsinh(0.5*xan/(j_anode)) - RTF * jnp.arcsinh(0.5*xca/(j_cathode))
    return V_pred


# In[80]:


@jax.jit
def inference_step_fno(func_I_truth, X_anode_truth, X_cathode_truth):
    cn_anode = model_fno.apply(params_anode_fno,X_anode_truth)
    cn_cathode = model_fno.apply(params_cathode_fno,X_cathode_truth)

    cn_anode_reshape = remove_padding(cn_anode, padding_r = padding_r, padding_t = padding_t)
    cn_cathode_reshape = remove_padding(cn_cathode, padding_r = padding_r, padding_t = padding_t)

    cn_anode_surf = cn_anode_reshape[:, -1, :, 0].clip(EPS, 1.0 - EPS)
    cn_cathode_surf = cn_cathode_reshape[:, -1, :, 0].clip(EPS, 1.0 - EPS)

    j_anode   = jnp.sqrt(cn_anode_surf * (1.0 - cn_anode_surf))
    j_cathode = jnp.sqrt(cn_cathode_surf * (1.0 - cn_cathode_surf))

    xan = -func_I_truth * kan
    xca = -func_I_truth * kca

    V_pred = U_OCP_ca(cn_cathode_surf) - U_OCP_an(cn_anode_surf) - RTF * jnp.arcsinh(0.5*xan/(j_anode)) - RTF * jnp.arcsinh(0.5*xca/(j_cathode))
    return V_pred


# In[81]:


lenr = r.shape[0]
lent = t.shape[0]


# In[82]:


@jax.jit
def inference_step_don(func_I_truth,test_c0_anode, test_c0_cathode, trunk_points):
    cn_anode = jax.vmap(model_don.apply, in_axes=(None,0,0,None))(params_don,func_I_truth,test_c0_anode, trunk_points)
    cn_cathode = jax.vmap(model_don.apply, in_axes=(None,0,0,None))(params_don,func_I_truth,test_c0_cathode, trunk_points)

    cn_anode_surf = cn_anode.reshape(-1,lenr,lent)[:, -1, :].clip(EPS, 1.0 - EPS)
    cn_cathode_surf = cn_cathode.reshape(-1,lenr,lent)[:, -1, :].clip(EPS, 1.0 - EPS)

    j_anode   = jnp.sqrt(cn_anode_surf * (1.0 - cn_anode_surf))
    j_cathode = jnp.sqrt(cn_cathode_surf * (1.0 - cn_cathode_surf))

    xan = -func_I_truth * kan
    xca = -func_I_truth * kca

    V_pred = U_OCP_ca(cn_cathode_surf) - U_OCP_an(cn_anode_surf) - RTF * jnp.arcsinh(0.5*xan/(j_anode)) - RTF * jnp.arcsinh(0.5*xca/(j_cathode))
    return V_pred


# In[83]:


# import pybamm
# import numpy as np

# # ▶  a baseline Li-ion model  (SPMe is fast; swap for DFN if you prefer)
# pybamm_model = pybamm.lithium_ion.SPM()
# pybamm_model.events = []

# # ▶  time grid that matches your measured voltage trace V_true
# t_eval = np.linspace(0, 3600, 75)   # dt = sample spacing [s]
# current_fun = pybamm.Interpolant(t_eval, -func_I[0].squeeze(), pybamm.t)

# #     Tell the model to use that function instead of the built-in C-rate
# params_bat["Current function [A]"] = current_fun

# params_bat["Negative particle diffusivity [m2.s-1]"] = 10 ** D_anode[0]
# params_bat["Positive particle diffusivity [m2.s-1]"] = 10 ** D_cathode


# In[84]:


import time
import gc


# In[85]:


@jax.jit
def run_single(func_I_batch, c0_anode_batch, c0_cathode, D_anode, D_cathode):
    X_anode_init = preprocess_data_fast(func_I_batch, c0_anode_batch)
    X_cathode_init = preprocess_data_fast(func_I_batch, c0_cathode)
    V_pred = inference_step(func_I_batch, X_anode_init, X_cathode_init, D_anode.reshape(-1,1), D_cathode.reshape(-1,1))
    return jax.block_until_ready(V_pred) 


# In[86]:


@jax.jit
def run_single_fno(func_I_batch, c0_anode_batch, c0_cathode):
    X_anode_init = preprocess_data_fast(func_I_batch, c0_anode_batch)
    X_cathode_init = preprocess_data_fast(func_I_batch, c0_cathode)
    V_pred = inference_step_fno(func_I_batch, X_anode_init, X_cathode_init)
    return jax.block_until_ready(V_pred)


# In[87]:


@jax.jit
def run_single_don(func_I_batch, c0_anode_batch, c0_cathode_batch, trunk_points):
    V_pred = inference_step_don(func_I_batch, c0_anode_batch, c0_cathode_batch, trunk_points)
    return jax.block_until_ready(V_pred)


# In[88]:


# # ---- measurement loop -------------------------------------
# N_runs = 100
# times = np.empty(N_runs)

# # Notice that for the real speed increase, the batch size should be large enough to amortize the JIT compilation time.
# # And it should run on the GPU, so make sure you have a GPU available and configured with JAX.

# for batch_size in [120,100,80,128,132]:#[2,4,8,16,32,48,64,100,128,200]:  # you can loop over different batch sizes

#     warmup_catcher = run_single(func_I[:batch_size], c0_anode[:batch_size], c0_cathode[:batch_size], D_anode[:batch_size], D_cathode[:batch_size])
#       # warmup JIT
#     for i in range(1,N_runs+1):
#         gc.collect()
#         idx = int(i*batch_size)
#         #print(time.perf_counter())                     # smooths Python GC noise
#         start = time.perf_counter()
#         run_single(func_I[idx:idx+batch_size], c0_anode[idx:idx+batch_size], c0_cathode[idx:idx+batch_size], D_anode[idx:idx+batch_size], D_cathode[idx:idx+batch_size])
#         times[i-1] = time.perf_counter() - start

#     print(f"{times.mean()*1e3/batch_size:.8f} ± {times.std(ddof=1)*1e3/batch_size:.8f} ms (n={N_runs})")
#     np.save(f"../runtimes/runtimes_capefno_{batch_size}.npy", times)   # keep the evidence


# In[89]:


# # ---- measurement loop -------------------------------------
# N_runs = 100
# times = np.empty(N_runs)

# # Notice that for the real speed increase, the batch size should be large enough to amortize the JIT compilation time.
# # And it should run on the GPU, so make sure you have a GPU available and configured with JAX.

# for batch_size in [80,100,120,128,132]:#[2,4,8,16,32,48,64,100,128,200]:  # you can loop over different batch sizes

#     warmup_catcher = run_single_fno(func_I[:batch_size], c0_anode[:batch_size], c0_cathode[:batch_size])
#       # warmup JIT
#     for i in range(1,N_runs+1):
#         gc.collect()
#         idx = int(i*batch_size)
#         #print(time.perf_counter())                     # smooths Python GC noise
#         start = time.perf_counter()
#         run_single_fno(func_I[idx:idx+batch_size], c0_anode[idx:idx+batch_size], c0_cathode[idx:idx+batch_size])
#         times[i-1] = time.perf_counter() - start

#     print(f"{times.mean()*1e3/batch_size:.8f} ± {times.std(ddof=1)*1e3/batch_size:.8f} ms (n={N_runs})")
#     np.save(f"../runtimes/runtimes_capefno_{batch_size}.npy", times)   # keep the evidence


# In[90]:


# ---- measurement loop -------------------------------------
N_runs = 100
times = np.empty(N_runs)

# Notice that for the real speed increase, the batch size should be large enough to amortize the JIT compilation time.
# And it should run on the GPU, so make sure you have a GPU available and configured with JAX.

for batch_size in [600,600,600]:#[2,4,8,16,32,48,64,100,128,200]:  # you can loop over different batch sizes

    warmup_catcher = run_single_don(func_I[:batch_size], c0_anode[:batch_size], c0_cathode[:batch_size], trunk_points)
      # warmup JIT
    for i in range(1,N_runs+1):
        gc.collect()
        idx = int(i*batch_size)
        #print(time.perf_counter())                     # smooths Python GC noise
        start = time.perf_counter()
        run_single_don(func_I[idx:idx+batch_size], c0_anode[idx:idx+batch_size], c0_cathode[idx:idx+batch_size], trunk_points)
        times[i-1] = time.perf_counter() - start

    print(f"{times.mean()*1e3/batch_size:.8f} ± {times.std(ddof=1)*1e3/batch_size:.8f} ms (n={N_runs})")
    np.save(f"../runtimes/runtimes_capefno_{batch_size}.npy", times) 


# In[53]:


from flax.traverse_util import flatten_dict


# In[54]:


def count_flax_params(params) -> int:
    """
    Count the total number of trainable parameters in a Flax model.

    Args:
        params (flax.core.FrozenDict | dict):  Parameter pytree
            – e.g. the result of `model.init(...)` or `state.params`.

    Returns:
        int: Total number of individual parameters.
    """
    flat = flatten_dict(params)           # flatten nested dict ⇒ {('layer', 'kernel'): ndarray, ...}
    return int(sum(jnp.prod(jnp.array(v.shape)) for v in flat.values()))


# In[61]:


print("Number of parameters in the CAPE FNO model:", count_flax_params(params_anode) + count_flax_params(params_cathode))  # anode and cathode
print("Number of parameters in the FNO model:", count_flax_params(params_anode_fno) + count_flax_params(params_cathode_fno))  # anode and cathode
print("Number of parameters in the DON model:", count_flax_params(params_don))# + count_flax_params(params_cathode_don))  # anode and cathode


# In[62]:


def show_shapes(tree, prefix=""):
    for k, v in flatten_dict(tree, sep="/").items():
        print(f"{prefix}{k:50} {v.shape}  ->  {v.size}")

print("PARAM SHAPES (M=64)")
show_shapes(params_don["params"])  

