#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import jax
import jax.numpy as jnp
import util.functions as functions
from models.FNO import CAPE_FNO2, CAPE_FNO
import pybamm
import flax


# In[2]:


from util.FNO_util import preprocess_data, train_test_split, remove_padding, normalise_diffusion
from util.postprocess import filter_anode_cathode, calc_error_metrics, calc_error_metrics_all


# In[3]:


family = "Triangle"                   
N_total = 33011
parameter_name = "Prada2013"
data = np.load(f"../data/{parameter_name}_{family}_{N_total}.npz")
random_seed = 42
test_ratio = 0.1

train_data, test_data = train_test_split(data, N_total=N_total, test_ratio=test_ratio, seed=random_seed)


# In[4]:


train_I = np.array(train_data["current"])[:2000]
test_I = np.array(test_data["current"])[:2000]

### Anode data ###
train_cn_anode = np.array(train_data["cn_anode"])[:2000]
test_cn_anode = np.array(test_data["cn_anode"])[:2000]
train_c0_anode = np.array(train_data["c0_anode"])[:2000]
test_c0_anode = np.array(test_data["c0_anode"])[:2000]
train_D_anode = np.array(train_data["D_n"])[:2000]
test_D_anode = np.array(test_data["D_n"])[:2000]
# train_eps_anode = np.array(train_data["eps_n"])
# test_eps_anode = np.array(test_data["eps_n"])
# train_L_anode = np.array(train_data["L_n"])
# test_L_anode = np.array(test_data["L_n"])
train_R_anode = np.array(train_data["R_n"])[:2000]
test_R_anode = np.array(test_data["R_n"])[:2000]

# train_A = np.array(train_data["A"])
# test_A = np.array(test_data["A"])


###Cathode data ###
train_cn_cathode = np.array(train_data["cn_cathode"])[:2000]
test_cn_cathode = np.array(test_data["cn_cathode"])[:2000]
train_c0_cathode = np.array(train_data["c0_cathode"])[:2000]
test_c0_cathode = np.array(test_data["c0_cathode"])[:2000]
train_D_cathode = np.array(train_data["D_p"])[:2000]
test_D_cathode = np.array(test_data["D_p"])[:2000]
# train_eps_cathode = np.array(train_data["eps_p"])
# test_eps_cathode = np.array(test_data["eps_p"])
# train_L_cathode = np.array(train_data["L_p"])
# test_L_cathode = np.array(test_data["L_p"])
train_R_cathode = np.array(train_data["R_p"])[:2000]
test_R_cathode = np.array(test_data["R_p"])[:2000]


# In[5]:


params_bat = pybamm.ParameterValues(parameter_name)
cs_max_a = params_bat["Maximum concentration in negative electrode [mol.m-3]"]
cs_max_c = params_bat["Maximum concentration in positive electrode [mol.m-3]"]

cs_max_a_norm = 1.
cs_max_c_norm = 1.
cs_min_a_norm = 0.0
cs_min_c_norm = 0.0


# In[6]:


train_cn_anode, train_cn_cathode, train_mask = filter_anode_cathode(train_cn_anode, train_cn_cathode,
                                                               anode_lo=cs_min_a_norm, anode_hi=cs_max_a_norm, 
                                                               cathode_lo=cs_min_c_norm, cathode_hi=cs_max_c_norm)


# In[7]:


test_cn_anode, test_cn_cathode, test_mask = filter_anode_cathode(test_cn_anode, test_cn_cathode,
                                                               anode_lo=cs_min_a_norm, anode_hi=cs_max_a_norm, 
                                                               cathode_lo=cs_min_c_norm, cathode_hi=cs_max_c_norm)


# In[8]:


train_I = train_I[train_mask]
test_I = test_I[test_mask]
train_c0_anode = train_c0_anode[train_mask]
test_c0_anode = test_c0_anode[test_mask]
train_c0_cathode = train_c0_cathode[train_mask]
test_c0_cathode = test_c0_cathode[test_mask]
train_D_anode = train_D_anode[train_mask]
test_D_anode = test_D_anode[test_mask]
train_D_cathode = train_D_cathode[train_mask]
test_D_cathode = test_D_cathode[test_mask]
train_R_anode = train_R_anode[train_mask]
test_R_anode = test_R_anode[test_mask]
train_R_cathode = train_R_cathode[train_mask]
test_R_cathode = test_R_cathode[test_mask]


# In[9]:


PARAM_BOUNDS = {
    # short name        | PyBaMM key                                      : (low, high)
    "D_n|Negative particle diffusivity [m2.s-1]"           : (1e-18, 1e-14),
    "D_p|Positive particle diffusivity [m2.s-1]"           : (1e-18, 1e-14),
    "R_n|Negative particle radius [m]"                     : (4e-6, 1.5e-5),
    "R_p|Positive particle radius [m]"                     : (1e-8, 1.5e-5),
    "eps_n|Negative electrode active material volume fraction": (0.30, 0.80),
    "eps_p|Positive electrode active material volume fraction": (0.30, 0.80),
    "L_n|Negative electrode thickness [m]"                 : (2e-5, 10e-5),
    "L_p|Positive electrode thickness [m]"                 : (2e-5, 10e-5),
    "A|Electrode area [m2]"                                : (5e-3, 5e-1),
}


# In[10]:


def scale_parameters(train_param_anode, test_param_anode, train_param_cathode, test_param_cathode, key_an, key_ca, log_scale=True):
    if log_scale:
        train_param_anode, test_param_anode, train_param_cathode, test_param_cathode = np.log10(train_param_anode), np.log10(test_param_anode), np.log10(train_param_cathode), np.log10(test_param_cathode)
        upper_an = np.log10(PARAM_BOUNDS[key_an][1])
        lower_an = np.log10(PARAM_BOUNDS[key_an][0])
        upper_ca = np.log10(PARAM_BOUNDS[key_ca][1])
        lower_ca = np.log10(PARAM_BOUNDS[key_ca][0])
    else:
        upper_an = PARAM_BOUNDS[key_an][1]
        lower_an = PARAM_BOUNDS[key_an][0]
        upper_ca = PARAM_BOUNDS[key_ca][1]
        lower_ca = PARAM_BOUNDS[key_ca][0]
    train_param_anode = normalise_diffusion(train_param_anode,lower=lower_an, upper=upper_an).reshape(-1, 1)
    test_param_anode = normalise_diffusion(test_param_anode,lower=lower_an, upper=upper_an).reshape(-1, 1)
    train_param_cathode = normalise_diffusion(train_param_cathode,lower=lower_ca, upper=upper_ca).reshape(-1, 1)
    test_param_cathode = normalise_diffusion(test_param_cathode,lower=lower_ca, upper=upper_ca).reshape(-1, 1)
    return train_param_anode, test_param_anode, train_param_cathode, test_param_cathode


# In[11]:


# --- helper that reverses normalise_diffusion -----------------------------
def denormalise_diffusion(x_scaled, *, lower, upper):
    """
    Inverse of the min–max map that sent [lower, upper] → [-1, 1].

    x_scaled : ndarray
        Values in [-1, 1].
    lower, upper : floats
        The original bounds used during normalisation.

    Returns
    -------
    x_orig : ndarray
        Back in the original linear (or log) scale.
    """
    return 0.5 * (x_scaled + 1.0) * (upper - lower) + lower


# --- full inverse ----------------------------------------------------------
def rescale_parameters(
    train_scaled_an,  test_scaled_an,
    train_scaled_ca,  test_scaled_ca,
    key_an,           key_ca,
    log_scale=True,
):
    """
    Undo `scale_parameters`: from scaled ([-1,1]) → original physical units.

    Parameters
    ----------
    train_scaled_* : ndarray
        Column vectors produced by `scale_parameters`.
    key_an / key_ca : str
        Short keys as used in PARAM_BOUNDS, e.g. "D_n", "D_p".
    log_scale : bool
        Must match the flag used during scaling.

    Returns
    -------
    train_orig_an, test_orig_an, train_orig_ca, test_orig_ca
        Arrays in the original linear scale (m² s⁻¹ for diffusivity).
    """
    # --- 1. set bounds -----------------------------------------------------
    if log_scale:
        lower_an = np.log10(PARAM_BOUNDS[key_an][0])
        upper_an = np.log10(PARAM_BOUNDS[key_an][1])
        lower_ca = np.log10(PARAM_BOUNDS[key_ca][0])
        upper_ca = np.log10(PARAM_BOUNDS[key_ca][1])
    else:
        lower_an, upper_an = PARAM_BOUNDS[key_an]
        lower_ca, upper_ca = PARAM_BOUNDS[key_ca]

    # --- 2. de-normalise ---------------------------------------------------
    train_an_log = denormalise_diffusion(train_scaled_an.ravel(),
                                         lower=lower_an, upper=upper_an)
    test_an_log  = denormalise_diffusion(test_scaled_an.ravel(),
                                         lower=lower_an, upper=upper_an)
    train_ca_log = denormalise_diffusion(train_scaled_ca.ravel(),
                                         lower=lower_ca, upper=upper_ca)
    test_ca_log  = denormalise_diffusion(test_scaled_ca.ravel(),
                                         lower=lower_ca, upper=upper_ca)

    # --- 3. if in log-space, exponentiate back ----------------------------
    if log_scale:
        train_an_orig = (10 ** train_an_log).reshape(-1, 1)
        test_an_orig  = (10 ** test_an_log ).reshape(-1, 1)
        train_ca_orig = (10 ** train_ca_log).reshape(-1, 1)
        test_ca_orig  = (10 ** test_ca_log ).reshape(-1, 1)
    else:
        train_an_orig = train_an_log.reshape(-1, 1)
        test_an_orig  = test_an_log.reshape(-1, 1)
        train_ca_orig = train_ca_log.reshape(-1, 1)
        test_ca_orig  = test_ca_log.reshape(-1, 1)

    return train_an_orig, test_an_orig, train_ca_orig, test_ca_orig


# In[12]:


train_D_anode, test_D_anode, train_D_cathode, test_D_cathode = scale_parameters(train_D_anode, test_D_anode, train_D_cathode, test_D_cathode, "D_n|Negative particle diffusivity [m2.s-1]", "D_p|Positive particle diffusivity [m2.s-1]", log_scale=True)
train_R_anode, test_R_anode, train_R_cathode, test_R_cathode = scale_parameters(train_R_anode, test_R_anode, train_R_cathode, test_R_cathode, "R_n|Negative particle radius [m]", "R_p|Positive particle radius [m]", log_scale=True)


# In[13]:


# Padding amounts
padding_t = 5  # along t-axis
padding_r = 2  # along r-axis

# Original sample counts
num_samples_I = 75
num_samples_c0 = 20


# In[14]:


X_test_anode, Y_test_anode = preprocess_data(test_I, test_c0_anode, test_cn_anode, num_samples_I, num_samples_c0, padding_r, padding_t)
X_test_cathode, Y_test_cathode = preprocess_data(test_I, test_c0_cathode, test_cn_cathode, num_samples_I, num_samples_c0, padding_r, padding_t)


# In[15]:


# Assume these hyperparameters
k_modes = (5,20)
fno_depth = 8
hidden_channels = 64
input_channels = X_test_anode.shape[-1]  # should be 4
output_channels = 1
cape_hidden_size = 32


# In[16]:


model = CAPE_FNO2(k_modes=k_modes, input_channels= input_channels, fno_depth=fno_depth, cape_hidden_size = cape_hidden_size, hidden_channels=hidden_channels, output_channels=output_channels)

# Initialize parameters
init_key = jax.random.PRNGKey(42)
dummy_D = jax.random.normal(init_key, (1,1))
dummy_R = jax.random.normal(init_key, (1,1))
params = model.init(init_key, X_test_anode[:1,...], dummy_D, dummy_R)

# Forward pass
out = model.apply(params, X_test_anode[:1,...], dummy_D, dummy_R)


# In[17]:


# anode_file = "../trained_models/cape_fno2/anode_Prada2013_CC_33000_2025-06-29_17-43-58.msgpack"
# cathode_file = "../trained_models/cape_fno2/cathode_Prada2013_CC_33000_2025-06-29_18-08-52.msgpack"

anode_file = "../trained_models/cape_fno2/anode_Prada2013_Triangle_33011_2025-07-01_22-00-47.msgpack"
cathode_file = "../trained_models/cape_fno2/cathode_Prada2013_Triangle_33011_2025-07-01_22-28-02.msgpack"

params_anode = functions.load_model_params(anode_file)
params_cathode = functions.load_model_params(cathode_file)

params_anode = flax.serialization.from_bytes(params, params_anode)
params_cathode = flax.serialization.from_bytes(params, params_cathode)


# In[18]:


# parameter_name = "Prada2013"
params_bat = pybamm.ParameterValues(parameter_name)

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
t_max = 3600

t = np.linspace(0, 1, num_samples_I)
r = np.linspace(0, 1, num_samples_c0)


# In[19]:


X_test_anode.shape


# In[20]:


c_test_true_anode = test_cn_anode
c_test_pred_anode = model.apply(params_anode, X_test_anode, test_D_anode, test_R_anode)
c_test_true_reshaped_anode = c_test_true_anode   # already (20,75)
c_test_pred_reshaped_anode = remove_padding(c_test_pred_anode, padding_r, padding_t)

# c_test_true_anode = test_cn_anode
# c_test_pred_anode = jax.vmap(model.apply, in_axes=(None,0,0,None))(params_anode,test_I,test_c0_anode, trunk_points)


# In[21]:


# # Reshape the training true and predicted concentration
# lenr = r.shape[0]
# lent = t.shape[0]

# # Reshape the test true and predicted concentrations
# c_test_true_reshaped_anode = c_test_true_anode.reshape(-1, lenr, lent)
# c_test_pred_reshaped_anode = c_test_pred_anode.reshape(-1, lenr, lent)


# In[22]:


diff = c_test_true_anode - c_test_pred_reshaped_anode.squeeze()


# In[23]:


# c_train_true_cathode = train_cn_cathode
# c_train_pred_cathode = model.apply(params_cathode,X_train_cathode, train_D_cathode)
# c_train_true_reshaped_cathode = c_train_true_cathode  # already (20,75)
# c_train_pred_reshaped_cathode = remove_padding(c_train_pred_cathode, padding_r, padding_t)

c_test_true_cathode = test_cn_cathode
c_test_pred_cathode = model.apply(params_cathode, X_test_cathode, test_D_cathode, test_R_cathode)
c_test_true_reshaped_cathode = c_test_true_cathode   # already (20,75)
c_test_pred_reshaped_cathode = remove_padding(c_test_pred_cathode, padding_r, padding_t)

# c_test_true_cathode = test_cn_cathode
# c_test_pred_cathode = c_test_pred_anode = jax.vmap(model.apply, in_axes=(None,0,0,None))(params_cathode,test_I,test_c0_cathode, trunk_points)


# In[24]:


# # Reshape the training true and predicted concentration
# lenr = r.shape[0]
# lent = t.shape[0]

# # Reshape the test true and predicted concentrations
# c_test_true_reshaped_cathode = c_test_true_cathode.reshape(-1,lenr, lent)
# c_test_pred_reshaped_cathode = c_test_pred_cathode.reshape(-1,lenr, lent)


# In[25]:


# diff = c_test_true_anode - c_test_pred_reshaped_anode.squeeze()
# diff


# In[26]:


#c_train_pred_scaled_anode = c_train_pred_reshaped_anode * cs_max_a
#c_train_true_scaled_anode = c_train_true_reshaped_anode * cs_max_a
#c_train_pred_scaled_cathode = c_train_pred_reshaped_cathode * cs_max_c
#c_train_true_scaled_cathode = c_train_true_reshaped_cathode * cs_max_c

c_test_pred_scaled_anode = c_test_pred_reshaped_anode * cs_max_a
c_test_true_scaled_anode = c_test_true_reshaped_anode * cs_max_a
c_test_pred_scaled_cathode = c_test_pred_reshaped_cathode * cs_max_c
c_test_true_scaled_cathode = c_test_true_reshaped_cathode * cs_max_c


# In[27]:


c_pred_an_surf = c_test_pred_reshaped_anode[:,-1,:].squeeze()
c_true_an_surf = c_test_true_reshaped_anode[:,-1,:].squeeze()
c_pred_ca_surf = c_test_pred_reshaped_cathode[:,-1,:].squeeze()
c_true_ca_surf = c_test_true_reshaped_cathode[:,-1,:].squeeze()


# In[28]:


# Compute V_pred and V_true using post_proc function
#from functions import post_proc
V_pred, V_true = functions.post_proc(params_bat, test_I, c_pred_an_surf, c_true_an_surf, c_pred_ca_surf, c_true_ca_surf, test_R_anode, test_R_cathode, epsan, epsca, Lan, Lca, A)


# In[29]:


V_max = params_bat["Upper voltage cut-off [V]"]
V_min = params_bat["Lower voltage cut-off [V]"]
V_pred_norm = (V_pred - V_min) / (V_max - V_min)
V_true_norm = (V_true - V_min) / (V_max - V_min)


# In[30]:


concentration_errors_anode = calc_error_metrics(c_test_pred_scaled_anode, c_test_true_scaled_anode)
concentration_errors_cathode = calc_error_metrics(c_test_pred_scaled_cathode, c_test_true_scaled_cathode)
concentration_errors_all = calc_error_metrics_all(concentration_errors_anode, concentration_errors_cathode)
voltage_errors = calc_error_metrics(V_pred, V_true, axis=(1,))


# In[31]:


concentration_errors_anode_norm = calc_error_metrics(c_test_pred_reshaped_anode, c_test_true_reshaped_anode)
concentration_errors_cathode_norm = calc_error_metrics(c_test_pred_reshaped_cathode, c_test_true_reshaped_cathode)
concentration_errors_all_norm = calc_error_metrics_all(concentration_errors_anode_norm, concentration_errors_cathode_norm)
voltage_errors_norm = calc_error_metrics(V_pred_norm, V_true_norm, axis=(1,))


# In[32]:


voltage_errors["mae"] * 1000


# In[33]:


concentration_errors_anode_norm["rel_l2"].mean() * 100


# In[34]:


a = concentration_errors_all["mse"]


# In[35]:


np.sqrt(a)


# In[36]:


concentration_errors_all_norm["mse"].mean()


# In[ ]:





# In[37]:


voltage_errors["mae"].mean() * 1e3


# In[38]:


for  key, value in concentration_errors_all.items():

    if key == "mse":
        value = np.sqrt(value)
    if key == "rel_l2":
        value = value * 100
    if key == "mae":
        value = value
    if key == "rel_linf":
        value = value * 100

    print(f"{key}: {value.mean():.4f}")


# In[39]:


for  key, value in voltage_errors.items():

    if key == "mse":
        value = np.sqrt(value)*1000
    if key == "rel_l2":
        value = value * 100
    if key == "mae":
        value = value * 1e3
    if key == "rel_linf":
        value = value * 100

    print(f"{key}: {value.mean():.4f}")

