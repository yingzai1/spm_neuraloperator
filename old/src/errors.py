#!/usr/bin/env python
# coding: utf-8

# In[124]:


import numpy as np
import jax
import jax.numpy as jnp
import util.functions as functions
from models.FNO import FNO
from models.DON import DeepONet, generate_trunk_points
import pybamm
import flax


# In[125]:


from util.FNO_util import preprocess_data, train_test_split, remove_padding, normalise_diffusion
from util.postprocess import filter_anode_cathode, calc_error_metrics, calc_error_metrics_all


# In[ ]:


family = "GRF"
N_total = 33000
parameter_name = "Prada2013"
data = np.load(f"../data/{parameter_name}_{family}_{N_total}.npz")
random_seed = 42
test_ratio = 0.1

train_data, test_data = train_test_split(data, N_total=N_total, test_ratio=test_ratio, seed=random_seed)


# In[127]:


train_I = np.array(train_data["current"])
test_I = np.array(test_data["current"])

### Anode data ###
train_cn_anode = np.array(train_data["cn_anode"])
test_cn_anode = np.array(test_data["cn_anode"])
train_c0_anode = np.array(train_data["c0_anode"])
test_c0_anode = np.array(test_data["c0_anode"])
train_D_anode = np.array(train_data["Dan"])
test_D_anode = np.array(test_data["Dan"])

###Cathode data ###
train_cn_cathode = np.array(train_data["cn_cathode"])
test_cn_cathode = np.array(test_data["cn_cathode"])
train_c0_cathode = np.array(train_data["c0_cathode"])
test_c0_cathode = np.array(test_data["c0_cathode"])
train_D_cathode = np.array(train_data["Dca"])
test_D_cathode = np.array(test_data["Dca"])


# In[128]:


params_bat = pybamm.ParameterValues(parameter_name)
cs_max_a = params_bat["Maximum concentration in negative electrode [mol.m-3]"]
cs_max_c = params_bat["Maximum concentration in positive electrode [mol.m-3]"]

cs_max_a_norm = 1.
cs_max_c_norm = 1.
cs_min_a_norm = 0.0
cs_min_c_norm = 0.0


# In[129]:


train_cn_anode, train_cn_cathode, train_mask = filter_anode_cathode(train_cn_anode, train_cn_cathode,
                                                               anode_lo=cs_min_a_norm, anode_hi=cs_max_a_norm, 
                                                               cathode_lo=cs_min_c_norm, cathode_hi=cs_max_c_norm)


# In[130]:


test_cn_anode, test_cn_cathode, test_mask = filter_anode_cathode(test_cn_anode, test_cn_cathode,
                                                               anode_lo=cs_min_a_norm, anode_hi=cs_max_a_norm, 
                                                               cathode_lo=cs_min_c_norm, cathode_hi=cs_max_c_norm)


# In[131]:


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


# In[132]:


train_D_anode = normalise_diffusion(train_D_anode).reshape(-1, 1)
test_D_anode = normalise_diffusion(test_D_anode).reshape(-1, 1)

train_D_cathode = normalise_diffusion(train_D_cathode).reshape(-1, 1)
test_D_cathode = normalise_diffusion(test_D_cathode).reshape(-1, 1)


# In[133]:


# Padding amounts
padding_t = 5  # along t-axis
padding_r = 2  # along r-axis

# Original sample counts
num_samples_I = 75
num_samples_c0 = 20


# In[134]:


X_test_anode, Y_test_anode = preprocess_data(test_I, test_c0_anode, test_cn_anode, num_samples_I, num_samples_c0, padding_r, padding_t)
X_test_cathode, Y_test_cathode = preprocess_data(test_I, test_c0_cathode, test_cn_cathode, num_samples_I, num_samples_c0, padding_r, padding_t)


# In[135]:


# Assume these hyperparameters
k_modes = 10
fno_depth = 6
hidden_channels = 32
input_channels = X_test_anode.shape[-1]  # should be 4
output_channels = 1


# In[136]:


# width = 500
# depth = 11
# amount_basis = 16
# num_samples_I = 75
# num_samples_c0 = 20
# t_max = 3600
# K = num_samples_I * num_samples_c0
# t = np.linspace(0, t_max, num_samples_I)
# r = np.linspace(0, 1, num_samples_c0)


# In[137]:


# trunk_points = generate_trunk_points(r, t/t_max)


# In[138]:


model = FNO(k_modes=k_modes, fno_depth=fno_depth, hidden_channels=hidden_channels, output_channels=output_channels)

# Initialize parameters
init_key = jax.random.PRNGKey(42)
params = model.init(init_key, X_test_anode[:1,...])

# Forward pass
out = model.apply(params, X_test_anode[:1,...])
print("Output shape:", out.shape)  # should be (1,24,85,1)


# In[139]:


# # Setup model hyperparameters
# branch_layers = [width] * depth + [amount_basis]  # M = 200 at the end
# trunk_layers = branch_layers

# model = DeepONet(branch_layers=branch_layers, trunk_layers=trunk_layers)

# # Create dummy inputs
# key1, key2 = jax.random.split(jax.random.PRNGKey(random_seed))
# dummy_I = jax.random.normal(key1, (num_samples_I,))
# dummy_c0 = jax.random.normal(key1, (num_samples_c0,))      # 75 current samples + 10 initial conditions
# dummy_trunk_input = jax.random.normal(key2, (num_samples_I*num_samples_c0, 2))   # K=750 points in the (t,r)-space

# # Initialize parameters
# params = model.init(jax.random.PRNGKey(42), dummy_I, dummy_c0, dummy_trunk_input)

# # Forward pass
# output = model.apply(params, dummy_I, dummy_c0, dummy_trunk_input)

# print("Output shape:", output.shape)


# In[140]:


# anode_file = "../trained_models/cape_fno/anode_Chen2020_GRF_33000_2025-06-17_22-31-44.msgpack"
# cathode_file = "../trained_models/cape_fno/cathode_Chen2020_GRF_33000_2025-06-17_23-01-46.msgpack"

# anode_file = "../trained_models/cape_fno/anode_Chen2020_PLS_33000_2025-06-17_21-22-22.msgpack"
# cathode_file = "../trained_models/cape_fno/cathode_Chen2020_PLS_33000_2025-06-17_21-46-34.msgpack"

# anode_file = "../trained_models/cape_fno/anode_Chen2020_Triangle_33000_2025-06-17_23-34-25.msgpack"
# cathode_file = "../trained_models/cape_fno/cathode_Chen2020_Triangle_33000_2025-06-17_23-58-46.msgpack"

# anode_file = "../trained_models/cape_fno/anode_Chen2020_CC_33000_2025-06-18_13-25-53.msgpack"
# cathode_file = "../trained_models/cape_fno/cathode_Chen2020_CC_33000_2025-06-18_13-50-09.msgpack"

# anode_file = "../trained_models/cape_fno/anode_Chen2020_CC_33000_2025-06-18_14-11-45.msgpack"
# cathode_file = "../trained_models/cape_fno/cathode_Chen2020_CC_33000_2025-06-18_14-35-01.msgpack"

anode_file = "../trained_models/fno/anode_Chen2020_GRF_11000_2025-06-19_20-29-25.msgpack"
cathode_file = "../trained_models/fno/cathode_Chen2020_GRF_11000_2025-06-19_20-32-00.msgpack"

params_anode = functions.load_model_params(anode_file)
params_cathode = functions.load_model_params(cathode_file)

params_anode = flax.serialization.from_bytes(params, params_anode)
params_cathode = flax.serialization.from_bytes(params, params_cathode)


# In[141]:


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


# In[142]:


# X_train_anode, Y_train_anode = preprocess_data(train_I, train_c0_anode, train_cn_anode, num_samples_I, num_samples_c0, padding_r, padding_t)
# X_test_anode, Y_test_anode = preprocess_data(test_I, test_c0_anode, test_cn_anode, num_samples_I, num_samples_c0, padding_r, padding_t)

# X_train_cathode, Y_train_cathode = preprocess_data(train_I, train_c0_cathode, train_cn_cathode, num_samples_I, num_samples_c0, padding_r, padding_t)
# X_test_cathode, Y_test_cathode = preprocess_data(test_I, test_c0_cathode, test_cn_cathode, num_samples_I, num_samples_c0, padding_r, padding_t)


# In[143]:


# c_train_true_anode = train_cn_anode
# c_train_pred_anode = model.apply(params_anode,X_train_anode, train_D_anode)
# c_train_true_reshaped_anode = c_train_true_anode  # already (20,75)
# c_train_pred_reshaped_anode = remove_padding(c_train_pred_anode, padding_r, padding_t)

c_test_true_anode = test_cn_anode
c_test_pred_anode = model.apply(params_anode,X_test_anode)
c_test_true_reshaped_anode = c_test_true_anode   # already (20,75)
c_test_pred_reshaped_anode = remove_padding(c_test_pred_anode, padding_r, padding_t)

# c_test_true_anode = test_cn_anode
# c_test_pred_anode = jax.vmap(model.apply, in_axes=(None,0,0,None))(params_anode,test_I,test_c0_anode, trunk_points)


# In[144]:


# # Reshape the training true and predicted concentration
# lenr = r.shape[0]
# lent = t.shape[0]

# # Reshape the test true and predicted concentrations
# c_test_true_reshaped_anode = c_test_true_anode.reshape(-1, lenr, lent)
# c_test_pred_reshaped_anode = c_test_pred_anode.reshape(-1, lenr, lent)


# In[145]:


diff = c_test_true_anode - c_test_pred_reshaped_anode.squeeze()


# In[146]:


# c_train_true_cathode = train_cn_cathode
# c_train_pred_cathode = model.apply(params_cathode,X_train_cathode, train_D_cathode)
# c_train_true_reshaped_cathode = c_train_true_cathode  # already (20,75)
# c_train_pred_reshaped_cathode = remove_padding(c_train_pred_cathode, padding_r, padding_t)

c_test_true_cathode = test_cn_cathode
c_test_pred_cathode = model.apply(params_cathode,X_test_cathode)
c_test_true_reshaped_cathode = c_test_true_cathode   # already (20,75)
c_test_pred_reshaped_cathode = remove_padding(c_test_pred_cathode, padding_r, padding_t)

# c_test_true_cathode = test_cn_cathode
# c_test_pred_cathode = c_test_pred_anode = jax.vmap(model.apply, in_axes=(None,0,0,None))(params_cathode,test_I,test_c0_cathode, trunk_points)


# In[147]:


# # Reshape the training true and predicted concentration
# lenr = r.shape[0]
# lent = t.shape[0]

# # Reshape the test true and predicted concentrations
# c_test_true_reshaped_cathode = c_test_true_cathode.reshape(-1,lenr, lent)
# c_test_pred_reshaped_cathode = c_test_pred_cathode.reshape(-1,lenr, lent)


# In[148]:


# diff = c_test_true_anode - c_test_pred_reshaped_anode.squeeze()
# diff


# In[149]:


#c_train_pred_scaled_anode = c_train_pred_reshaped_anode * cs_max_a
#c_train_true_scaled_anode = c_train_true_reshaped_anode * cs_max_a
#c_train_pred_scaled_cathode = c_train_pred_reshaped_cathode * cs_max_c
#c_train_true_scaled_cathode = c_train_true_reshaped_cathode * cs_max_c

c_test_pred_scaled_anode = c_test_pred_reshaped_anode * cs_max_a
c_test_true_scaled_anode = c_test_true_reshaped_anode * cs_max_a
c_test_pred_scaled_cathode = c_test_pred_reshaped_cathode * cs_max_c
c_test_true_scaled_cathode = c_test_true_reshaped_cathode * cs_max_c


# In[150]:


c_pred_an_surf = c_test_pred_reshaped_anode[:,-1,:].squeeze()
c_true_an_surf = c_test_true_reshaped_anode[:,-1,:].squeeze()
c_pred_ca_surf = c_test_pred_reshaped_cathode[:,-1,:].squeeze()
c_true_ca_surf = c_test_true_reshaped_cathode[:,-1,:].squeeze()


# In[151]:


# Compute V_pred and V_true using post_proc function
#from functions import post_proc
V_pred, V_true = functions.post_proc(params_bat, test_I, c_pred_an_surf, c_true_an_surf, c_pred_ca_surf, c_true_ca_surf, Ran, Rca, epsan, epsca, Lan, Lca, A)


# In[152]:


V_max = params_bat["Upper voltage cut-off [V]"]
V_min = params_bat["Lower voltage cut-off [V]"]
V_pred_norm = (V_pred - V_min) / (V_max - V_min)
V_true_norm = (V_true - V_min) / (V_max - V_min)


# In[153]:


concentration_errors_anode = calc_error_metrics(c_test_pred_scaled_anode, c_test_true_scaled_anode)
concentration_errors_cathode = calc_error_metrics(c_test_pred_scaled_cathode, c_test_true_scaled_cathode)
concentration_errors_all = calc_error_metrics_all(concentration_errors_anode, concentration_errors_cathode)
voltage_errors = calc_error_metrics(V_pred, V_true, axis=(1,))


# In[154]:


concentration_errors_anode_norm = calc_error_metrics(c_test_pred_reshaped_anode, c_test_true_reshaped_anode)
concentration_errors_cathode_norm = calc_error_metrics(c_test_pred_reshaped_cathode, c_test_true_reshaped_cathode)
concentration_errors_all_norm = calc_error_metrics_all(concentration_errors_anode_norm, concentration_errors_cathode_norm)
voltage_errors_norm = calc_error_metrics(V_pred_norm, V_true_norm, axis=(1,))


# In[155]:


voltage_errors["mae"] * 1000


# In[156]:


concentration_errors_anode_norm["rel_l2"].mean() * 100


# In[157]:


a = concentration_errors_all["mse"]


# In[158]:


np.sqrt(a)


# In[159]:


concentration_errors_all_norm["mse"].mean()


# In[160]:


voltage_errors["mae"].mean() * 1e3


# In[161]:


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


# In[162]:


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

