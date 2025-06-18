import numpy as np
import jax
import jax.numpy as jnp
import src.util.functions as functions
from src.models.FNO import CAPE_FNO
import pybamm
import flax

from skopt.utils import use_named_args
from skopt.space import Real
from skopt import gp_minimize

from src.util.FNO_util import preprocess_data, train_test_split, remove_padding
# from util.plotting import create_plot_FNO_results, create_plot_voltage
from src.util.postprocess import filter_anode_cathode, calc_error_metrics, calc_error_metrics_all
from src.inference.parameter_estimation import get_jnp_U_OCP_anode, get_jnp_U_OCP_cathode

LOG10_D_MIN, LOG10_D_MAX = -15.0, -12.0
EPS = 1e-6
EPS_pybamm = 1e-6  # PyBaMM uses this value for numerical stability

def  _normalise_diffusion(D, lower=LOG10_D_MIN, upper=LOG10_D_MAX):
    return 2*(D-lower)/(upper-lower) - 1


def estimate_diffusion_parameters(idx):

    family = "GRF"
    N_total = 33000
    data = np.load(f"data/{family}_{N_total}.npz")
    split_seed = 42
    test_ratio = 0.1

    parameter_name = "Prada2013"
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

    t_max = 3600

    train_data, test_data = train_test_split(data, N_total=N_total, test_ratio=test_ratio, seed=split_seed)  # noqa: F405

    func_I = np.array(test_data["current"])

    ### Anode data ###
    cn_anode = np.array(test_data["cn_anode"])
    c0_anode = np.array(test_data["c0_anode"])
    D_anode = np.array(test_data["Dan"])

    ### Cathode data ###
    cn_cathode = np.array(test_data["cn_cathode"])
    c0_cathode = np.array(test_data["c0_cathode"])
    D_cathode = np.array(test_data["Dca"])

    socs = np.array(test_data["soc"])

    D_anode = _normalise_diffusion(D_anode)
    D_cathode = _normalise_diffusion(D_cathode)

    parameter_name = "Prada2013"
    params_bat = pybamm.ParameterValues(parameter_name)
    cs_max_a_norm = 1 #params_bat["Maximum concentration in negative electrode [mol.m-3]"]
    cs_max_c_norm = 1 #params_bat["Maximum concentration in positive electrode [mol.m-3]"]
    cs_min_a_norm = 0.0
    cs_min_c_norm = 0.0

    cn_anode, cn_cathode, mask = filter_anode_cathode(cn_anode, cn_cathode,
                                                        anode_lo=cs_min_a_norm, anode_hi=cs_max_a_norm, 
                                                        cathode_lo=cs_min_c_norm, cathode_hi=cs_max_c_norm)

    func_I = func_I[mask]
    c0_anode = c0_anode[mask]
    c0_cathode = c0_cathode[mask]
    D_anode = D_anode[mask]
    D_cathode = D_cathode[mask]
    socs = socs[mask]

    # Padding amounts
    padding_t = 5  # along t-axis
    padding_r = 2  # along r-axis

    # Original sample counts
    num_samples_I = 75
    num_samples_c0 = 20

    X_anode, Y_anode = preprocess_data(func_I, c0_anode, cn_anode, num_samples_I, num_samples_c0, padding_r, padding_t)
    X_cathode, Y_cathode = preprocess_data(func_I, c0_cathode, cn_cathode, num_samples_I, num_samples_c0, padding_r, padding_t)

    # Assume these hyperparameters
    k_modes = 10
    fno_depth = 8
    hidden_channels = 64
    input_channels = 4 # We should automate this TODO
    output_channels = 1
    cape_hidden_size = 32

    N_test = func_I.shape[0]
    rng = np.random.default_rng(split_seed)
    test_idx = idx #0 #rng.integers(0, N_test) #TODO vorgeben

    model = CAPE_FNO(k_modes=k_modes, input_channels= input_channels, 
                    fno_depth=fno_depth, cape_hidden_size = cape_hidden_size, 
                    hidden_channels=hidden_channels, output_channels=output_channels)

    main_key = jax.random.PRNGKey(split_seed)
    # Initialize parameters
    dummy_D = jax.random.normal(main_key, (1,1))
    params = model.init(main_key, X_anode[:1,...], dummy_D)

    # Forward pass
    out = model.apply(params, X_anode[:1,...], dummy_D)

    anode_file = "trained_models/cape_fno/anode_GRF_2025-06-03_17-11-44.msgpack"
    cathode_file = "trained_models/cape_fno/cathode_GRF_2025-06-03_17-35-54.msgpack"

    params_anode = functions.load_model_params(anode_file)
    params_cathode = functions.load_model_params(cathode_file)

    params_anode = flax.serialization.from_bytes(params, params_anode)
    params_cathode = flax.serialization.from_bytes(params, params_cathode)

    R_gas = params_bat['Ideal gas constant [J.K-1.mol-1]']
    F = params_bat['Faraday constant [C.mol-1]']
    Temp = params_bat["Ambient temperature [K]"]

    def calc_voltage(params_bat, c_an_surf, c_ca_surf, func_I,):
        # Calculate the anode and cathode OCPs

        c_an_surf = c_an_surf.clip(EPS_pybamm, 1.0 - EPS_pybamm)
        c_ca_surf = c_ca_surf.clip(EPS_pybamm, 1.0 - EPS_pybamm)

        j_anode   = jnp.sqrt(c_an_surf * (1.0 - c_an_surf))
        j_cathode = jnp.sqrt(c_ca_surf * (1.0 - c_ca_surf))

        xan = functions.in_arcsinh(-func_I, Ran, epsan, Lan, A)
        xca = functions.in_arcsinh(-func_I, Rca, epsca, Lca, A)

        V_pred = U_OCP_ca(c_ca_surf) - U_OCP_an(c_an_surf) - 2 * R_gas*Temp/F * jnp.arcsinh(0.5*xan/(j_anode)) - 2 * R_gas*Temp/F * jnp.arcsinh(0.5*xca/(j_cathode))

        return V_pred


    kan = Ran / (3 * epsan * Lan * A)
    kca = Rca / (3 * epsca * Lca * A)
    RTF = 2 * R_gas*Temp/F

    def inference_concentration_norm(X_anode_truth, X_cathode_truth, D_anode_candidate, D_cathode_candidate):
        
        cn_anode = model.apply(params_anode,X_anode_truth, D_anode_candidate)
        cn_cathode = model.apply(params_cathode,X_cathode_truth, D_cathode_candidate)

        cn_anode_reshape = remove_padding(cn_anode, padding_r = padding_r, padding_t = padding_t)
        cn_cathode_reshape = remove_padding(cn_cathode, padding_r = padding_r, padding_t = padding_t)

        return cn_anode_reshape, cn_cathode_reshape


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


    # ---- design space ----------------------------------------
    dimensions = [
        Real(LOG10_D_MIN, LOG10_D_MAX, name="log10_Dan"),
        Real(LOG10_D_MIN, LOG10_D_MAX, name="log10_Dca"),
    ]

    V_data = functions.post_proc_true(params_bat, func_I[test_idx], cn_anode[test_idx,-1,:], cn_cathode[test_idx,-1,:], Ran, Rca, epsan, epsca, Lan, Lca, A)
    assert np.isfinite(V_data).all()

    @jax.jit
    def calc_rmse(V_pred, V_data):
        rmse = jnp.sqrt(jnp.mean((V_pred - V_data) ** 2))
        return rmse
    
    @jax.jit
    def calc_rel_ls(V_pred, V_data):
        rel_l2 = jnp.sqrt(jnp.mean((V_pred - V_data) ** 2)) / jnp.sqrt(jnp.mean(V_data ** 2))
        return rel_l2

    @use_named_args(dimensions)
    def obj_voltage(log10_Dan, log10_Dca):
            
            log10_Dan_norm = _normalise_diffusion(jnp.array(log10_Dan).reshape(-1, 1))
            log10_Dca_norm = _normalise_diffusion(jnp.array(log10_Dca).reshape(-1, 1))

            V_pred = inference_step(
                func_I[test_idx],
                X_anode[test_idx][None, ...],
                X_cathode[test_idx][None, ...],
                log10_Dan_norm,
                log10_Dca_norm,
            )

            rmse = calc_rel_ls(V_pred, V_data)
            rmse_val = float(rmse)

            return rmse_val


    res = gp_minimize(
        obj_voltage,
        dimensions=dimensions,
        n_calls=60,            # ±25–35 evaluations per dimension is usually plenty
        n_initial_points=12,   # random Sobol/Bach search first
        acq_func="EI",         # expected improvement
        random_state=0,
    )

    best_log10_Dan, best_log10_Dca = res.x
    # print(f"Best log10(Dan) = {best_log10_Dan:.3f}")
    # print(f"Best log10(Dca) = {best_log10_Dca:.3f}")
    # print(f"Voltage-curve RMSE = {res.fun:.4e}")

    V_best_cape_fno_from_cape_fno = inference_step(func_I[test_idx],
        X_anode[test_idx][None, ...], X_cathode[test_idx][None, ...],
        _normalise_diffusion(jnp.array(best_log10_Dan).reshape(-1,1)), _normalise_diffusion(jnp.array(best_log10_Dca).reshape(-1,1))
    )
    # print(f"Best voltage prediction: {V_best_cape_fno_from_cape_fno}")

    log_Dan_true = float(D_anode[test_idx])
    log_Dca_true = float(D_cathode[test_idx])

    def unnormalise_diffusion(log10_D):
        return (log10_D +1)/2*(LOG10_D_MAX-LOG10_D_MIN) + LOG10_D_MIN

    log_Dan_true = unnormalise_diffusion(log_Dan_true)
    log_Dca_true = unnormalise_diffusion(log_Dca_true)


    # V_test_pred = inference_step(func_I[test_idx],
    #     X_anode[test_idx][None, ...], X_cathode[test_idx][None, ...],
    #     D_anode[test_idx][None, ...].reshape(-1,1), D_cathode[test_idx][None, ...].reshape(-1,1)
    # )

    #assert np.isclose(V_test_pred, V_data, rtol=1e-1).all(), "Test prediction does not match the data!"

    # ---- design space ----------------------------------------
    dimensions_pybamm = [
        Real(LOG10_D_MIN, LOG10_D_MAX, name="log10_Dan_pybamm"),
        Real(LOG10_D_MIN, LOG10_D_MAX, name="log10_Dca_pybamm"),
    ]

    pybamm_model = pybamm.lithium_ion.SPM()
    pybamm_model.events = []

    # ▶  time grid that matches your measured voltage trace V_true
    t_eval = np.linspace(0, 3600, 75)   # dt = sample spacing [s]

    # ▶  current profile: use your measured current array `func_I`
    #     (shape (T,)).  Convert to an interpolant so PyBaMM can call it at any t.
    current_fun = pybamm.Interpolant(t_eval, -func_I[test_idx].squeeze(), pybamm.t)

    #     Tell the model to use that function instead of the built-in C-rate
    params_bat["Current function [A]"] = current_fun



    @use_named_args(dimensions_pybamm)
    def obj_voltage_pybamm(log10_Dan_pybamm, log10_Dca_pybamm):

        # --- 1.  normalise the candidate diffusivities ------------
        D_anode_candidate = 10 ** log10_Dan_pybamm
        D_cathode_candidate = 10 ** log10_Dca_pybamm

        params_bat["Negative particle diffusivity [m2.s-1]"] = D_anode_candidate
        params_bat["Positive particle diffusivity [m2.s-1]"] = D_cathode_candidate

        sim = pybamm.Simulation(pybamm_model, parameter_values=params_bat)
        sim.solve(initial_soc = socs[test_idx], t_eval=t_eval)

        #V_pred = sim.solution["Terminal voltage [V]"](t_eval)

        c_an_surf = sim.solution["Negative particle concentration"].entries[-1,0, :]
        c_ca_surf = sim.solution["Positive particle concentration"].entries[-1,0, :]
        # --- 3.  voltage RMSE (you can use MAE or any other scalar) -------------
        V_pred = calc_voltage(params_bat, c_an_surf, c_ca_surf, func_I[test_idx])

        rmse = jnp.sqrt(jnp.mean((V_pred - V_data) ** 2))/ jnp.sqrt(jnp.mean(V_data ** 2))

        return float(rmse)
    
        
    res_pybamm = gp_minimize(
        obj_voltage_pybamm,
        dimensions=dimensions_pybamm,
        n_calls=60,            # ±25–35 evaluations per dimension is usually plenty
        n_initial_points=12,   # random Sobol/Bach search first
        acq_func="EI",         # expected improvement
        random_state=0,
    )

    best_log10_Dan_pybamm, best_log10_Dca_pybamm = res_pybamm.x

    params_bat["Negative particle diffusivity [m2.s-1]"] = 10 ** best_log10_Dan
    params_bat["Positive particle diffusivity [m2.s-1]"] = 10 ** best_log10_Dca

    sim = pybamm.Simulation(model=pybamm_model, parameter_values=params_bat)
    sim.solve(initial_soc = socs[test_idx], t_eval=t_eval)

    c_an_pybamm_from_cape_fno = sim.solution["Negative particle concentration"].entries
    c_ca_pybamm_from_cape_fno = sim.solution["Positive particle concentration"].entries

    c_an_surf_pybamm_from_cape_fno = c_an_pybamm_from_cape_fno[-1, 0, :]
    c_ca_surf_pybamm_from_cape_fno = c_ca_pybamm_from_cape_fno[-1, 0, :]

    # V_best_pybamm_from_cape_fno = sim.solution["Local voltage [V]"](t_eval)
    # V_best_pybamm_from_cape_fno = calc_voltage(params_bat, c_an_surf_pybamm_from_cape_fno, c_ca_surf_pybamm_from_cape_fno, func_I[test_idx])


    params_bat["Negative particle diffusivity [m2.s-1]"] = 10 ** log_Dan_true
    params_bat["Positive particle diffusivity [m2.s-1]"] = 10 ** log_Dca_true

    sim = pybamm.Simulation(model=pybamm_model, parameter_values=params_bat)
    sim.solve(initial_soc = socs[test_idx], t_eval=t_eval)

    c_an_true_pybamm = sim.solution["Negative particle concentration"].entries
    c_ca_true_pybamm = sim.solution["Positive particle concentration"].entries

    c_an_surf_true_pybamm = c_an_true_pybamm[-1, 0, :]
    c_ca_surf_true_pybamm = c_ca_true_pybamm[-1, 0, :]

    # V_true_pybamm = sim.solution["Local voltage [V]"](t_eval)
    # V_true_pybamm = calc_voltage(params_bat, c_an_surf_true_pybamm, c_ca_surf_true_pybamm, func_I[test_idx])


    params_bat["Negative particle diffusivity [m2.s-1]"] = 10 ** best_log10_Dan_pybamm
    params_bat["Positive particle diffusivity [m2.s-1]"] = 10 ** best_log10_Dca_pybamm

    sim = pybamm.Simulation(model=pybamm_model, parameter_values=params_bat)
    sim.solve(initial_soc = socs[test_idx], t_eval=t_eval)
    # V_best_pybamm_from_pybamm = sim.solution["Local voltage [V]"](t_eval)

    c_an_pybamm_from_pybamm = sim.solution["Negative particle concentration"].entries
    c_ca_pybamm_from_pybamm = sim.solution["Positive particle concentration"].entries

    c_an_surf_pybamm_from_pybamm = c_an_pybamm_from_pybamm[-1, 0, :]
    c_ca_surf_pybamm_from_pybamm = c_ca_pybamm_from_pybamm[-1, 0, :]


    V_best_pybamm_from_cape_fno = calc_voltage(params_bat, c_an_surf_pybamm_from_cape_fno, c_ca_surf_pybamm_from_cape_fno, func_I[test_idx])

    V_true_pybamm = calc_voltage(params_bat, c_an_surf_true_pybamm, c_ca_surf_true_pybamm, func_I[test_idx])

    V_best_pybamm_from_pybamm = calc_voltage(params_bat, c_an_surf_pybamm_from_pybamm, c_ca_surf_pybamm_from_pybamm, func_I[test_idx])

    V_best_cape_fno_from_pybamm = inference_step(func_I[test_idx],
    X_anode[test_idx][None, ...], X_cathode[test_idx][None, ...],
    _normalise_diffusion(jnp.array(best_log10_Dan_pybamm).reshape(-1,1)),
    _normalise_diffusion(jnp.array(best_log10_Dca_pybamm).reshape(-1,1))
    )

    c_an_cape_fno_from_pybamm, c_ca_cape_fno_from_pybamm = inference_concentration_norm(
    X_anode[test_idx][None, ...], X_cathode[test_idx][None, ...],
    _normalise_diffusion(jnp.array(best_log10_Dan_pybamm).reshape(-1,1)),
    _normalise_diffusion(jnp.array(best_log10_Dca_pybamm).reshape(-1,1))
    )   

    # V_test_pred = inference_step(func_I[test_idx],
    #     X_anode[test_idx][None, ...], X_cathode[test_idx][None, ...],
    #     D_anode[test_idx][None, ...].reshape(-1,1), D_cathode[test_idx][None, ...].reshape(-1,1)
    # )

    V_best_cape_fno_from_cape_fno = inference_step(func_I[test_idx],
        X_anode[test_idx][None, ...], X_cathode[test_idx][None, ...],
        _normalise_diffusion(jnp.array(best_log10_Dan).reshape(-1,1)), _normalise_diffusion(jnp.array(best_log10_Dca).reshape(-1,1))
    )

    c_an_cape_fno_from_cape_fno, c_ca_cape_fno_from_cape_fno = inference_concentration_norm(
    X_anode[test_idx][None, ...], X_cathode[test_idx][None, ...],
    _normalise_diffusion(jnp.array(best_log10_Dan).reshape(-1,1)),
    _normalise_diffusion(jnp.array(best_log10_Dca).reshape(-1,1))
    )   

    diffs_anode = {
        "log10_Dan_cape_fno": best_log10_Dan,
        "log10_Dan_pybamm": best_log10_Dan_pybamm,
        "log10_Dan_true": log_Dan_true
    }

    diffs_cathode = {
        "log10_Dca_cape_fno": best_log10_Dca,
        "log10_Dca_pybamm": best_log10_Dca_pybamm,
        "log10_Dca_true": log_Dca_true
    }

    concentrations_anode = {
        "c_an_pybamm_from_cape_fno": c_an_pybamm_from_cape_fno.squeeze(),
        "c_an_data": cn_anode[test_idx, :, :],
        "c_an_pybamm_from_pybamm": c_an_pybamm_from_pybamm.squeeze(),
        "c_an_cape_fno_from_cape_fno": c_an_cape_fno_from_cape_fno.squeeze(),
        "c_an_cape_fno_from_pybamm": c_an_cape_fno_from_pybamm.squeeze(),
    }

    concentrations_cathode = {
        "c_ca_pybamm_from_cape_fno": c_ca_pybamm_from_cape_fno.squeeze(),
        "c_ca_data": cn_cathode[test_idx, :, :],
        "c_ca_pybamm_from_pybamm": c_ca_pybamm_from_pybamm.squeeze(),
        "c_ca_cape_fno_from_cape_fno": c_ca_cape_fno_from_cape_fno.squeeze(),
        "c_ca_cape_fno_from_pybamm": c_ca_cape_fno_from_pybamm.squeeze(),
    }

    voltages = {
        "V_best_cape_fno_from_cape_fno": V_best_cape_fno_from_cape_fno.squeeze(),
        "V_best_cape_fno_from_pybamm": V_best_cape_fno_from_pybamm.squeeze(),
        "V_best_pybamm_from_cape_fno": V_best_pybamm_from_cape_fno.squeeze(),
        "V_best_pybamm_from_pybamm": V_best_pybamm_from_pybamm.squeeze(),
        "V_data": V_data.squeeze(),
    }

    convergence_metrics = {
        "res_cape_fno": res.func_vals.squeeze(),
        "res_pybamm": res_pybamm.func_vals.squeeze(),
    }



    return diffs_anode, diffs_cathode, concentrations_anode, concentrations_cathode, voltages, convergence_metrics, res.fun