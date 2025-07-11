import jax
import jax.numpy as jnp


def get_jnp_U_OCP_anode(name):

    if name == "Prada2013":
        return lambda sto: (
            1.9793 * jnp.exp(-39.3631 * sto)
            + 0.2482
            - 0.0909 * jnp.tanh(29.8538 * (sto - 0.1234))
            - 0.04478 * jnp.tanh(14.9159 * (sto - 0.2769))
            - 0.0205 * jnp.tanh(30.4444 * (sto - 0.6103))
        )
    
    raise ValueError(f"Unknown name: {name}")


def get_jnp_U_OCP_cathode(name):

    if name == "Prada2013":
        return lambda sto: (3.4077 - 0.020269 * sto 
                            + 0.5 * jnp.exp(-150 * sto) 
                            - 0.9 * jnp.exp(-30 * (1 - sto))
        )

    raise ValueError(f"Unknown name: {name}")

def _stuffinarcsinh(func_I, R, epsilon, L, A):
    x = func_I * R / (3 * epsilon * L * A)
    return x


# def inference_step(func_I, X_anode, X_cathode, D_anode, D_cathode2):

#     cn_anode = model.apply(params_anode,X_anode, D_anode)
#     cn_cathode = model.apply(params_cathode,X_cathode, D_cathode)

#     cn_anode_reshape = remove_padding(cn_anode, padding_r = padding_r, padding_t = padding_t)
#     cn_cathode_reshape = remove_padding(cn_cathode, padding_r = padding_r, padding_t = padding_t)

#     cn_anode_surf = cn_anode_reshape[:, -1, :, 0].clip(EPS, 1.0 - EPS)
#     cn_cathode_surf = cn_cathode_reshape[:, -1, :, 0].clip(EPS, 1.0 - EPS)

#     j_anode   = jnp.sqrt(cn_anode_surf * (1.0 - cn_anode_surf))
#     j_cathode = jnp.sqrt(cn_cathode_surf * (1.0 - cn_cathode_surf))

#     xan = _stuffinarcsinh(-func_I, Ran, epsan, Lan, A)
#     xca = _stuffinarcsinh(-func_I, Rca, epsca, Lca, A)

#     V_pred = U_OCP_ca(cn_cathode_surf) - U_OCP_an(cn_anode_surf) - 2 * R_gas*Temp/F * jnp.arcsinh(0.5*xan/(j_anode)) - 2 * R_gas*Temp/F * jnp.arcsinh(0.5*xca/(j_cathode))