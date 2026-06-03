import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import jax.numpy as jnp
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import seaborn as sns

def create_plot_FNO_results(c_train_pred_scaled, c_train_true_scaled, c_test_pred_scaled, c_test_true_scaled,
                     I_train, I_test, t_max, Ran):

    train_error_line = jnp.mean(jnp.abs(c_train_pred_scaled - c_train_true_scaled), axis=0)  # shape (75,)
    test_error_line = jnp.mean(jnp.abs(c_test_pred_scaled - c_test_true_scaled), axis=0)     

    t = np.linspace(0, 1, c_train_pred_scaled.shape[1])
    r = np.linspace(0, 1, c_train_pred_scaled.shape[0])
    T_plot, R_plot = jnp.meshgrid(t, r)

    # Extract data for anode and cathode
    a_data = np.concatenate([
        c_train_pred_scaled.ravel(),
        c_train_true_scaled.ravel()
    ])
    c_data = np.concatenate([
        c_test_pred_scaled.ravel(),
        c_test_true_scaled.ravel()
    ])

    a_min, a_max = a_data.min(), a_data.max()
    c_min, c_max = c_data.min(), c_data.max()

    # Create separate norms for anode and cathode
    a_norm = Normalize(vmin=a_min, vmax=a_max)
    c_norm = Normalize(vmin=c_min, vmax=c_max)

    # Create the figure and main GridSpec
    fig = plt.figure(figsize=(20, 15))
    gs = gridspec.GridSpec(2, 3, figure=fig)

    def rasterize_contour(contour):
        for c in contour.collections:
            c.set_rasterized(True)

    # Anode Plots (use a_norm)
    ax1 = fig.add_subplot(gs[0,0])
    contour1 = ax1.contourf(T_plot*t_max, R_plot*Ran*1e6, c_train_pred_scaled,
                            levels=50, cmap='viridis', norm=a_norm)
    #rasterize_contour(contour1)
    ax1.set_xlabel('Time [$s$]')
    ax1.set_ylabel('Radial position [$µm$]')
    ax1.set_title('Predicted Lithium Concentration in\nParticle from Training Sample [$\\frac{mol}{m^3}$]', pad=14)

    ax4 = fig.add_subplot(gs[1,0])
    contour3 = ax4.contourf(T_plot*t_max, R_plot*Ran*1e6, c_train_true_scaled,
                            levels=50, cmap='viridis', norm=a_norm)
    #rasterize_contour(contour3)
    ax4.set_xlabel('Time [$s$]')
    ax4.set_ylabel('Radial position [$µm$]')
    ax4.set_title('True Lithium Concentration in\nParticle from Training Sample [$\\frac{mol}{m^3}$]', pad=14)

    # Cathode Plots (use c_norm)
    ax2 = fig.add_subplot(gs[0,1])
    contour2 = ax2.contourf(T_plot*t_max, R_plot*Ran*1e6, c_test_pred_scaled,
                            levels=50, cmap='viridis', norm=c_norm)
    #rasterize_contour(contour2)
    ax2.set_xlabel('Time [$s$]')
    ax2.set_ylabel('Radial position [$µm$]')
    ax2.set_title('Predicted Lithium Concentration in\nParticle from Test Sample [$\\frac{mol}{m^3}$]', pad=14)

    ax5 = fig.add_subplot(gs[1,1])
    contour4 = ax5.contourf(T_plot*t_max, R_plot*Ran*1e6, c_test_true_scaled,
                            levels=50, cmap='viridis', norm=c_norm)
    #rasterize_contour(contour4)
    ax5.set_xlabel('Time [$s$]')
    ax5.set_ylabel('Radial position [$µm$]')
    ax5.set_title('True Lithium Concentration in\nParticle from Test Sample [$\\frac{mol}{m^3}$]', pad=14)

    # Right column subdivided into three rows: Current, Voltage, Error
    right_gs = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[:, 2], hspace=0.4)

    ax_curr = fig.add_subplot(right_gs[0,0])
    ax_curr.plot(t, I_train)
    ax_curr.set_title('Training Current', pad=14)
    ax_curr.set_xlabel('Time [$s$]')
    ax_curr.set_ylabel('Current [$A$]')

    ax_volt = fig.add_subplot(right_gs[1,0])
    ax_volt.plot(t, I_test, linestyle='-')
    #ax_volt.plot(t_plot[:sim_length], V_pred[:sim_length], label='PINN', linestyle='--')
    ax_volt.set_title('Test Current', pad=14)
    ax_volt.set_xlabel('Time [$s$]')
    ax_volt.set_ylabel('Current [$A$]')

    ax_err = fig.add_subplot(right_gs[2,0])
    ax_err.plot(t, train_error_line, color='grey', label='Training Error', linestyle='-')
    ax_err.plot(t, test_error_line, color='black', label='Test Error', linestyle='-')
    ax_err.set_title('Absolute Error', pad=14)
    ax_err.set_xlabel('Time [$s$]')
    ax_err.set_ylabel('Absolute Error [$\\frac{mol}{m^3}$]')
    ax_err.legend()

    plt.tight_layout(rect=[0,0.125,1,1])  # Leave more space at the bottom for two colorbars

    # Create two separate ScalarMappables for the colorbars
    sm_anode = ScalarMappable(norm=a_norm, cmap='viridis')
    sm_anode.set_array([])
    sm_cathode = ScalarMappable(norm=c_norm, cmap='viridis')
    sm_cathode.set_array([])

    # Add the two colorbars side by side at the bottom
    # Adjust positions as needed
    cbar_width = 0.5
    cbar_height = 0.02
    cbar_anode_ax = fig.add_axes([0.25, 0.05, cbar_width, cbar_height])
    cbar_cathode_ax = fig.add_axes([0.25, 0.05, cbar_width, cbar_height])

    cbar_anode = fig.colorbar(sm_anode, cax=cbar_anode_ax, orientation='horizontal')
    cbar_anode.set_label('Training Scale [$\\frac{mol}{m^3}$]', labelpad=12)
    cbar_anode.ax.xaxis.set_label_position('top')
    cbar_anode.ax.xaxis.set_ticks_position('top')

    cbar_cathode = fig.colorbar(sm_cathode, cax=cbar_cathode_ax, orientation='horizontal')
    cbar_cathode.set_label('Test Scale [$\\frac{mol}{m^3}$]')

    axes = {
        'anode_pred': ax1, 'anode_true': ax4,
        'cathode_pred': ax2, 'cathode_true': ax5,
        'current': ax_curr, 'error': ax_err
    }
    return fig, axes


def plot_losses(train_losses, test_losses):
    """
    Plot training and testing losses over epochs.
    
    Parameters
    ----------
    train_losses : list or np.ndarray
        Training losses for each epoch.
    test_losses : list or np.ndarray
        Testing losses for each epoch.
    """
    # After training, plot results
    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('$L_2$-Loss')
    #plt.title('Training vs Test Loss (Bias-Variance)')
    plt.legend()
    #plt.grid(True)
    plt.tight_layout()
    #plt.savefig("DON/Loss_DON" + ".svg", format='svg', transparent=True)
    plt.show()

def create_plot_voltage(c_test_pred_scaled_anode, c_test_true_scaled_anode,
                        c_test_pred_scaled_cathode, c_test_true_scaled_cathode,
                        func_I, V_pred, V_true, t_max, Ran, Rca):

    # Compute error lines
    anode_error_line = jnp.mean(jnp.abs(c_test_pred_scaled_anode - c_test_true_scaled_anode), axis=0)  # shape (75,)
    cathode_error_line = jnp.mean(jnp.abs(c_test_pred_scaled_cathode - c_test_true_scaled_cathode), axis=0)      # shape (75,)
    diff = jnp.abs(V_pred-V_true)

    t = np.linspace(0,1,c_test_pred_scaled_anode.shape[1])
    r = np.linspace(0,1,c_test_pred_scaled_anode.shape[0])

    T_plot, R_plot = jnp.meshgrid(t, r)

    # Extract data for anode and cathode
    a_data = np.concatenate([
        c_test_pred_scaled_anode.ravel(),
        c_test_true_scaled_anode.ravel()
    ])
    c_data = np.concatenate([
        c_test_pred_scaled_cathode.ravel(),
        c_test_true_scaled_cathode.ravel()
    ])

    a_min, a_max = a_data.min(), a_data.max()
    c_min, c_max = c_data.min(), c_data.max()

    # Create separate norms for anode and cathode
    a_norm = Normalize(vmin=a_min, vmax=a_max)
    c_norm = Normalize(vmin=c_min, vmax=c_max)

    # Create the figure and main GridSpec
    fig = plt.figure(figsize=(20, 15))
    gs = gridspec.GridSpec(2, 3, figure=fig)

    def rasterize_contour(contour):
        for c in contour.collections:
            c.set_rasterized(True)

    # Anode Plots (use a_norm)
    ax1 = fig.add_subplot(gs[0,0])
    contour1 = ax1.contourf(T_plot*t_max, R_plot*Ran*1e6, c_test_pred_scaled_anode,
                            levels=50, cmap='viridis', norm=a_norm)
    #rasterize_contour(contour1)
    ax1.set_xlabel('Time [$s$]')
    ax1.set_ylabel('Radial position [$µm$]')
    ax1.set_title('Predicted Lithium Concentration in\n Anode Particle [$\\frac{mol}{m^3}$]', pad=14)

    ax4 = fig.add_subplot(gs[1,0])
    contour3 = ax4.contourf(T_plot*t_max, R_plot*Ran*1e6, c_test_true_scaled_anode,
                            levels=50, cmap='viridis', norm=a_norm)
    #rasterize_contour(contour3)
    ax4.set_xlabel('Time [$s$]')
    ax4.set_ylabel('Radial position [$µm$]')
    ax4.set_title('True Lithium Concentration in\n Anode Particle [$\\frac{mol}{m^3}$]', pad=14)

    # Cathode Plots (use c_norm)
    ax2 = fig.add_subplot(gs[0,1])
    contour2 = ax2.contourf(T_plot*t_max, R_plot*Rca*1e6, c_test_pred_scaled_cathode,
                            levels=50, cmap='viridis', norm=c_norm)
    #rasterize_contour(contour2)
    ax2.set_xlabel('Time [$s$]')
    ax2.set_ylabel('Radial position [$µm$]')
    ax2.set_title('Predicted Lithium Concentration in\n Cathode Particle [$\\frac{mol}{m^3}$]', pad=14)

    ax5 = fig.add_subplot(gs[1,1])
    contour4 = ax5.contourf(T_plot*t_max, R_plot*Rca*1e6, c_test_true_scaled_cathode,
                            levels=50, cmap='viridis', norm=c_norm)
    #rasterize_contour(contour4)
    ax5.set_xlabel('Time [$s$]')
    ax5.set_ylabel('Radial position [$µm$]')
    ax5.set_title('True Lithium Concentration in\n Cathode Particle [$\\frac{mol}{m^3}$]', pad=14)

    # Right column subdivided into three rows: Current, Voltage, Error
    right_gs = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[:, 2], hspace=0.4)

    ax_curr = fig.add_subplot(right_gs[0,0])
    ax_curr.plot(t, func_I)
    ax_curr.set_title('Input Current', pad=14)
    ax_curr.set_xlabel('Time [$s$]')
    ax_curr.set_ylabel('Current [$A$]')

    ax_volt = fig.add_subplot(right_gs[1,0])
    ax_volt.plot(t, V_true, label='Ground Truth', linestyle='-')
    ax_volt.plot(t, V_pred, label='FNO', linestyle='--')
    ax_volt.set_title('Ground Truth vs FNO Voltage', pad=14)
    ax_volt.set_xlabel('Time [$s$]')
    ax_volt.set_ylabel('Cell Voltage [$V$]')
    ax_volt.legend()

    ax_err = fig.add_subplot(right_gs[2,0])
    line1, = ax_err.plot(t, anode_error_line, color='grey', label='Anode Error', linestyle='-')
    line2, = ax_err.plot(t, cathode_error_line, color='black', label='Cathode Error', linestyle='-')
    ax_err.set_title('Absolute Error', pad=14)
    ax_err.set_xlabel('Time [$s$]')
    ax_err.set_ylabel('Concentration [$\\frac{mol}{m^3}$]')
    #ax_err.legend()

    # Create a secondary y-axis for voltage
    ax_voltage_err = ax_err.twinx()
    line3, = ax_voltage_err.plot(t, diff * 1000, color='blue', label='Voltage Error', linestyle='-')
    ax_voltage_err.set_ylabel('Voltage [$mV$]')
    #ax_voltage_err.legend(loc='upper right')

    # Combine the legend handles and labels from both axes
    lines = [line1, line2, line3]
    labels = [line.get_label() for line in lines]
    ax_err.legend(lines, labels, loc='upper right')#, bbox_to_anchor=(0.5, 1.15), ncol=3)

    plt.tight_layout(rect=[0,0.125,1,1])  # Leave more space at the bottom for two colorbars

    # Create two separate ScalarMappables for the colorbars
    sm_anode = ScalarMappable(norm=a_norm, cmap='viridis')
    sm_anode.set_array([])
    sm_cathode = ScalarMappable(norm=c_norm, cmap='viridis')
    sm_cathode.set_array([])

    # Add the two colorbars side by side at the bottom
    # Adjust positions as needed
    cbar_width = 0.5
    cbar_height = 0.02
    cbar_anode_ax = fig.add_axes([0.25, 0.05, cbar_width, cbar_height])
    cbar_cathode_ax = fig.add_axes([0.25, 0.05, cbar_width, cbar_height])

    cbar_anode = fig.colorbar(sm_anode, cax=cbar_anode_ax, orientation='horizontal')
    cbar_anode.set_label('Anode Scale [$\\frac{mol}{m^3}$]', labelpad=12)
    cbar_anode.ax.xaxis.set_label_position('top')
    cbar_anode.ax.xaxis.set_ticks_position('top')

    cbar_cathode = fig.colorbar(sm_cathode, cax=cbar_cathode_ax, orientation='horizontal')
    cbar_cathode.set_label('Cathode Scale [$\\frac{mol}{m^3}$]')

    axes = {
        'anode_pred'   : ax1,
        'anode_true'   : ax4,
        'cathode_pred' : ax2,
        'cathode_true' : ax5,
        'current'      : ax_curr,
        'voltage'      : ax_volt,
        'voltage_err'  : ax_voltage_err,  # the twin-y secondary axis
        'error'        : ax_err
    }

    return fig, axes


def create_plot_paper6(
        pred_sets,
        c_true_anode, c_true_cathode,
        func_I, V_true,
        t_max, Ran, Rca,
        pred_labels=None,
        cmap='viridis',
        err_cmap='plasma'):
    """
    Four-row layout
    --------------
     Row-0 : Anode |error| maps  + legend cell
     Row-1 : Cathode |error| maps
     Row-2 : True Anode  + Anode predictions   + I(t)
     Row-3 : True Cath.  + Cathode predictions + V(t) + |V-err|

    Two global concentration colour-bars sit *just* below Row-3.
    """
    # ..................................................................

    n_pred = len(pred_sets)
    if pred_labels is None:
        pred_labels = [f'Pred-{i+1}' for i in range(n_pred)]
    if len(pred_labels) != n_pred:
        raise ValueError('pred_labels length must match pred_sets length')

    # meshes -----------------------------------------------------------
    R, T = c_true_anode.shape
    t = np.linspace(0, 1, T)
    r = np.linspace(0, 1, R)
    T_plot, R_plot = np.meshgrid(t, r)

    # colour norms -----------------------------------------------------
    a_norm = Normalize(np.min([c_true_anode] +
                              [p['anode'] for p in pred_sets]),
                       np.max([c_true_anode] +
                              [p['anode'] for p in pred_sets]))
    c_norm = Normalize(np.min([c_true_cathode] +
                              [p['cathode'] for p in pred_sets]),
                       np.max([c_true_cathode] +
                              [p['cathode'] for p in pred_sets]))

    err_list = []
    for p in pred_sets:
        err_list.append(np.abs(p['anode']   - c_true_anode))
        err_list.append(np.abs(p['cathode'] - c_true_cathode))
    err_norm = Normalize(np.min(err_list), np.max(err_list))

    # figure + gridspec -----------------------------------------------
    n_cols = 1 + n_pred + 1                         # truth | preds | lines
    fig = plt.figure(figsize=(4.8 * n_cols, 12))
    gs  = gridspec.GridSpec(
        4, n_cols,
        width_ratios=[1] * (n_cols - 1) + [0.9],
        hspace=0.45, wspace=0.28, figure=fig
    )

    axes = {}          # collect axes for optional post-processing

    # =====================  ROW-0 : Anode errors  =====================
    fig.add_subplot(gs[0, 0]).axis('off')           # empty truth cell

    for col, (p, lbl) in enumerate(zip(pred_sets, pred_labels), start=1):
        ax = fig.add_subplot(gs[0, col])
        ax.contourf(T_plot * t_max, R_plot * Ran * 1e6,
                    np.abs(p['anode'] - c_true_anode),
                    levels=50, cmap=err_cmap, norm=err_norm)
        ax.set_xticks([])
        ax.set_title(f'{lbl} Anode |error|')
        axes[f'{lbl}_err_anode'] = ax

    # legend placeholder (top-right)
    ax_legend = fig.add_subplot(gs[0, -1])
    ax_legend.axis('off')
    axes['legend'] = ax_legend

    # =====================  ROW-1 : Cathode errors ====================
    fig.add_subplot(gs[1, 0]).axis('off')

    for col, (p, lbl) in enumerate(zip(pred_sets, pred_labels), start=1):
        ax = fig.add_subplot(gs[1, col])
        ax.contourf(T_plot * t_max, R_plot * Rca * 1e6,
                    np.abs(p['cathode'] - c_true_cathode),
                    levels=50, cmap=err_cmap, norm=err_norm)
        ax.set_xticks([])
        ax.set_title(f'{lbl} Cathode |error|')
        axes[f'{lbl}_err_cathode'] = ax

    # =====================  ROW-2 : Anode concentration ===============

    ax_true_an = fig.add_subplot(gs[2, 0])
    ax_true_an.contourf(T_plot * t_max, R_plot * Ran * 1e6,
                        c_true_anode, levels=50, cmap=cmap, norm=a_norm)
    ax_true_an.set_xlabel('Time [s]')
    ax_true_an.set_ylabel('Radial position [µm]')
    ax_true_an.set_title('True Anode\n[$mol\,m^{-3}$]')
    axes['true_anode'] = ax_true_an

    for col, (p, lbl) in enumerate(zip(pred_sets, pred_labels), start=1):
        ax = fig.add_subplot(gs[2, col])
        ax.contourf(T_plot * t_max, R_plot * Ran * 1e6,
                    p['anode'], levels=50, cmap=cmap, norm=a_norm)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Radial position [µm]')
        ax.set_title(f'{lbl} Anode')
        axes[f'{lbl}_anode'] = ax

    # right column: 3 stacked time-series occupying rows 2-3
    line_gs = gridspec.GridSpecFromSubplotSpec(
        3, 1, subplot_spec=gs[2:4, -1], hspace=0.38
    )

    ax_I = fig.add_subplot(line_gs[0])
    ax_I.plot(t * t_max, func_I)
    ax_I.set_title('Input Current')
    ax_I.set_xlabel('Time [s]')
    ax_I.set_ylabel('Current [A]')
    axes['current'] = ax_I

    ax_V = fig.add_subplot(line_gs[1])
    ax_V.plot(t * t_max, V_true, label='Ground Truth', lw=2)
    colour_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i, (p, lbl) in enumerate(zip(pred_sets, pred_labels)):
        ax_V.plot(t * t_max, p['V'],
                  linestyle='--', color=colour_cycle[i % len(colour_cycle)],
                  label=lbl)
    ax_V.set_title('Voltage')
    ax_V.set_xlabel('Time [s]')
    ax_V.set_ylabel('Cell Voltage [V]')
    axes['voltage'] = ax_V

    ax_Verr = fig.add_subplot(line_gs[2])
    for i, (p, lbl) in enumerate(zip(pred_sets, pred_labels)):
        ax_Verr.plot(t * t_max,
                     np.abs(p['V'] - V_true) * 1e3,
                     color=colour_cycle[i % len(colour_cycle)],
                     label=f'{lbl} |V-err|')
    ax_Verr.set_title('Voltage Error')
    ax_Verr.set_xlabel('Time [s]')
    ax_Verr.set_ylabel('Error [mV]')
    axes['voltage_error'] = ax_Verr

    # =====================  ROW-3 : Cathode concentration =============

    ax_true_ca = fig.add_subplot(gs[3, 0])
    ax_true_ca.contourf(T_plot * t_max, R_plot * Rca * 1e6,
                        c_true_cathode, levels=50, cmap=cmap, norm=c_norm)
    ax_true_ca.set_xlabel('Time [s]')
    ax_true_ca.set_ylabel('Radial position [µm]')
    ax_true_ca.set_title('True Cathode\n[$mol\,m^{-3}$]')
    axes['true_cathode'] = ax_true_ca

    for col, (p, lbl) in enumerate(zip(pred_sets, pred_labels), start=1):
        ax = fig.add_subplot(gs[3, col])
        ax.contourf(T_plot * t_max, R_plot * Rca * 1e6,
                    p['cathode'], levels=50, cmap=cmap, norm=c_norm)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Radial position [µm]')
        ax.set_title(f'{lbl} Cathode')
        axes[f'{lbl}_cathode'] = ax

    # --------------------- shared error colour-bar --------------------
    left  = fig.axes[1].get_position().x0           # first error-axis
    right = fig.axes[n_pred].get_position().x1      # last error-axis
    top   = fig.axes[1].get_position().y1 + 0.015
    err_cax = fig.add_axes([left, top, right - left, 0.02])
    fig.colorbar(
        ScalarMappable(norm=err_norm, cmap=err_cmap),
        cax=err_cax, orientation='horizontal',
        label='|Error| scale [mol m$^{-3}$]'
    )
    err_cax.xaxis.set_label_position('top')
    err_cax.xaxis.set_ticks_position('top')

    # populate legend axis --------------------------------------------
    h_v, l_v = ax_V.get_legend_handles_labels()
    h_e, l_e = ax_Verr.get_legend_handles_labels()
    ax_legend.legend(h_v + h_e, l_v + l_e,
                     loc='center', frameon=False, fontsize=10)

    # --------------------- concentration colour-bars -----------------
    y_cbar = 0.07                                   # slightly higher now
    cbar_w, cbar_h = 0.35, 0.022

    ax_cbar_an = fig.add_axes([0.15, y_cbar, cbar_w, cbar_h])
    ax_cbar_ca = fig.add_axes([0.55, y_cbar, cbar_w, cbar_h])

    fig.colorbar(ScalarMappable(norm=a_norm, cmap=cmap),
                 cax=ax_cbar_an, orientation='horizontal',
                 label='Anode scale [mol m$^{-3}$]')
    fig.colorbar(ScalarMappable(norm=c_norm, cmap=cmap),
                 cax=ax_cbar_ca, orientation='horizontal',
                 label='Cathode scale [mol m$^{-3}$]')

    for cax in (ax_cbar_an, ax_cbar_ca):
        cax.xaxis.set_label_position('top')
        cax.xaxis.set_ticks_position('top')

    # final tidy -------------------------------------------------------
    fig.tight_layout(rect=[0, 0.1, 1, 1])

    return fig, axes

def create_plot_paper7(
        pred_sets,
        c_true_anode, c_true_cathode,
        func_I, V_true,
        t_max, Ran, Rca,
        pred_labels=None,
        cmap='viridis',
        err_cmap='plasma'):
    """
    Finalised 4-row layout
    ----------------------
      Row-0 : Anode |error| maps    + legend (right)
      Row-1 : True Anode & Anode predictions        + I(t)
      Row-2 : Cathode |error| maps
      Row-3 : True Cathode & Cathode predictions    + V(t) & |V-err|

    Two concentration colour-bars sit right under Row-3.
    """

    # ---------------------  basic sanity / defaults  -----------------
    n_pred = len(pred_sets)
    if pred_labels is None:
        pred_labels = [f'Pred-{i+1}' for i in range(n_pred)]
    if len(pred_labels) != n_pred:
        raise ValueError('pred_labels length must match pred_sets length')

    # meshes ----------------------------------------------------------
    R, T = c_true_anode.shape
    t = np.linspace(0, 1, T)
    r = np.linspace(0, 1, R)
    TT, RR = np.meshgrid(t, r)

    # colour-scales ---------------------------------------------------
    a_norm = Normalize(np.min([c_true_anode] + [p['anode']   for p in pred_sets]),
                       np.max([c_true_anode] + [p['anode']   for p in pred_sets]))
    c_norm = Normalize(np.min([c_true_cathode] + [p['cathode'] for p in pred_sets]),
                       np.max([c_true_cathode] + [p['cathode'] for p in pred_sets]))

    all_err = [np.abs(p['anode']   - c_true_anode)   for p in pred_sets] + \
              [np.abs(p['cathode'] - c_true_cathode) for p in pred_sets]
    err_norm = Normalize(np.min(all_err), np.max(all_err))

    # -----------------------  figure & grid  -------------------------
    n_cols = 1 + n_pred + 1                       # truth | preds | lines
    fig = plt.figure(figsize=(4.8 * n_cols, 11))
    gs  = gridspec.GridSpec(
        4, n_cols,
        width_ratios=[1] * (n_cols - 1) + [0.9],
        hspace=0.45, wspace=0.28, figure=fig
    )

    axes = {}

    # =============  ROW-0 : Anode error maps  ========================
    fig.add_subplot(gs[0, 0]).axis('off')         # no truth error map
    for col, (p, lbl) in enumerate(zip(pred_sets, pred_labels), start=1):
        ax = fig.add_subplot(gs[0, col])
        ax.contourf(TT * t_max, RR * Ran * 1e6,
                    np.abs(p['anode'] - c_true_anode),
                    levels=50, cmap=err_cmap, norm=err_norm)
        ax.set_xticks([])
        ax.set_title(f'{lbl} Anode |error|')
        axes[f'{lbl}_err_anode'] = ax

    ax_legend = fig.add_subplot(gs[0, -1])
    ax_legend.axis('off')
    axes['legend'] = ax_legend

    # =============  ROW-1 : True Anode + predictions  ================
    ax_true_an = fig.add_subplot(gs[1, 0])
    ax_true_an.contourf(TT * t_max, RR * Ran * 1e6,
                        c_true_anode, levels=50, cmap=cmap, norm=a_norm)
    ax_true_an.set_xlabel('Time [s]')
    ax_true_an.set_ylabel('Radial position [µm]')
    ax_true_an.set_title('True Anode\n[$mol\,m^{-3}$]')
    axes['true_anode'] = ax_true_an

    for col, (p, lbl) in enumerate(zip(pred_sets, pred_labels), start=1):
        ax = fig.add_subplot(gs[1, col])
        ax.contourf(TT * t_max, RR * Ran * 1e6,
                    p['anode'], levels=50, cmap=cmap, norm=a_norm)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Radial position [µm]')
        ax.set_title(f'{lbl} Anode')
        axes[f'{lbl}_anode'] = ax

    # right column rows 1-3 will host the line plots
    line_gs = gridspec.GridSpecFromSubplotSpec(
        3, 1, subplot_spec=gs[1:4, -1], hspace=0.4
    )

    ax_I = fig.add_subplot(line_gs[0])
    ax_I.plot(t * t_max, func_I)
    ax_I.set_title('Input Current')
    ax_I.set_xlabel('Time [s]')
    ax_I.set_ylabel('Current [A]')
    axes['current'] = ax_I

    # =============  ROW-2 : Cathode error maps  ======================
    fig.add_subplot(gs[2, 0]).axis('off')
    for col, (p, lbl) in enumerate(zip(pred_sets, pred_labels), start=1):
        ax = fig.add_subplot(gs[2, col])
        ax.contourf(TT * t_max, RR * Rca * 1e6,
                    np.abs(p['cathode'] - c_true_cathode),
                    levels=50, cmap=err_cmap, norm=err_norm)
        ax.set_xticks([])
        ax.set_title(f'{lbl} Cathode |error|')
        axes[f'{lbl}_err_cathode'] = ax

    # =============  ROW-3 : True Cathode + predictions  ==============
    ax_true_ca = fig.add_subplot(gs[3, 0])
    ax_true_ca.contourf(TT * t_max, RR * Rca * 1e6,
                        c_true_cathode, levels=50, cmap=cmap, norm=c_norm)
    ax_true_ca.set_xlabel('Time [s]')
    ax_true_ca.set_ylabel('Radial position [µm]')
    ax_true_ca.set_title('True Cathode\n[$mol\,m^{-3}$]')
    axes['true_cathode'] = ax_true_ca

    for col, (p, lbl) in enumerate(zip(pred_sets, pred_labels), start=1):
        ax = fig.add_subplot(gs[3, col])
        ax.contourf(TT * t_max, RR * Rca * 1e6,
                    p['cathode'], levels=50, cmap=cmap, norm=c_norm)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Radial position [µm]')
        ax.set_title(f'{lbl} Cathode')
        axes[f'{lbl}_cathode'] = ax

    # -----------  the rest of the line-plot column  ------------------
    ax_V = fig.add_subplot(line_gs[1])
    ax_V.plot(t * t_max, V_true, lw=2, label='Ground Truth')
    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i, (p, lbl) in enumerate(zip(pred_sets, pred_labels)):
        ax_V.plot(t * t_max, p['V'],
                  linestyle='--', color=cycle[i % len(cycle)], label=lbl)
    ax_V.set_title('Voltage')
    ax_V.set_xlabel('Time [s]')
    ax_V.set_ylabel('Cell Voltage [V]')
    axes['voltage'] = ax_V

    ax_Verr = fig.add_subplot(line_gs[2])
    for i, (p, lbl) in enumerate(zip(pred_sets, pred_labels)):
        ax_Verr.plot(t * t_max,
                     np.abs(p['V'] - V_true) * 1e3,
                     color=cycle[i % len(cycle)],
                     label=f'{lbl} |V-err|')
    ax_Verr.set_title('Voltage Error')
    ax_Verr.set_xlabel('Time [s]')
    ax_Verr.set_ylabel('Error [mV]')
    axes['voltage_error'] = ax_Verr

    # ----------------  shared |error| colour-bar  --------------------
    left  = fig.axes[1].get_position().x0
    right = fig.axes[n_pred].get_position().x1
    top   = fig.axes[1].get_position().y1 + 0.012
    cax_err = fig.add_axes([left, top, right - left, 0.018])
    fig.colorbar(ScalarMappable(norm=err_norm, cmap=err_cmap),
                 cax=cax_err, orientation='horizontal',
                 label='|Error| scale [mol m$^{-3}$]')
    cax_err.xaxis.set_label_position('top')
    cax_err.xaxis.set_ticks_position('top')

    # legend (top-right) ---------------------------------------------
    h_v, l_v = ax_V.get_legend_handles_labels()
    h_e, l_e = ax_Verr.get_legend_handles_labels()
    ax_legend.legend(h_v + h_e, l_v + l_e,
                     loc='center', frameon=False, fontsize=10)

    # ----------------  concentration colour-bars  -------------------
    y_cbar = 0.09                                   # ← higher than before
    bar_w, bar_h = 0.35, 0.022
    cbar_an_ax = fig.add_axes([0.15, y_cbar, bar_w, bar_h])
    cbar_ca_ax = fig.add_axes([0.55, y_cbar, bar_w, bar_h])

    fig.colorbar(ScalarMappable(norm=a_norm, cmap=cmap),
                 cax=cbar_an_ax, orientation='horizontal',
                 label='Anode scale [mol m$^{-3}$]')
    fig.colorbar(ScalarMappable(norm=c_norm, cmap=cmap),
                 cax=cbar_ca_ax, orientation='horizontal',
                 label='Cathode scale [mol m$^{-3}$]')

    for cax in (cbar_an_ax, cbar_ca_ax):
        cax.xaxis.set_label_position('top')
        cax.xaxis.set_ticks_position('top')

    # ---------------------------  tidy  ------------------------------
    fig.tight_layout(rect=[0, 0.12, 1, 1])
    return fig, axes


def create_plot_paper8(
        pred_sets,
        c_true_anode, c_true_cathode,
        func_I, V_true,
        t_max, Ran, Rca,
        pred_labels=None,
        cmap='viridis',
        err_cmap='plasma'):
    """
    Final 4-row layout
    ------------------
      Row-0 : Anode |error| maps               + legend (right-most)
      Row-1 : Cathode |error| maps             + I(t)
      Row-2 : True Anode & Anode predictions   + V(t)
      Row-3 : True Cathode & Cath. predictions + |V-err|

    Two global concentration colour-bars sit just below Row-3.
    """

    # ──────────────────────── sanity / defaults ──────────────────────
    n_pred = len(pred_sets)
    if pred_labels is None:
        pred_labels = [f'Pred-{i+1}' for i in range(n_pred)]
    if len(pred_labels) != n_pred:
        raise ValueError('pred_labels length must match pred_sets length')

    # meshes -----------------------------------------------------------
    R, T = c_true_anode.shape
    t = np.linspace(0, 1, T)
    r = np.linspace(0, 1, R)
    TT, RR = np.meshgrid(t, r)

    # colour-scales ----------------------------------------------------
    a_norm = Normalize(np.min([c_true_anode] + [p['anode']   for p in pred_sets]),
                       np.max([c_true_anode] + [p['anode']   for p in pred_sets]))
    c_norm = Normalize(np.min([c_true_cathode] + [p['cathode'] for p in pred_sets]),
                       np.max([c_true_cathode] + [p['cathode'] for p in pred_sets]))

    all_err = ([np.abs(p['anode']   - c_true_anode)   for p in pred_sets] +
               [np.abs(p['cathode'] - c_true_cathode) for p in pred_sets])
    err_norm = Normalize(np.min(all_err), np.max(all_err))

    # figure & grid ----------------------------------------------------
    n_cols = 1 + n_pred + 1                      # truth | preds | time-series
    fig = plt.figure(figsize=(4.8 * n_cols, 11))
    gs  = gridspec.GridSpec(
        4, n_cols,
        width_ratios=[1] * (n_cols - 1) + [0.9],
        hspace=0.45, wspace=0.28, figure=fig
    )

    axes = {}

    # ===========  ROW-0 : Anode |error| maps & legend  ===============
    fig.add_subplot(gs[2, 0]).axis('off')        # blank (truth column)

    for col, (p, lbl) in enumerate(zip(pred_sets, pred_labels), start=1):
        ax = fig.add_subplot(gs[3, col])
        ax.contourf(TT * t_max, RR * Ran * 1e6,
                    np.abs(p['anode'] - c_true_anode),
                    levels=50, cmap=err_cmap, norm=err_norm)
        ax.set_xticks([])
        ax.set_title(f'{lbl} Anode |error|')
        axes[f'{lbl}_err_anode'] = ax

    ax_legend = fig.add_subplot(gs[3, -1])
    ax_legend.axis('off')
    axes['legend'] = ax_legend

    # ===========  ROW-1 : Cathode |error| maps & I(t)  ===============
    fig.add_subplot(gs[3, 0]).axis('off')        # blank

    for col, (p, lbl) in enumerate(zip(pred_sets, pred_labels), start=1):
        ax = fig.add_subplot(gs[2, col])
        ax.contourf(TT * t_max, RR * Rca * 1e6,
                    np.abs(p['cathode'] - c_true_cathode),
                    levels=50, cmap=err_cmap, norm=err_norm)
        ax.set_xticks([])
        ax.set_title(f'{lbl} Cathode |error|')
        axes[f'{lbl}_err_cathode'] = ax

    ax_I = fig.add_subplot(gs[0, -1])
    ax_I.plot(t * t_max, func_I)
    ax_I.set_title('Input Current')
    ax_I.set_xlabel('Time [s]')
    ax_I.set_ylabel('Current [A]')
    axes['current'] = ax_I

    # ===========  ROW-2 : True Anode & predictions  + V(t) ===========
    ax_true_an = fig.add_subplot(gs[0, 0])
    ax_true_an.contourf(TT * t_max, RR * Ran * 1e6,
                        c_true_anode, levels=150, cmap=cmap, norm=a_norm)
    ax_true_an.set_xlabel('Time [s]')
    ax_true_an.set_ylabel('Radial position [µm]')
    ax_true_an.set_title('True Anode\n[$mol\,m^{-3}$]')
    axes['true_anode'] = ax_true_an

    for col, (p, lbl) in enumerate(zip(pred_sets, pred_labels), start=1):
        ax = fig.add_subplot(gs[0, col])
        ax.contourf(TT * t_max, RR * Ran * 1e6,
                    p['anode'], levels=150, cmap=cmap, norm=a_norm)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Radial position [µm]')
        ax.set_title(f'{lbl} Anode')
        axes[f'{lbl}_anode'] = ax

    ax_V = fig.add_subplot(gs[1, -1])
    ax_V.plot(t * t_max, V_true, lw=2, label='Ground Truth')
    colour_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i, (p, lbl) in enumerate(zip(pred_sets, pred_labels)):
        ax_V.plot(t * t_max, p['V'],
                  linestyle='--', color=colour_cycle[i % len(colour_cycle)],
                  label=lbl)
    ax_V.set_title('Voltage')
    ax_V.set_xlabel('Time [s]')
    ax_V.set_ylabel('Cell Voltage [V]')
    axes['voltage'] = ax_V

    # ===========  ROW-3 : True Cathode & predictions + |V-err| =======
    ax_true_ca = fig.add_subplot(gs[1, 0])
    ax_true_ca.contourf(TT * t_max, RR * Rca * 1e6,
                        c_true_cathode, levels=150, cmap=cmap, norm=c_norm)
    ax_true_ca.set_xlabel('Time [s]')
    ax_true_ca.set_ylabel('Radial position [µm]')
    ax_true_ca.set_title('True Cathode\n[$mol\,m^{-3}$]')
    axes['true_cathode'] = ax_true_ca

    for col, (p, lbl) in enumerate(zip(pred_sets, pred_labels), start=1):
        ax = fig.add_subplot(gs[1, col])
        ax.contourf(TT * t_max, RR * Rca * 1e6,
                    p['cathode'], levels=150, cmap=cmap, norm=c_norm)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Radial position [µm]')
        ax.set_title(f'{lbl} Cathode')
        axes[f'{lbl}_cathode'] = ax

    ax_Verr = fig.add_subplot(gs[2, -1])
    for i, (p, lbl) in enumerate(zip(pred_sets, pred_labels)):
        ax_Verr.plot(t * t_max,
                     np.abs(p['V'] - V_true) * 1e3,
                     color=colour_cycle[i % len(colour_cycle)],
                     label=f'{lbl} |V-err|')
    ax_Verr.set_title('Voltage Error')
    ax_Verr.set_xlabel('Time [s]')
    ax_Verr.set_ylabel('Error [mV]')
    axes['voltage_error'] = ax_Verr

    # ───────────── shared |error| colour-bar  (top centre) ───────────
    left  = fig.axes[1].get_position().x0               # 1st error axis
    right = fig.axes[n_pred].get_position().x1          # last error axis
    top   = fig.axes[1].get_position().y1 + 0.015
    cax_err = fig.add_axes([left, top, right - left, 0.02])
    fig.colorbar(ScalarMappable(norm=err_norm, cmap=err_cmap),
                 cax=cax_err, orientation='horizontal',
                 label='|Error| scale [mol m$^{-3}$]')
    cax_err.xaxis.set_label_position('top')
    cax_err.xaxis.set_ticks_position('top')

    # populate legend cell -------------------------------------------
    h_v, l_v = ax_V.get_legend_handles_labels()
    h_e, l_e = ax_Verr.get_legend_handles_labels()
    ax_legend.legend(h_v + h_e, l_v + l_e,
                     loc='center', frameon=False, fontsize=10)

    # ───────────── two concentration colour-bars (bottom) ───────────
    y_cbar = 0.07
    bar_w, bar_h = 0.35, 0.022
    ax_cbar_an = fig.add_axes([0.15, y_cbar, bar_w, bar_h])
    ax_cbar_ca = fig.add_axes([0.55, y_cbar, bar_w, bar_h])

    fig.colorbar(ScalarMappable(norm=a_norm, cmap=cmap),
                 cax=ax_cbar_an, orientation='horizontal',
                 label='Anode scale [mol m$^{-3}$]')
    fig.colorbar(ScalarMappable(norm=c_norm, cmap=cmap),
                 cax=ax_cbar_ca, orientation='horizontal',
                 label='Cathode scale [mol m$^{-3}$]')

    for cax in (ax_cbar_an, ax_cbar_ca):
        cax.xaxis.set_label_position('top')
        cax.xaxis.set_ticks_position('top')

    # final tidy-up ---------------------------------------------------
    fig.tight_layout(rect=[0, 0.1, 1, 1])
    return fig, axes


def create_plot_paper9(
        pred_sets,
        c_true_anode, c_true_cathode,
        func_I, V_true,
        t_max, Ran, Rca,
        pred_labels=None,
        cmap='viridis',
        err_cmap='plasma'):
    """
    4-row composite

      Row-0 : (blank) │ Anode-error maps │ legend
      Row-1 : (blank) │ Cathode-error maps │ I(t)
      Row-2 : True-An │ Anode predictions  │ V(t)
      Row-3 : True-Ca │ Cathode predictions│ |V-err|

    * ONE concentration bar (shared anode+cathode) at the bottom.
    * TWO error bars (anode, cathode) stacked in the blank gs[3,0] cell.
    """

    # ───── sanity / defaults ─────────────────────────────────────────
    n_pred = len(pred_sets)
    if pred_labels is None:
        pred_labels = [f'Pred-{i+1}' for i in range(n_pred)]
    if len(pred_labels) != n_pred:
        raise ValueError('pred_labels length must match pred_sets length')

    # meshes ----------------------------------------------------------
    R, T = c_true_anode.shape
    t = np.linspace(0, 1, T)
    r = np.linspace(0, 1, R)
    TT, RR = np.meshgrid(t, r)

    # ───── concentration norm (single) ───────────────────────────────
    conc_arrays = ([c_true_anode, c_true_cathode] +
                   [p['anode'] for p in pred_sets] +
                   [p['cathode'] for p in pred_sets])
    conc_norm = Normalize(np.min(conc_arrays), np.max(conc_arrays))

    a_norm = Normalize(np.min([c_true_anode] + [p['anode']   for p in pred_sets]),
                       np.max([c_true_anode] + [p['anode']   for p in pred_sets]))
    c_norm = Normalize(np.min([c_true_cathode] + [p['cathode'] for p in pred_sets]),
                       np.max([c_true_cathode] + [p['cathode'] for p in pred_sets]))



    # ───── separate error norms ──────────────────────────────────────
    an_err_arrays = [np.abs(p['anode']   - c_true_anode)   for p in pred_sets]
    ca_err_arrays = [np.abs(p['cathode'] - c_true_cathode) for p in pred_sets]
    an_err_norm = Normalize(np.min(an_err_arrays), np.max(an_err_arrays))
    ca_err_norm = Normalize(np.min(ca_err_arrays), np.max(ca_err_arrays))

    # figure & GridSpec ----------------------------------------------
    n_cols = 1 + n_pred + 1                      # truth | preds | series
    fig = plt.figure(figsize=(4.8 * n_cols, 11))
    gs  = gridspec.GridSpec(
        4, n_cols,
        width_ratios=[1]*(n_cols-1) + [0.9],
        hspace=0.45, wspace=0.28, figure=fig
    )
    axes = {}

    # ───── two error colour-bars inside blank gs[3,0] cell ───────────
    dummy_ax = fig.add_subplot(gs[3, 0])          # the otherwise-blank slot
    bbox = dummy_ax.get_position()                # its rectangle on canvas
    dummy_ax.remove()                             # keep the cell visually empty

    bar_h   = 0.013                               # height of each bar
    spacing = 0.05                                # vertical gap between bars

    model_err_norm = {str(lbl): Normalize(vmin=0, vmax=1) for lbl in pred_labels}
    for i, (p, lbl) in enumerate(zip(reversed(pred_sets), reversed(pred_labels))):
        # model-specific error range (max over anode & cathode errors)
        err_max = np.max([np.abs(p['anode'] - c_true_anode),
                        np.abs(p['cathode'] - c_true_cathode)])
        # print(f'Error max for {lbl}: {err_max:.3e}')
        model_err_norm[lbl] = Normalize(vmin=0, vmax=err_max)

        cax = fig.add_axes([bbox.x0,
                            bbox.y0 + i*(bar_h + spacing),
                            bbox.width, bar_h])
        # cb = fig.colorbar(ScalarMappable(norm=model_err_norm[lbl], cmap=err_cmap),
        #                 cax=cax, orientation='horizontal',
        #                 label=str(lbl) + ' |err| [$\mathrm{mol\,m^{-3}}$]')

        # # ── label on top, numbers underneath ──────────────────────────────
        # cb.ax.xaxis.set_label_position('top')        # title above the bar
        # cb.ax.xaxis.set_ticks_position('bottom')     # ticks on bottom edge
        # cb.ax.tick_params(axis='x',
        #                 bottom=True, top=False,    # show ticks only below
        #                 labelbottom=True, labeltop=False)  # numbers only below
        
        # 1.  Use the title slot (works with every Matplotlib version)
        cb = fig.colorbar(ScalarMappable(norm=model_err_norm[lbl], cmap=err_cmap), cax=cax,
                        orientation='horizontal')
        cb.ax.set_title(str(lbl) + r' $|err|\;[\mathrm{mol\,m^{-3}}]$')
        cb.ax.xaxis.set_ticks_position('bottom')          # ticks only below
        cb.ax.tick_params(bottom=True, top=False,
                        labelbottom=True, labeltop=False)


    # ===========  ROW-3 : anode-error maps + legend  ================
    fig.add_subplot(gs[2, 0]).axis('off')
    for col, (p, lbl) in enumerate(zip(pred_sets, pred_labels), start=1):
        ax = fig.add_subplot(gs[2, col])
        ax.contourf(TT*t_max, RR*Ran*1e6,
                    np.abs(p['anode'] - c_true_anode),
                    levels=50, cmap=err_cmap, norm=model_err_norm[lbl])
        ax.set_xlabel('Time [$s$]')
        ax.set_ylabel('Radial position [µm]')
        ax.set_title(f'{lbl} Anode $|err|$')
        axes[f'{lbl}_err_anode'] = ax

    ax_legend = fig.add_subplot(gs[3, -1])
    ax_legend.axis('off')
    axes['legend'] = ax_legend

    # ===========  ROW-4 : cathode-error maps + I(t) ==================
    fig.add_subplot(gs[3, 0]).axis('off')
    for col, (p, lbl) in enumerate(zip(pred_sets, pred_labels), start=1):
        ax = fig.add_subplot(gs[3, col])
        ax.contourf(TT*t_max, RR*Rca*1e6,
                    np.abs(p['cathode'] - c_true_cathode),
                    levels=50, cmap=err_cmap, norm=model_err_norm[lbl])
        #ax.set_xticks([])
        ax.set_xlabel('Time [$s$]')
        ax.set_ylabel('Radial position [µm]')
        ax.set_title(f'{lbl} Cathode $|err|$')
        axes[f'{lbl}_err_cathode'] = ax

    ax_I = fig.add_subplot(gs[0, -1])
    ax_I.plot(t*t_max, func_I)
    ax_I.set_title('Input Current')
    ax_I.set_xlabel('Time [$s$]')
    ax_I.set_ylabel('Current [A]')
    axes['current'] = ax_I

    # ===========  ROW-1 : True-An + preds   +  V(t)  ================
    ax_true_an = fig.add_subplot(gs[0, 0])
    ax_true_an.contourf(TT*t_max, RR*Ran*1e6,
                        c_true_anode, levels=50, cmap=cmap, norm=conc_norm)
    ax_true_an.set_xlabel('Time [$s$]')
    ax_true_an.set_ylabel('Radial position [µm]')
    ax_true_an.set_title('Ground Truth Anode')
    axes['true_anode'] = ax_true_an

    for col, (p, lbl) in enumerate(zip(pred_sets, pred_labels), start=1):
        ax = fig.add_subplot(gs[0, col])
        ax.contourf(TT*t_max, RR*Ran*1e6,
                    p['anode'], levels=50, cmap=cmap, norm=conc_norm)
        ax.set_xlabel('Time [$s$]')
        ax.set_ylabel('Radial position [µm]')
        ax.set_title(f'{lbl} Anode')
        axes[f'{lbl}_anode'] = ax

    ax_V = fig.add_subplot(gs[1, -1])
    ax_V.plot(t*t_max, V_true, lw=2, label='Ground Truth')
    colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i, (p, lbl) in enumerate(zip(pred_sets, pred_labels)):
        ax_V.plot(t*t_max, p['V'],
                  linestyle='--', color=colours[i % len(colours)],
                  label=lbl)
    ax_V.set_title('Voltage')
    ax_V.set_xlabel('Time [$s$]')
    ax_V.set_ylabel('Cell Voltage [V]')
    axes['voltage'] = ax_V

    # ===========  ROW-2 : True-Ca + preds   +  |V-err|  =============
    ax_true_ca = fig.add_subplot(gs[1, 0])
    ax_true_ca.contourf(TT*t_max, RR*Rca*1e6,
                        c_true_cathode, levels=50, cmap=cmap,
                        norm=conc_norm)
    ax_true_ca.set_xlabel('Time [$s$]')
    ax_true_ca.set_ylabel('Radial position [µm]')
    ax_true_ca.set_title('Ground Truth Cathode')
    axes['true_cathode'] = ax_true_ca

    for col, (p, lbl) in enumerate(zip(pred_sets, pred_labels), start=1):
        ax = fig.add_subplot(gs[1, col])
        ax.contourf(TT*t_max, RR*Rca*1e6,
                    p['cathode'], levels=50, cmap=cmap, norm=conc_norm)
        ax.set_xlabel('Time [$s$]')
        ax.set_ylabel('Radial position [µm]')
        ax.set_title(f'{lbl} Cathode')
        axes[f'{lbl}_cathode'] = ax

    ax_Verr = fig.add_subplot(gs[2, -1])
    for i, (p, lbl) in enumerate(zip(pred_sets, pred_labels)):
        ax_Verr.plot(t*t_max,
                     np.abs(p['V'] - V_true)*1e3,
                     color=colours[i % len(colours)],
                     label=f'{lbl} |V-err|')
    ax_Verr.set_title('Voltage Error')
    ax_Verr.set_xlabel('Time [$s$]')
    ax_Verr.set_ylabel('Error [mV]')
    axes['voltage_error'] = ax_Verr

    # tmp = fig.add_subplot(gs[2, 0])       # placeholder axis
    # bbox = tmp.get_position()             # [x0, y0, width, height]
    # tmp.remove()    
    # half_w = (bbox.width - 0.01) / 2                 # small gap in between
    # bar_h  = bbox.height * 0.30
    
    # for i, part_lbl in enumerate(['Low range', 'High range']):
    #     ax_split = fig.add_axes([bbox.x0 + i*(half_w + 0.01),
    #                              bbox.y0 + (bbox.height - bar_h)/2,
    #                              half_w,
    #                              bar_h])
    #     fig.colorbar(ScalarMappable(norm=conc_norm, cmap=cmap),
    #                  cax=ax_split, orientation='horizontal',
    #                  label=f'Concentration ({part_lbl})')
    #     ax_split.xaxis.set_label_position('top')
    #     ax_split.xaxis.set_ticks_position('top')

    tmp_ax = fig.add_subplot(gs[2, 0])      # placeholder for the blank cell
    bbox   = tmp_ax.get_position()          # [x0, y0, width, height]
    tmp_ax.remove()

    gap    = 0.075                       # vertical gap between the bars (in figure units)
    bar_h  = 0.02 #(bbox.height - gap) / 2        # each bar gets half of the cell height minus the gap

    # for i, part_lbl in enumerate(['Low range', 'High range']):   # i = 0 (top), 1 (bottom)
    #     y0 = bbox.y0 + bbox.height - (i + 1) * bar_h - i * gap   # top bar first
    #     cax = fig.add_axes([bbox.x0,
    #                         y0,
    #                         bbox.width,
    #                         bar_h])
    #     fig.colorbar(ScalarMappable(norm=a_norm, cmap=cmap),
    #                 cax=cax, orientation='horizontal',
    #                 label=f'Concentration ({part_lbl})')
    #     cax.xaxis.set_label_position('top')
    #     cax.xaxis.set_ticks_position('top')

    y0 = bbox.y0 + bbox.height - (0 + 1.5) * bar_h - 0 * gap  # top bar first
    cax = fig.add_axes([bbox.x0,
                        y0,
                        bbox.width,
                        bar_h])
    cb = fig.colorbar(ScalarMappable(norm=a_norm, cmap=cmap),
                    cax=cax, orientation='horizontal')

    # ── put label above bar, ticks below ───────────────────────────────
    cb.ax.xaxis.set_label_position('top')
    cb.ax.xaxis.set_ticks_position('bottom')

    # add / update the label with a bit of extra gap
    # cb.set_label(r'Concentration Anode [$\mathrm{mol\,m^{-3}}$]')
    cb.ax.set_title(r'Concentration Anode [$\mathrm{mol\,m^{-3}}$]')

    y0 = bbox.y0 + bbox.height - (1 + 1.5) * bar_h - 1 * gap  # top bar first
    cax = fig.add_axes([bbox.x0,
                        y0,
                        bbox.width,
                        bar_h])
    cb = fig.colorbar(ScalarMappable(norm=c_norm, cmap=cmap),
                    cax=cax, orientation='horizontal')

    # ── put label above bar, ticks below ───────────────────────────────
    cb.ax.xaxis.set_label_position('top')
    cb.ax.xaxis.set_ticks_position('bottom')

    # add / update the label with a bit of extra gap
    # cb.set_label(r'Concentration Cathode [$\mathrm{mol\,m^{-3}}$]')
    cb.ax.set_title(r'Concentration Cathode [$\mathrm{mol\,m^{-3}}$]')    



    # # ───── two error colour-bars inside blank gs[3,0] cell ───────────
    # dummy_ax = fig.add_subplot(gs[3, 0])          # the otherwise-blank slot
    # bbox = dummy_ax.get_position()                # its rectangle on canvas
    # dummy_ax.remove()                             # keep the cell visually empty

    # bar_h   = 0.013                               # height of each bar
    # spacing = 0.05                                # vertical gap between bars
    # for i, (p, lbl) in reversed(enumerate(zip(pred_sets, pred_labels))):
    #     # model-specific error range (max over anode & cathode errors)
    #     err_max = np.max([np.abs(p['anode'] - c_true_anode),
    #                     np.abs(p['cathode'] - c_true_cathode)])
    #     print(f'Error max for {lbl}: {err_max:.3e}')
    #     model_err_norm = Normalize(vmin=0, vmax=err_max)

    #     cax = fig.add_axes([bbox.x0,
    #                         bbox.y0 + i*(bar_h + spacing),
    #                         bbox.width, bar_h])
    #     fig.colorbar(ScalarMappable(norm=model_err_norm, cmap=err_cmap),
    #                 cax=cax, orientation='horizontal',
    #                 label=f'{lbl} |err| [mol m⁻³]')
    #     cax.xaxis.set_label_position('top')
    #     cax.xaxis.set_ticks_position('bottom')

    # legend ----------------------------------------------------------
    # h_v, l_v = ax_V.get_legend_handles_labels()
    # h_e, l_e = ax_Verr.get_legend_handles_labels()
    # ax_legend.legend(h_v + h_e, l_v + l_e,
    #                  loc='center', frameon=False, fontsize=10)

    proxy_handles = [Line2D([], [], lw=2, color='k', label='Ground Truth')]  # keep GT

    for i, lbl in enumerate(pred_labels):
        proxy_handles.append(
            Line2D([], [], lw=2, color=colours[i % len(colours)], label=lbl)
        )

    ax_legend.legend(proxy_handles,
                    [h.get_label() for h in proxy_handles],
                    loc='center', frameon=False, fontsize=10)

    # # ───── single concentration bar (bottom centre)  ─────────────────
    # cbar_ax = fig.add_axes([0.25, 0.07, 0.5, 0.022])
    # fig.colorbar(ScalarMappable(norm=conc_norm, cmap=cmap),
    #              cax=cbar_ax, orientation='horizontal',
    #              label='Concentration scale [mol m⁻³]')
    # cbar_ax.xaxis.set_label_position('top')
    # cbar_ax.xaxis.set_ticks_position('bottom')

    radial_axes = [ax_true_an, ax_true_ca]                   # truth columns
    radial_axes += [axes[f'{lbl}_anode']   for lbl in pred_labels]
    radial_axes += [axes[f'{lbl}_cathode'] for lbl in pred_labels]
    radial_axes += [ax for ax in fig.axes if 'err' in ax.get_title()]  # error maps

    fig.align_ylabels(radial_axes)     

    # tidy ------------------------------------------------------------
    fig.tight_layout(rect=[0, 0.1, 1, 1])
    return fig, axes



def create_plot_paper10(
        pred_sets,
        c_true_anode, c_true_cathode,
        func_I, V_true,
        t_max, Ran, Rca,
        pred_labels=None,
        cmap='viridis',
        err_cmap='plasma'):

    # ───── sanity / defaults ─────────────────────────────────────────
    n_pred = len(pred_sets)
    if pred_labels is None:
        pred_labels = [f'Pred-{i+1}' for i in range(n_pred)]
    if len(pred_labels) != n_pred:
        raise ValueError('pred_labels length must match pred_sets length')

    # ───── global colour palette for every *line* (GT + preds) ──────
    gt_colour = 'k'                                                     # keep Ground Truth black
    palette   = plt.rcParams['axes.prop_cycle'].by_key()['color']       # default matplotlib cycle
    # pred_colours = [palette[i % len(palette)] for i in range(n_pred)]   # as many as we need
    # label_colour = {'Ground Truth': gt_colour, **dict(zip(pred_labels, pred_colours))}

    pred_colours = sns.color_palette("colorblind", n_pred)   # pick one row above

    label_colour = {'Ground Truth': gt_colour,
                **dict(zip(pred_labels, pred_colours))}

    # meshes ----------------------------------------------------------
    R, T = c_true_anode.shape
    t = np.linspace(0, 1, T)
    r = np.linspace(0, 1, R)
    TT, RR = np.meshgrid(t, r)

    # ───── concentration norm (single) ───────────────────────────────
    conc_arrays = ([c_true_anode, c_true_cathode] +
                   [p['anode'] for p in pred_sets] +
                   [p['cathode'] for p in pred_sets])
    conc_norm = Normalize(np.min(conc_arrays), np.max(conc_arrays))

    a_norm = Normalize(np.min([c_true_anode] + [p['anode']   for p in pred_sets]),
                       np.max([c_true_anode] + [p['anode']   for p in pred_sets]))
    c_norm = Normalize(np.min([c_true_cathode] + [p['cathode'] for p in pred_sets]),
                       np.max([c_true_cathode] + [p['cathode'] for p in pred_sets]))

    # ───── separate error norms ──────────────────────────────────────
    an_err_arrays = [np.abs(p['anode']   - c_true_anode)   for p in pred_sets]
    ca_err_arrays = [np.abs(p['cathode'] - c_true_cathode) for p in pred_sets]
    an_err_norm = Normalize(np.min(an_err_arrays), np.max(an_err_arrays))
    ca_err_norm = Normalize(np.min(ca_err_arrays), np.max(ca_err_arrays))

    # figure & GridSpec ----------------------------------------------
    n_cols = 1 + n_pred + 1                      # truth | preds | series
    fig = plt.figure(figsize=(4.8 * n_cols, 11))
    gs  = gridspec.GridSpec(
        4, n_cols,
        width_ratios=[1]*(n_cols-1) + [0.9],
        hspace=0.45, wspace=0.28, figure=fig
    )
    axes = {}

    # ───── two error colour-bars inside blank gs[3,0] cell ───────────
    dummy_ax = fig.add_subplot(gs[3, 0])
    bbox = dummy_ax.get_position()
    dummy_ax.remove()

    bar_h, spacing = 0.013, 0.05
    model_err_norm = {}

    for i, (p, lbl) in enumerate(zip(reversed(pred_sets), reversed(pred_labels))):
        err_max = np.max([np.abs(p['anode'] - c_true_anode),
                          np.abs(p['cathode'] - c_true_cathode)])
        model_err_norm[lbl] = Normalize(vmin=0, vmax=err_max)

        cax = fig.add_axes([bbox.x0,
                            bbox.y0 + i*(bar_h + spacing),
                            bbox.width, bar_h])

        cb = fig.colorbar(ScalarMappable(norm=model_err_norm[lbl], cmap=err_cmap),
                          cax=cax, orientation='horizontal')
        cb.ax.set_title(fr'{lbl} $|err|\;[\mathrm{{mol\,m^{{-3}}}}]$')
        cb.ax.xaxis.set_ticks_position('bottom')
        cb.ax.tick_params(bottom=True, top=False,
                          labelbottom=True, labeltop=False)

    # ===========  ROW-3 : anode-error maps + legend  ================
    fig.add_subplot(gs[2, 0]).axis('off')
    for col, (p, lbl) in enumerate(zip(pred_sets, pred_labels), start=1):
        ax = fig.add_subplot(gs[2, col])
        ax.contourf(TT*t_max, RR*Ran*1e6,
                    np.abs(p['anode'] - c_true_anode),
                    levels=50, cmap=err_cmap, norm=model_err_norm[lbl])
        ax.set_xlabel('Time [$s$]')
        ax.set_ylabel('Radial position [µm]')
        ax.set_title(f'{lbl} Anode $|err|$')
        axes[f'{lbl}_err_anode'] = ax

    ax_legend = fig.add_subplot(gs[3, -1])
    ax_legend.axis('off')
    axes['legend'] = ax_legend

    # ===========  ROW-4 : cathode-error maps + I(t) ==================
    fig.add_subplot(gs[3, 0]).axis('off')
    for col, (p, lbl) in enumerate(zip(pred_sets, pred_labels), start=1):
        ax = fig.add_subplot(gs[3, col])
        ax.contourf(TT*t_max, RR*Rca*1e6,
                    np.abs(p['cathode'] - c_true_cathode),
                    levels=50, cmap=err_cmap, norm=model_err_norm[lbl])
        ax.set_xlabel('Time [$s$]')
        ax.set_ylabel('Radial position [µm]')
        ax.set_title(f'{lbl} Cathode $|err|$')
        axes[f'{lbl}_err_cathode'] = ax

    # ---------- 1-D LINE PLOTS (new palette applied here) ------------
    ax_I = fig.add_subplot(gs[0, -1])
    ax_I.plot(t*t_max, func_I, color=label_colour['Ground Truth'])
    ax_I.set_title('Input Current')
    ax_I.set_xlabel('Time [$s$]')
    ax_I.set_ylabel('Current [A]')
    axes['current'] = ax_I

    # ===========  ROW-1 : True-An + preds   +  V(t)  ================
    ax_true_an = fig.add_subplot(gs[0, 0])
    ax_true_an.contourf(TT*t_max, RR*Ran*1e6,
                        c_true_anode, levels=50, cmap=cmap, norm=conc_norm)
    ax_true_an.set_xlabel('Time [$s$]')
    ax_true_an.set_ylabel('Radial position [µm]')
    ax_true_an.set_title('Ground Truth Anode')
    axes['true_anode'] = ax_true_an

    for col, (p, lbl) in enumerate(zip(pred_sets, pred_labels), start=1):
        ax = fig.add_subplot(gs[0, col])
        ax.contourf(TT*t_max, RR*Ran*1e6,
                    p['anode'], levels=50, cmap=cmap, norm=conc_norm)
        ax.set_xlabel('Time [$s$]')
        ax.set_ylabel('Radial position [µm]')
        ax.set_title(f'{lbl} Anode')
        axes[f'{lbl}_anode'] = ax

    ax_V = fig.add_subplot(gs[1, -1])
    ax_V.plot(t*t_max, V_true, lw=2,
              color=label_colour['Ground Truth'], label='Ground Truth')
    for p, lbl in zip(pred_sets, pred_labels):
        ax_V.plot(t*t_max, p['V'], linestyle='--',
                  color=label_colour[lbl], label=lbl)
    ax_V.set_title('Voltage')
    ax_V.set_xlabel('Time [$s$]')
    ax_V.set_ylabel('Cell Voltage [V]')
    axes['voltage'] = ax_V

    # ===========  ROW-2 : True-Ca + preds   +  |V-err|  =============
    ax_true_ca = fig.add_subplot(gs[1, 0])
    ax_true_ca.contourf(TT*t_max, RR*Rca*1e6,
                        c_true_cathode, levels=50, cmap=cmap,
                        norm=conc_norm)
    ax_true_ca.set_xlabel('Time [$s$]')
    ax_true_ca.set_ylabel('Radial position [µm]')
    ax_true_ca.set_title('Ground Truth Cathode')
    axes['true_cathode'] = ax_true_ca

    for col, (p, lbl) in enumerate(zip(pred_sets, pred_labels), start=1):
        ax = fig.add_subplot(gs[1, col])
        ax.contourf(TT*t_max, RR*Rca*1e6,
                    p['cathode'], levels=50, cmap=cmap, norm=conc_norm)
        ax.set_xlabel('Time [$s$]')
        ax.set_ylabel('Radial position [µm]')
        ax.set_title(f'{lbl} Cathode')
        axes[f'{lbl}_cathode'] = ax

    ax_Verr = fig.add_subplot(gs[2, -1])
    for p, lbl in zip(pred_sets, pred_labels):
        ax_Verr.plot(t*t_max,
                     np.abs(p['V'] - V_true)*1e3,
                     color=label_colour[lbl],
                     label=f'{lbl} |V-err|')
    ax_Verr.set_title('Voltage Error')
    ax_Verr.set_xlabel('Time [$s$]')
    ax_Verr.set_ylabel('Error [mV]')
    axes['voltage_error'] = ax_Verr

    # concentration colour-bars (unchanged) --------------------------
    tmp_ax = fig.add_subplot(gs[2, 0])
    bbox   = tmp_ax.get_position()
    tmp_ax.remove()

    gap, bar_h = 0.075, 0.02

    # anode bar
    y0 = bbox.y0 + bbox.height - (0 + 1.5) * bar_h
    cax = fig.add_axes([bbox.x0, y0, bbox.width, bar_h])
    cb = fig.colorbar(ScalarMappable(norm=a_norm, cmap=cmap),
                      cax=cax, orientation='horizontal')
    cb.ax.set_title(r'Concentration Anode [$\mathrm{mol\,m^{-3}}$]')
    cb.ax.xaxis.set_label_position('top')
    cb.ax.xaxis.set_ticks_position('bottom')

    # cathode bar
    y0 = bbox.y0 + bbox.height - (1 + 1.5) * bar_h - gap
    cax = fig.add_axes([bbox.x0, y0, bbox.width, bar_h])
    cb = fig.colorbar(ScalarMappable(norm=c_norm, cmap=cmap),
                      cax=cax, orientation='horizontal')
    cb.ax.set_title(r'Concentration Cathode [$\mathrm{mol\,m^{-3}}$]')
    cb.ax.xaxis.set_label_position('top')
    cb.ax.xaxis.set_ticks_position('bottom')

    # ---------- legend (uses the same palette) ----------------------
    proxy_handles = [Line2D([], [], lw=2,
                            color=label_colour['Ground Truth'],
                            label='Ground Truth')]
    proxy_handles += [Line2D([], [], lw=2,
                             color=label_colour[lbl], label=lbl)
                      for lbl in pred_labels]

    ax_legend.legend(proxy_handles,
                        [h.get_label() for h in proxy_handles],
                        loc='center', frameon=False,
                        fontsize=14,      # bigger text
                        handlelength=3,   # longer line dashes
                        handletextpad=1.0,
                        borderpad=1.2)    # extra breathing room

    # align y-labels -------------------------------------------------
    radial_axes = [ax_true_an, ax_true_ca]
    radial_axes += [axes[f'{lbl}_anode']   for lbl in pred_labels]
    radial_axes += [axes[f'{lbl}_cathode'] for lbl in pred_labels]
    radial_axes += [ax for ax in fig.axes if 'err' in ax.get_title()]
    fig.align_ylabels(radial_axes)

    # tidy ------------------------------------------------------------
    fig.tight_layout(rect=[0, 0.1, 1, 1])
    return fig, axes


# ── helper: contourf without crack artefacts ──────────────────────────
# def _contourf_no_cracks(ax, *args, **kwargs):
#     """
#     Wrapper around ax.contourf that hides polygon edges
#     (prevents white lines in vector outputs).
#     """
#     cf = ax.contourf(*args, **kwargs)
#     for coll in cf.collections:
#         coll.set_edgecolor("face")
#         coll.set_linewidth(0)
#     return cf

def _contourf_no_cracks(ax, *args, **kwargs):
    """
    Filled contour without hair-line seams in PDF/SVG:
    • edges painted the same colour as faces
    • layer rasterised so no vector seams can show through
    """
    cf = ax.contourf(*args, **kwargs)

    for coll in cf.collections:
        coll.set_edgecolor("face")
        coll.set_linewidth(0)
        coll.set_rasterized(True)      # ► embed as bitmap inside PDF/SVG

    return cf


# ── main figure builder ───────────────────────────────────────────────
def create_plot_paper11(
        pred_sets,
        c_true_anode, c_true_cathode,
        func_I, V_true,
        t_max, Ran, Rca,
        pred_labels=None,
        cmap='viridis',
        err_cmap='plasma'):

    # ───── sanity / defaults ─────────────────────────────────────────
    n_pred = len(pred_sets)
    if pred_labels is None:
        pred_labels = [f'Pred-{i+1}' for i in range(n_pred)]
    if len(pred_labels) != n_pred:
        raise ValueError('pred_labels length must match pred_sets length')

    # ───── global line-colour palette ────────────────────────────────
    gt_colour   = 'k'
    pred_colours = sns.color_palette("colorblind", n_pred)
    label_colour = {'Ground Truth': gt_colour,
                    **dict(zip(pred_labels, pred_colours))}

    # meshes ----------------------------------------------------------
    R, T = c_true_anode.shape
    t = np.linspace(0, 1, T)
    r = np.linspace(0, 1, R)
    TT, RR = np.meshgrid(t, r)

    # ───── norms ─────────────────────────────────────────────────────
    conc_arrays = ([c_true_anode, c_true_cathode] +
                   [p['anode']   for p in pred_sets] +
                   [p['cathode'] for p in pred_sets])
    conc_norm = Normalize(np.min(conc_arrays), np.max(conc_arrays))

    a_norm = Normalize(np.min([c_true_anode]   + [p['anode']   for p in pred_sets]),
                       np.max([c_true_anode]   + [p['anode']   for p in pred_sets]))
    c_norm = Normalize(np.min([c_true_cathode] + [p['cathode'] for p in pred_sets]),
                       np.max([c_true_cathode] + [p['cathode'] for p in pred_sets]))

    model_err_norm = {}
    for p, lbl in zip(pred_sets, pred_labels):
        err_max = np.max([np.abs(p['anode']   - c_true_anode),
                          np.abs(p['cathode'] - c_true_cathode)])
        model_err_norm[lbl] = Normalize(0, err_max)

    # figure & GridSpec ----------------------------------------------
    n_cols = 1 + n_pred + 1
    fig = plt.figure(figsize=(4.8 * n_cols, 11))
    gs  = gridspec.GridSpec(
        4, n_cols,
        width_ratios=[1]*(n_cols-1) + [0.9],
        hspace=0.45, wspace=0.28, figure=fig
    )
    axes = {}

    # ───── two stacked error colour-bars ─────────────────────────────
    dummy_ax = fig.add_subplot(gs[3, 0])
    bbox = dummy_ax.get_position()
    dummy_ax.remove()

    bar_h, spacing = 0.013, 0.05
    for i, lbl in enumerate(reversed(pred_labels)):
        cax = fig.add_axes([bbox.x0,
                            bbox.y0 + i*(bar_h + spacing),
                            bbox.width, bar_h])

        cb = fig.colorbar(ScalarMappable(norm=model_err_norm[lbl], cmap=err_cmap),
                          cax=cax, orientation='horizontal')
        cb.ax.set_title(fr'{lbl} $|err|\;[\mathrm{{mol\,m^{{-3}}}}]$')
        cb.ax.xaxis.set_ticks_position('bottom')
        cb.ax.tick_params(bottom=True, top=False,
                          labelbottom=True, labeltop=False)

    # ===========  ROW-3 : anode-error maps + legend  ================
    fig.add_subplot(gs[2, 0]).axis('off')
    for col, (p, lbl) in enumerate(zip(pred_sets, pred_labels), start=1):
        ax = fig.add_subplot(gs[2, col])
        _contourf_no_cracks(ax,
                            TT*t_max, RR*Ran*1e6,
                            np.abs(p['anode'] - c_true_anode),
                            levels=100, cmap=err_cmap, norm=model_err_norm[lbl])
        ax.set_xlabel('Time [$s$]')
        ax.set_ylabel('Radial position [µm]')
        ax.set_title(f'{lbl} Anode $|err|$')
        axes[f'{lbl}_err_anode'] = ax

    ax_legend = fig.add_subplot(gs[3, -1])
    ax_legend.axis('off')
    axes['legend'] = ax_legend

    # ===========  ROW-4 : cathode-error maps + I(t) ==================
    fig.add_subplot(gs[3, 0]).axis('off')
    for col, (p, lbl) in enumerate(zip(pred_sets, pred_labels), start=1):
        ax = fig.add_subplot(gs[3, col])
        _contourf_no_cracks(ax,
                            TT*t_max, RR*Rca*1e6,
                            np.abs(p['cathode'] - c_true_cathode),
                            levels=100, cmap=err_cmap, norm=model_err_norm[lbl])
        ax.set_xlabel('Time [$s$]')
        ax.set_ylabel('Radial position [µm]')
        ax.set_title(f'{lbl} Cathode $|err|$')
        axes[f'{lbl}_err_cathode'] = ax

    # ---------- 1-D LINE PLOTS --------------------------------------
    ax_I = fig.add_subplot(gs[0, -1])
    ax_I.plot(t*t_max, func_I, color=label_colour['Ground Truth'])
    ax_I.set_title('Applied Current')
    ax_I.set_xlabel('Time [$s$]')
    ax_I.set_ylabel(r'$I\;[A]$')
    axes['current'] = ax_I

    # ===========  ROW-1 : True-An + preds + V(t) =====================
    ax_true_an = fig.add_subplot(gs[0, 0])
    _contourf_no_cracks(ax_true_an,
                        TT*t_max, RR*Ran*1e6,
                        c_true_anode, levels=100, cmap=cmap, norm=a_norm)
    ax_true_an.set_xlabel('Time [$s$]')
    ax_true_an.set_ylabel('Radial position [µm]')
    ax_true_an.set_title('Ground Truth Anode')
    axes['true_anode'] = ax_true_an

    for col, (p, lbl) in enumerate(zip(pred_sets, pred_labels), start=1):
        ax = fig.add_subplot(gs[0, col])
        _contourf_no_cracks(ax,
                            TT*t_max, RR*Ran*1e6,
                            p['anode'], levels=100, cmap=cmap, norm=a_norm)
        ax.set_xlabel('Time [$s$]')
        ax.set_ylabel('Radial position [µm]')
        ax.set_title(f'{lbl} Anode')
        axes[f'{lbl}_anode'] = ax

    ax_V = fig.add_subplot(gs[1, -1])
    ax_V.plot(t*t_max, V_true, lw=2,
              color=label_colour['Ground Truth'], label='Ground Truth')
    for p, lbl in zip(pred_sets, pred_labels):
        ax_V.plot(t*t_max, p['V'], linestyle='--',
                  color=label_colour[lbl], label=lbl)
    ax_V.set_title('Cell Voltage')
    ax_V.set_xlabel('Time [$s$]')
    ax_V.set_ylabel(r'$V\;[V]$')
    axes['voltage'] = ax_V

    # ===========  ROW-2 : True-Ca + preds + |V-err| ==================
    ax_true_ca = fig.add_subplot(gs[1, 0])
    _contourf_no_cracks(ax_true_ca,
                        TT*t_max, RR*Rca*1e6,
                        c_true_cathode, levels=100, cmap=cmap, norm=c_norm)
    ax_true_ca.set_xlabel('Time [$s$]')
    ax_true_ca.set_ylabel('Radial position [µm]')
    ax_true_ca.set_title('Ground Truth Cathode')
    axes['true_cathode'] = ax_true_ca

    for col, (p, lbl) in enumerate(zip(pred_sets, pred_labels), start=1):
        ax = fig.add_subplot(gs[1, col])
        _contourf_no_cracks(ax,
                            TT*t_max, RR*Rca*1e6,
                            p['cathode'], levels=100, cmap=cmap, norm=c_norm)
        ax.set_xlabel('Time [$s$]')
        ax.set_ylabel('Radial position [µm]')
        ax.set_title(f'{lbl} Cathode')
        axes[f'{lbl}_cathode'] = ax

    ax_Verr = fig.add_subplot(gs[2, -1])
    for p, lbl in zip(pred_sets, pred_labels):
        ax_Verr.plot(t*t_max,
                     np.abs(p['V'] - V_true)*1e3,
                     color=label_colour[lbl],
                     label=f'{lbl} |V-err|')
    ax_Verr.set_title('Absolute Voltage Error')
    ax_Verr.set_xlabel('Time [$s$]')
    ax_Verr.set_ylabel('Error [mV]')
    axes['voltage_error'] = ax_Verr

    # ---------- concentration colour-bars ---------------------------
    tmp_ax = fig.add_subplot(gs[2, 0])
    bbox   = tmp_ax.get_position()
    tmp_ax.remove()

    gap, bar_h = 0.075, 0.02
    # anode bar
    y0 = bbox.y0 + bbox.height - (0 + 1.5) * bar_h
    cax = fig.add_axes([bbox.x0, y0, bbox.width, bar_h])
    cb  = fig.colorbar(ScalarMappable(norm=a_norm, cmap=cmap),
                       cax=cax, orientation='horizontal')
    cb.ax.set_title(r'Concentration Anode [$\mathrm{mol\,m^{-3}}$]')
    cb.ax.xaxis.set_label_position('top')
    cb.ax.xaxis.set_ticks_position('bottom')

    # cathode bar
    y0 = bbox.y0 + bbox.height - (1 + 1.5) * bar_h - gap
    cax = fig.add_axes([bbox.x0, y0, bbox.width, bar_h])
    cb  = fig.colorbar(ScalarMappable(norm=c_norm, cmap=cmap),
                       cax=cax, orientation='horizontal')
    cb.ax.set_title(r'Concentration Cathode [$\mathrm{mol\,m^{-3}}$]')
    cb.ax.xaxis.set_label_position('top')
    cb.ax.xaxis.set_ticks_position('bottom')

    # ---------- legend (you already styled it) ----------------------
    proxy_handles = [Line2D([], [], lw=2,
                            color=label_colour['Ground Truth'],
                            label='Ground Truth')]
    proxy_handles += [Line2D([], [], lw=2,
                             color=label_colour[lbl], label=lbl)
                      for lbl in pred_labels]

    ax_legend.legend(proxy_handles,
                     [h.get_label() for h in proxy_handles],
                     loc='center', frameon=False,
                     fontsize=14,
                     handlelength=3,
                     handletextpad=1.0,
                     borderpad=1.2)

    # align y-labels -------------------------------------------------
    radial_axes = [axes['true_anode'], axes['true_cathode']] \
                  + [axes[f'{lbl}_anode']   for lbl in pred_labels] \
                  + [axes[f'{lbl}_cathode'] for lbl in pred_labels] \
                  + [ax for ax in fig.axes if 'err' in ax.get_title()]
    fig.align_ylabels(radial_axes)

    fig.tight_layout(rect=[0, 0.1, 1, 1])
    return fig, axes


# ── main figure builder ──────────────────────────────────────────────
def create_plot_paper(
        pred_sets,
        c_true_anode, c_true_cathode,
        func_I, V_true,
        t_max, Ran, Rca,
        pred_labels=None,
        cmap='viridis',
        err_cmap='plasma'):

    # ───── sanity / defaults ────────────────────────────────────────
    n_pred = len(pred_sets)
    if pred_labels is None:
        pred_labels = [f'Pred-{i+1}' for i in range(n_pred)]
    if len(pred_labels) != n_pred:
        raise ValueError('pred_labels length must match pred_sets length')

    # ───── global line-colour palette ───────────────────────────────
    gt_colour   = 'k'
    pred_colours = sns.color_palette("colorblind", n_pred)
    label_colour = {'Ground Truth': gt_colour,
                    **dict(zip(pred_labels, pred_colours))}

    # meshes ---------------------------------------------------------
    R, T = c_true_anode.shape
    t = np.linspace(0, 1, T)
    r = np.linspace(0, 1, R)
    TT, RR = np.meshgrid(t, r)

    # ───── norms ────────────────────────────────────────────────────
    a_norm = Normalize(c_true_anode.min(),
                       max(c_true_anode.max(),
                           *(p['anode'].max()   for p in pred_sets)))
    c_norm = Normalize(c_true_cathode.min(),
                       max(c_true_cathode.max(),
                           *(p['cathode'].max() for p in pred_sets)))

    model_err_norm = {}
    for p, lbl in zip(pred_sets, pred_labels):
        err_max = np.max([np.abs(p['anode']   - c_true_anode),
                          np.abs(p['cathode'] - c_true_cathode)])
        model_err_norm[lbl] = Normalize(0, err_max)

    # figure & GridSpec ---------------------------------------------
    n_cols = 1 + n_pred + 1
    fig = plt.figure(figsize=(4.8 * n_cols, 11))
    gs  = gridspec.GridSpec(
        4, n_cols,
        width_ratios=[1]*(n_cols-1) + [0.9],
        hspace=0.45, wspace=0.28, figure=fig
    )
    axes = {}

    # ───── two stacked error colour-bars ────────────────────────────
    dummy_ax = fig.add_subplot(gs[3, 0])
    bbox = dummy_ax.get_position(); dummy_ax.remove()

    bar_h, spacing = 0.013, 0.05
    bar_w = 0.6 * bbox.width                      # 60 % of cell width
    x0_bar = bbox.x0 + 0.5*(bbox.width - bar_w)   # centred

    for i, lbl in enumerate(reversed(pred_labels)):
        cax = fig.add_axes([x0_bar,
                            bbox.y0 + i*(bar_h + spacing),
                            bar_w, bar_h])

        cb = fig.colorbar(ScalarMappable(norm=model_err_norm[lbl], cmap=err_cmap),
                          cax=cax, orientation='horizontal')
        cb.ax.set_title(fr'{lbl} $|err|\;[\mathrm{{mol\,m^{{-3}}}}]$')
        cb.ax.xaxis.set_ticks_position('bottom')
        cb.ax.tick_params(bottom=True, top=False,
                          labelbottom=True, labeltop=False)

    # ===========  ROW-3 : anode error maps + legend ================
    fig.add_subplot(gs[2, 0]).axis('off')
    for col, (p, lbl) in enumerate(zip(pred_sets, pred_labels), start=1):
        ax = fig.add_subplot(gs[2, col])
        _contourf_no_cracks(ax,
                            TT*t_max, RR*Ran*1e6,
                            np.abs(p['anode'] - c_true_anode),
                            cmap=err_cmap, norm=model_err_norm[lbl])
        ax.set_xlabel('Time [$s$]')
        ax.set_ylabel('Radial position [µm]')
        ax.set_title(f'{lbl} Anode $|err|$', fontweight='semibold')
        axes[f'{lbl}_err_anode'] = ax

    ax_legend = fig.add_subplot(gs[3, -1])
    ax_legend.axis('off')
    axes['legend'] = ax_legend

    # ===========  ROW-4 : cathode error maps + current =============
    fig.add_subplot(gs[3, 0]).axis('off')
    for col, (p, lbl) in enumerate(zip(pred_sets, pred_labels), start=1):
        ax = fig.add_subplot(gs[3, col])
        _contourf_no_cracks(ax,
                            TT*t_max, RR*Rca*1e6,
                            np.abs(p['cathode'] - c_true_cathode),
                            cmap=err_cmap, norm=model_err_norm[lbl])
        ax.set_xlabel('Time [$s$]')
        ax.set_ylabel('Radial position [µm]')
        ax.set_title(f'{lbl} Cathode $|err|$', fontweight='semibold')
        axes[f'{lbl}_err_cathode'] = ax

    # ---------- 1-D LINE PLOTS -------------------------------------
    ax_I = fig.add_subplot(gs[0, -1])
    ax_I.plot(t*t_max, func_I, color=label_colour['Ground Truth'])
    ax_I.set_title('Applied current')
    ax_I.set_xlabel('Time [$s$]')
    ax_I.set_ylabel('I [A]')
    axes['current'] = ax_I

    # ---------- concentration heat-maps & voltage plot -------------
    #   Row 0 : Anode concentrations
    ax_true_an = fig.add_subplot(gs[0, 0])
    _contourf_no_cracks(ax_true_an,
                        TT*t_max, RR*Ran*1e6,
                        c_true_anode, cmap=cmap, norm=a_norm)
    ax_true_an.set_xlabel('Time [$s$]')
    ax_true_an.set_ylabel('Radial position [µm]')
    ax_true_an.set_title('Ground Truth Anode', fontweight='semibold')
    axes['true_anode'] = ax_true_an

    for col, (p, lbl) in enumerate(zip(pred_sets, pred_labels), start=1):
        ax = fig.add_subplot(gs[0, col])
        _contourf_no_cracks(ax,
                            TT*t_max, RR*Ran*1e6,
                            p['anode'], cmap=cmap, norm=a_norm)
        ax.set_xlabel('Time [$s$]')
        ax.set_ylabel('Radial position [µm]')
        ax.set_title(f'{lbl} Anode', fontweight='semibold')
        axes[f'{lbl}_anode'] = ax

    #   Voltage line plot
    ax_V = fig.add_subplot(gs[1, -1])
    ax_V.plot(t*t_max, V_true, lw=2,
              color=label_colour['Ground Truth'], label='Ground Truth')
    for p, lbl in zip(pred_sets, pred_labels):
        ax_V.plot(t*t_max, p['V'], linestyle='--',
                  color=label_colour[lbl], label=lbl)
    ax_V.set_title('Cell voltage')
    ax_V.set_xlabel('Time [$s$]')
    ax_V.set_ylabel('V [V]')
    axes['voltage'] = ax_V

    #   Row 1 : Cathode concentrations
    ax_true_ca = fig.add_subplot(gs[1, 0])
    _contourf_no_cracks(ax_true_ca,
                        TT*t_max, RR*Rca*1e6,
                        c_true_cathode, cmap=cmap, norm=c_norm)
    ax_true_ca.set_xlabel('Time [$s$]')
    ax_true_ca.set_ylabel('Radial position [µm]')
    ax_true_ca.set_title('Ground Truth Cathode', fontweight='semibold')
    axes['true_cathode'] = ax_true_ca

    for col, (p, lbl) in enumerate(zip(pred_sets, pred_labels), start=1):
        ax = fig.add_subplot(gs[1, col])
        _contourf_no_cracks(ax,
                            TT*t_max, RR*Rca*1e6,
                            p['cathode'], cmap=cmap, norm=c_norm)
        ax.set_xlabel('Time [$s$]')
        ax.set_ylabel('Radial position [µm]')
        ax.set_title(f'{lbl} Cathode', fontweight='semibold')
        axes[f'{lbl}_cathode'] = ax

    #   Voltage-error line plot
    ax_Verr = fig.add_subplot(gs[2, -1])
    for p, lbl in zip(pred_sets, pred_labels):
        ax_Verr.plot(t*t_max,
                     np.abs(p['V'] - V_true)*1e3,
                     color=label_colour[lbl],
                     label=f'{lbl} |V-err|')
    ax_Verr.set_title('Voltage error')
    ax_Verr.set_xlabel('Time [$s$]')
    ax_Verr.set_ylabel(r'$|ΔV|\;[mV]$')
    axes['voltage_error'] = ax_Verr

    # ---------- concentration colour-bars --------------------------
    tmp_ax = fig.add_subplot(gs[2, 0]); bbox = tmp_ax.get_position(); tmp_ax.remove()
    bar_w = 0.6 * bbox.width
    x0_bar = bbox.x0 + 0.5*(bbox.width - bar_w)
    gap, bar_h = 0.075, 0.02

    # anode bar
    y0 = bbox.y0 + bbox.height - (0 + 1.5) * bar_h
    cax = fig.add_axes([x0_bar, y0, bar_w, bar_h])
    cb  = fig.colorbar(ScalarMappable(norm=a_norm, cmap=cmap),
                       cax=cax, orientation='horizontal')
    cb.ax.set_title(r'Concentration Anode [$\mathrm{mol\,m^{-3}}$]')
    cb.ax.xaxis.set_label_position('top')
    cb.ax.xaxis.set_ticks_position('bottom')

    # cathode bar
    y0 = bbox.y0 + bbox.height - (1 + 1.5) * bar_h - gap
    cax = fig.add_axes([x0_bar, y0, bar_w, bar_h])
    cb  = fig.colorbar(ScalarMappable(norm=c_norm, cmap=cmap),
                       cax=cax, orientation='horizontal')
    cb.ax.set_title(r'Concentration Cathode [$\mathrm{mol\,m^{-3}}$]')
    cb.ax.xaxis.set_label_position('top')
    cb.ax.xaxis.set_ticks_position('bottom')

    # ---------- legend --------------------------------------------
    proxy_handles = [Line2D([], [], lw=2,
                            color=label_colour['Ground Truth'],
                            label='Ground Truth')]
    proxy_handles += [Line2D([], [], lw=2,
                             color=label_colour[lbl], label=lbl)
                      for lbl in pred_labels]

    ax_legend.legend(proxy_handles,
                     [h.get_label() for h in proxy_handles],
                     loc='center', frameon=False,
                     fontsize=14,
                     handlelength=3,
                     handletextpad=1.0,
                     borderpad=1.2,
                     alignment='left')   # ← left-justify

    # align y-labels -----------------------------------------------
    radial_axes = [axes['true_anode'], axes['true_cathode']] \
                  + [axes[f'{lbl}_anode']   for lbl in pred_labels] \
                  + [axes[f'{lbl}_cathode'] for lbl in pred_labels] \
                  + [ax for ax in fig.axes if 'err' in ax.get_title()]
    fig.align_ylabels(radial_axes)

    fig.tight_layout(rect=[0, 0.1, 1, 1])
    return fig, axes


# ── main figure builder ───────────────────────────────────────────────
def create_plot_paper12(
        pred_sets,
        c_true_anode, c_true_cathode,
        func_I, V_true,
        t_max, Ran, Rca,
        pred_labels=None,
        cmap='viridis',
        err_cmap='plasma'):

    # ───── sanity / defaults ─────────────────────────────────────────
    n_pred = len(pred_sets)
    if pred_labels is None:
        pred_labels = [f'Pred-{i+1}' for i in range(n_pred)]
    if len(pred_labels) != n_pred:
        raise ValueError('pred_labels length must match pred_sets length')

    # ───── global line-colour palette ────────────────────────────────
    gt_colour   = 'k'
    pred_colours = sns.color_palette("colorblind", n_pred)
    label_colour = {'Ground Truth': gt_colour,
                    **dict(zip(pred_labels, pred_colours))}

    # meshes ----------------------------------------------------------
    R, T = c_true_anode.shape
    t = np.linspace(0, 1, T)
    r = np.linspace(0, 1, R)
    TT, RR = np.meshgrid(t, r)

    # ───── norms ─────────────────────────────────────────────────────
    conc_arrays = ([c_true_anode, c_true_cathode] +
                   [p['anode']   for p in pred_sets] +
                   [p['cathode'] for p in pred_sets])
    conc_norm = Normalize(np.min(conc_arrays), np.max(conc_arrays))

    a_norm = Normalize(np.min([c_true_anode]   + [p['anode']   for p in pred_sets]),
                       np.max([c_true_anode]   + [p['anode']   for p in pred_sets]))
    c_norm = Normalize(np.min([c_true_cathode] + [p['cathode'] for p in pred_sets]),
                       np.max([c_true_cathode] + [p['cathode'] for p in pred_sets]))

    model_err_norm = {}
    for p, lbl in zip(pred_sets, pred_labels):
        err_max = np.max([np.abs(p['anode']   - c_true_anode),
                          np.abs(p['cathode'] - c_true_cathode)])
        model_err_norm[lbl] = Normalize(0, err_max)

    # figure & GridSpec ----------------------------------------------
    n_cols = 1 + n_pred + 1
    # fig = plt.figure(figsize=(4.8 * n_cols, 11))
    fig = plt.figure(figsize=(1.496 * n_cols, 1.5 * 7.48031596))
    gs  = gridspec.GridSpec(
        4, n_cols,
        width_ratios=[1]*(n_cols-1) + [0.9],
        hspace=0.45, wspace=0.28, figure=fig
    )
    axes = {}

    plt.rcParams.update({
        "savefig.bbox":   "tight",
        "font.family":    "sans-serif",
        "font.sans-serif":["Arial","Helvetica","DejaVu Sans"],
        "font.size":        8,
        "axes.titlesize":   9,
        "axes.labelsize":   8,
        "xtick.labelsize":  7,
        "ytick.labelsize":  7,
        "legend.fontsize":  7,
        "axes.formatter.useoffset": False,
    })

    # ───── two stacked error colour-bars ─────────────────────────────
    dummy_ax = fig.add_subplot(gs[3, 0])
    bbox = dummy_ax.get_position()
    dummy_ax.remove()

    bar_h, spacing = 0.013, 0.05
    for i, lbl in enumerate(reversed(pred_labels)):
        cax = fig.add_axes([bbox.x0,
                            bbox.y0 + i*(bar_h + spacing),
                            bbox.width, bar_h])

        cb = fig.colorbar(ScalarMappable(norm=model_err_norm[lbl], cmap=err_cmap),
                          cax=cax, orientation='horizontal')
        cb.ax.set_title(fr'{lbl} $|err|\;[\mathrm{{mol\,m^{{-3}}}}]$')
        cb.ax.xaxis.set_ticks_position('bottom')
        cb.ax.tick_params(bottom=True, top=False,
                          labelbottom=True, labeltop=False)

    # ===========  ROW-3 : anode-error maps + legend  ================
    fig.add_subplot(gs[2, 0]).axis('off')
    for col, (p, lbl) in enumerate(zip(pred_sets, pred_labels), start=1):
        ax = fig.add_subplot(gs[2, col])
        _contourf_no_cracks(ax,
                            TT*t_max, RR*Ran*1e6,
                            np.abs(p['anode'] - c_true_anode),
                            levels=100, cmap=err_cmap, norm=model_err_norm[lbl])
        ax.set_xlabel('Time [$s$]')
        if col == 1:
            ax.set_ylabel('Radial position [µm]')
        # ax.set_ylabel('Radial position [µm]')
        ax.set_title('Anode $|err|$')
        axes[f'{lbl}_err_anode'] = ax

    ax_legend = fig.add_subplot(gs[3, -1])
    ax_legend.axis('off')
    axes['legend'] = ax_legend

    # ===========  ROW-4 : cathode-error maps + I(t) ==================
    fig.add_subplot(gs[3, 0]).axis('off')
    for col, (p, lbl) in enumerate(zip(pred_sets, pred_labels), start=1):
        ax = fig.add_subplot(gs[3, col])
        _contourf_no_cracks(ax,
                            TT*t_max, RR*Rca*1e6,
                            np.abs(p['cathode'] - c_true_cathode),
                            levels=100, cmap=err_cmap, norm=model_err_norm[lbl])
        ax.set_xlabel('Time [$s$]')
        if col == 1:
            ax.set_ylabel('Radial position [µm]')
        ax.set_title('Cathode $|err|$')
        axes[f'{lbl}_err_cathode'] = ax

    # ---------- 1-D LINE PLOTS --------------------------------------
    ax_I = fig.add_subplot(gs[0, -1])
    ax_I.plot(t*t_max, func_I, color=label_colour['Ground Truth'])
    ax_I.set_title('Applied Current')
    ax_I.set_xlabel('Time [$s$]')
    ax_I.set_ylabel(r'$I\;[A]$')
    axes['current'] = ax_I

    # ===========  ROW-1 : True-An + preds + V(t) =====================
    ax_true_an = fig.add_subplot(gs[0, 0])
    _contourf_no_cracks(ax_true_an,
                        TT*t_max, RR*Ran*1e6,
                        c_true_anode, levels=100, cmap=cmap, norm=a_norm)
    ax_true_an.set_xlabel('Time [$s$]')
    ax_true_an.set_ylabel('Radial position [µm]')
    ax_true_an.set_title("Ground Truth\nAnode")
    text = ax_true_an.text(-10, 5,"Anode", size=8,
        verticalalignment='center', rotation=90)
    axes['true_anode'] = ax_true_an

    for col, (p, lbl) in enumerate(zip(pred_sets, pred_labels), start=1):
        ax = fig.add_subplot(gs[0, col])
        _contourf_no_cracks(ax,
                            TT*t_max, RR*Ran*1e6,
                            p['anode'], levels=100, cmap=cmap, norm=a_norm)
        ax.set_xlabel('Time [$s$]')
        # ax.set_ylabel('Radial position [µm]')
        ax.set_title(f'{lbl}')
        axes[f'{lbl}_anode'] = ax

    ax_V = fig.add_subplot(gs[1, -1])
    ax_V.plot(t*t_max, V_true, lw=2,
              color=label_colour['Ground Truth'], label='Ground Truth')
    for p, lbl in zip(pred_sets, pred_labels):
        ax_V.plot(t*t_max, p['V'], linestyle='--',
                  color=label_colour[lbl], label=lbl)
    ax_V.set_title('Cell Voltage')
    ax_V.set_xlabel('Time [$s$]')
    ax_V.set_ylabel(r'$V\;[V]$')
    axes['voltage'] = ax_V

    # ===========  ROW-2 : True-Ca + preds + |V-err| ==================
    ax_true_ca = fig.add_subplot(gs[1, 0])
    _contourf_no_cracks(ax_true_ca,
                        TT*t_max, RR*Rca*1e6,
                        c_true_cathode, levels=100, cmap=cmap, norm=c_norm)
    ax_true_ca.set_xlabel('Time [$s$]')
    ax_true_ca.set_ylabel('Radial position [µm]')
    ax_true_ca.set_title('Cathode')
    axes['true_cathode'] = ax_true_ca
    text = ax_true_ca.text(0, 0,"Cathode", size=8,
        verticalalignment='center', rotation=90)

    for col, (p, lbl) in enumerate(zip(pred_sets, pred_labels), start=1):
        ax = fig.add_subplot(gs[1, col])
        _contourf_no_cracks(ax,
                            TT*t_max, RR*Rca*1e6,
                            p['cathode'], levels=100, cmap=cmap, norm=c_norm)
        ax.set_xlabel('Time [$s$]')
        # ax.set_ylabel('Radial position [µm]')
        ax.set_title(f'{lbl}')
        axes[f'{lbl}_cathode'] = ax

    ax_Verr = fig.add_subplot(gs[2, -1])
    for p, lbl in zip(pred_sets, pred_labels):
        ax_Verr.plot(t*t_max,
                     np.abs(p['V'] - V_true)*1e3,
                     color=label_colour[lbl],
                     label=f'{lbl} |V-err|')
    ax_Verr.set_title('Absolute Voltage Error')
    ax_Verr.set_xlabel('Time [$s$]')
    ax_Verr.set_ylabel('Error [mV]')
    axes['voltage_error'] = ax_Verr

    # ---------- concentration colour-bars ---------------------------
    tmp_ax = fig.add_subplot(gs[2, 0])
    bbox   = tmp_ax.get_position()
    tmp_ax.remove()

    gap, bar_h = 0.075, 0.02
    # anode bar
    y0 = bbox.y0 + bbox.height - (0 + 1.5) * bar_h
    cax = fig.add_axes([bbox.x0, y0, bbox.width, bar_h])
    cb  = fig.colorbar(ScalarMappable(norm=a_norm, cmap=cmap),
                       cax=cax, orientation='horizontal')
    cb.ax.set_title(r'Concentration Anode [$\mathrm{mol\,m^{-3}}$]')
    cb.ax.xaxis.set_label_position('top')
    cb.ax.xaxis.set_ticks_position('bottom')

    # cathode bar
    y0 = bbox.y0 + bbox.height - (1 + 1.5) * bar_h - gap
    cax = fig.add_axes([bbox.x0, y0, bbox.width, bar_h])
    cb  = fig.colorbar(ScalarMappable(norm=c_norm, cmap=cmap),
                       cax=cax, orientation='horizontal')
    cb.ax.set_title(r'Concentration Cathode [$\mathrm{mol\,m^{-3}}$]')
    cb.ax.xaxis.set_label_position('top')
    cb.ax.xaxis.set_ticks_position('bottom')

    # ---------- legend (you already styled it) ----------------------
    proxy_handles = [Line2D([], [], lw=2,
                            color=label_colour['Ground Truth'],
                            label='Ground Truth')]
    proxy_handles += [Line2D([], [], lw=2,
                             color=label_colour[lbl], label=lbl)
                      for lbl in pred_labels]

    ax_legend.legend(proxy_handles,
                     [h.get_label() for h in proxy_handles],
                     loc='center', frameon=False,
                     fontsize=7,
                     handlelength=3,
                     handletextpad=1.0,
                     borderpad=1.2)

    # align y-labels -------------------------------------------------
    radial_axes = [axes['true_anode'], axes['true_cathode']] \
                  + [axes[f'{lbl}_anode']   for lbl in pred_labels] \
                  + [axes[f'{lbl}_cathode'] for lbl in pred_labels] \
                  + [ax for ax in fig.axes if 'err' in ax.get_title()]
    fig.align_ylabels(radial_axes)

    fig.tight_layout(rect=[0, 0.1, 1, 1])
    return fig, axes


# ── main figure builder ───────────────────────────────────────────────
def create_plot_papera(
        pred_sets,
        c_true_anode, c_true_cathode,
        func_I, V_true,
        t_max, Ran, Rca,
        pred_labels=None,
        cmap='viridis', err_cmap='plasma'):
    """
    Build the multi‑panel figure exactly in the format required for an
    Elsevier 2‑column paper (190 mm width, 7–10 pt fonts).
    """

    # ─────────────────── basics & defaults ──────────────────────────
    n_pred = len(pred_sets)                         # #prediction rows
    if pred_labels is None:
        pred_labels = [f'Pred-{i+1}' for i in range(n_pred)]
    if len(pred_labels) != n_pred:
        raise ValueError('pred_labels length must match pred_sets length')

    gt_colour    = 'k'
    pred_colours = sns.color_palette("colorblind", n_pred)
    label_colour = {'Ground Truth': gt_colour,
                    **dict(zip(pred_labels, pred_colours))}

    # ─────────────────── meshes & norms ─────────────────────────────
    R, T = c_true_anode.shape
    t = np.linspace(0, 1, T)
    r = np.linspace(0, 1, R)
    TT, RR = np.meshgrid(t, r)

    a_norm = Normalize(np.min([c_true_anode]   + [p['anode']   for p in pred_sets]),
                       np.max([c_true_anode]   + [p['anode']   for p in pred_sets]))
    c_norm = Normalize(np.min([c_true_cathode] + [p['cathode'] for p in pred_sets]),
                       np.max([c_true_cathode] + [p['cathode'] for p in pred_sets]))

    model_err_norm = {}
    for p, lbl in zip(pred_sets, pred_labels):
        err_max = np.max([np.abs(p['anode'] - c_true_anode),
                          np.abs(p['cathode'] - c_true_cathode)])
        model_err_norm[lbl] = Normalize(0, err_max)

    # ─────────────────── figure & GridSpec ──────────────────────────
    mm_to_inch = 1 / 25.4
    fig_width  = 190 * mm_to_inch              # 190 mm = full Elsevier width
    row_h_in   = 40 * mm_to_inch               # ≈ 40 mm per data row
    n_rows     = n_pred + 2                    # GT + preds + line‑plot row
    fig_height = row_h_in * n_rows
    fig = plt.figure(figsize=(fig_width, fig_height))

    gs = gridspec.GridSpec(
        n_rows, 4,
        height_ratios=[1]*n_rows,
        width_ratios=[1, 1, 1, 1],             # ← equal columns
        hspace=0.45, wspace=0.35, figure=fig
    )
    cell = lambda r, c: gs[r, c]               # convenience
    axes = {}

    # ─────────────────── typography helper ─────────────────────────
    def style_axis(ax, title=None, xlabel=None, ylabel=None):
        if title  is not None: ax.set_title(title,  fontsize=9)
        if xlabel is not None: ax.set_xlabel(xlabel, fontsize=8)
        if ylabel is not None: ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(axis='both', labelsize=7)

    # ─────────────────── row‑0 : ground truth + colour‑bars ─────────
    ax = fig.add_subplot(cell(0, 0))
    _contourf_no_cracks(ax, TT*t_max, RR*Ran*1e6,
                        c_true_anode, levels=100, cmap=cmap, norm=a_norm)
    style_axis(ax, 'Ground Truth Anode', 'Time [$s$]', 'Radial [µm]')
    axes['true_anode'] = ax

    ax = fig.add_subplot(cell(0, 1))
    _contourf_no_cracks(ax, TT*t_max, RR*Rca*1e6,
                        c_true_cathode, levels=100, cmap=cmap, norm=c_norm)
    style_axis(ax, 'Ground Truth Cathode', 'Time [$s$]', 'Radial [µm]')
    axes['true_cathode'] = ax

    # concentration bars (col‑2, row‑0)
    tmp = fig.add_subplot(cell(0, 2)); bbox_c = tmp.get_position(); tmp.remove()
    bar_h, gap = 0.022, 0.05
    y = bbox_c.y0 + bbox_c.height - bar_h
    cax = fig.add_axes([bbox_c.x0, y, bbox_c.width, bar_h])
    cb  = fig.colorbar(ScalarMappable(norm=a_norm, cmap=cmap),
                       cax=cax, orientation='horizontal')
    cb.ax.set_title(r'Concentration Anode [$\mathrm{mol\,m^{-3}}$]', fontsize=7)
    cb.ax.tick_params(labelsize=7)

    y -= (bar_h + gap)
    cax = fig.add_axes([bbox_c.x0, y, bbox_c.width, bar_h])
    cb  = fig.colorbar(ScalarMappable(norm=c_norm, cmap=cmap),
                       cax=cax, orientation='horizontal')
    cb.ax.set_title(r'Concentration Cathode [$\mathrm{mol\,m^{-3}}$]', fontsize=7)
    cb.ax.tick_params(labelsize=7)

    # error bars (col‑3, row‑0)
    tmp = fig.add_subplot(cell(0, 3)); bbox_e = tmp.get_position(); tmp.remove()
    bar_h, spacing = 0.014, 0.035
    for i, lbl in enumerate(reversed(pred_labels)):
        cax = fig.add_axes([bbox_e.x0,
                            bbox_e.y0 + i*(bar_h + spacing),
                            bbox_e.width, bar_h])
        cb = fig.colorbar(ScalarMappable(norm=model_err_norm[lbl], cmap=err_cmap),
                          cax=cax, orientation='horizontal')
        cb.ax.set_title(fr'{lbl} $|err|\,[\mathrm{{mol\,m^{{-3}}}}]$', fontsize=7)
        cb.ax.tick_params(labelsize=7)

    # ─────────────────── rows‑1…n_pred : predictions ───────────────
    for r, (p, lbl) in enumerate(zip(pred_sets, pred_labels), start=1):
        ax = fig.add_subplot(cell(r, 0))
        _contourf_no_cracks(ax, TT*t_max, RR*Ran*1e6,
                            p['anode'], levels=100, cmap=cmap, norm=a_norm)
        style_axis(ax, f'{lbl} Anode', 'Time [$s$]', 'Radial [µm]')
        axes[f'{lbl}_anode'] = ax

        ax = fig.add_subplot(cell(r, 1))
        _contourf_no_cracks(ax, TT*t_max, RR*Rca*1e6,
                            p['cathode'], levels=100, cmap=cmap, norm=c_norm)
        style_axis(ax, f'{lbl} Cathode', 'Time [$s$]', 'Radial [µm]')
        axes[f'{lbl}_cathode'] = ax

        ax = fig.add_subplot(cell(r, 2))
        _contourf_no_cracks(ax, TT*t_max, RR*Ran*1e6,
                            np.abs(p['anode'] - c_true_anode),
                            levels=100, cmap=err_cmap, norm=model_err_norm[lbl])
        style_axis(ax, f'{lbl} Anode |err|', 'Time [$s$]', 'Radial [µm]')
        axes[f'{lbl}_err_anode'] = ax

        ax = fig.add_subplot(cell(r, 3))
        _contourf_no_cracks(ax, TT*t_max, RR*Rca*1e6,
                            np.abs(p['cathode'] - c_true_cathode),
                            levels=100, cmap=err_cmap, norm=model_err_norm[lbl])
        style_axis(ax, f'{lbl} Cathode |err|', 'Time [$s$]', 'Radial [µm]')
        axes[f'{lbl}_err_cathode'] = ax

    # ─────────────────── bottom row (line plots) ────────────────────
    last = n_rows - 1

    ax = fig.add_subplot(cell(last, 0))
    ax.plot(t*t_max, func_I, color=gt_colour)
    style_axis(ax, 'Applied Current', 'Time [$s$]', r'$I\,\mathrm{[A]}$')
    axes['current'] = ax

    ax = fig.add_subplot(cell(last, 1))
    ax.plot(t*t_max, V_true, lw=2, color=gt_colour, label='Ground Truth')
    for p, lbl in zip(pred_sets, pred_labels):
        ax.plot(t*t_max, p['V'], '--', color=label_colour[lbl], label=lbl)
    style_axis(ax, 'Cell Voltage', 'Time [$s$]', r'$V\,\mathrm{[V]}$')
    axes['voltage'] = ax

    ax = fig.add_subplot(cell(last, 2))
    for p, lbl in zip(pred_sets, pred_labels):
        ax.plot(t*t_max, np.abs(p['V']-V_true)*1e3,
                color=label_colour[lbl], label=f'{lbl} |V-err|')
    style_axis(ax, 'Absolute Voltage Error', 'Time [$s$]', 'Error [mV]')
    axes['voltage_error'] = ax

    # legend (col‑3, bottom row)
    ax_leg = fig.add_subplot(cell(last, 3))
    ax_leg.axis('off')
    handles = ([Line2D([], [], lw=2, color=gt_colour, label='Ground Truth')] +
               [Line2D([], [], lw=2, color=label_colour[lbl], label=lbl)
                for lbl in pred_labels])
    ax_leg.legend(handles, [h.get_label() for h in handles],
                  loc='center', frameon=False, fontsize=8, handlelength=3)
    axes['legend'] = ax_leg

    # ─────────────────── tidy up ────────────────────────────────────
    radial_axes = [axes['true_anode'], axes['true_cathode']] \
                  + [axes[f'{lbl}_anode']   for lbl in pred_labels] \
                  + [axes[f'{lbl}_cathode'] for lbl in pred_labels] \
                  + [axes[f'{lbl}_err_anode'] for lbl in pred_labels] \
                  + [axes[f'{lbl}_err_cathode'] for lbl in pred_labels]
    fig.align_ylabels(radial_axes)

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    return fig, axes

# ── main figure builder ───────────────────────────────────────────────
def create_plot_paperb(
        pred_sets,
        c_true_anode, c_true_cathode,
        func_I, V_true,
        t_max, Ran, Rca,
        pred_labels=None,
        cmap='viridis', err_cmap='plasma'):
    """
    Compact 5×4 Elsevier‑width multi‑panel plot.
    """

    # ───────────────────── basics ───────────────────────────────────
    n_pred = len(pred_sets)                       # rows with predictions
    if pred_labels is None:
        pred_labels = [f'Pred-{i+1}' for i in range(n_pred)]
    if len(pred_labels) != n_pred:
        raise ValueError('pred_labels length must match pred_sets length')

    gt_colour   = 'k'
    pred_cols   = sns.color_palette("colorblind", n_pred)
    label_col   = {'Ground Truth': gt_colour,
                   **dict(zip(pred_labels, pred_cols))}

    # meshes
    R, T = c_true_anode.shape
    t  = np.linspace(0, 1, T)
    r  = np.linspace(0, 1, R)
    TT, RR = np.meshgrid(t, r)

    # norms
    a_norm = Normalize(np.min([c_true_anode]   + [p['anode']   for p in pred_sets]),
                       np.max([c_true_anode]   + [p['anode']   for p in pred_sets]))
    c_norm = Normalize(np.min([c_true_cathode] + [p['cathode'] for p in pred_sets]),
                       np.max([c_true_cathode] + [p['cathode'] for p in pred_sets]))
    err_norm = {lbl: Normalize(0,
                               np.max([np.abs(p['anode']   - c_true_anode),
                                       np.abs(p['cathode'] - c_true_cathode)]))
                for p, lbl in zip(pred_sets, pred_labels)}

    # ───────────────────── figure & GridSpec ────────────────────────
    mm = 1 / 25.4
    fig_w = 190 * mm                     # 190 mm Elsevier full width
    row_h = 42  * mm                     # ~42 mm per row
    n_rows = n_pred + 2                  # GT row + n_pred + line‑plot row
    fig = plt.figure(figsize=(fig_w, row_h * n_rows))

    gs = gridspec.GridSpec(
        n_rows, 4, height_ratios=[1]*n_rows, width_ratios=[1, 1, 1, 1],
        hspace=0.4, wspace=0.25, figure=fig
    )
    cell = lambda r, c: gs[r, c]
    axes = {}

    # helpers --------------------------------------------------------
    def style(ax, title=None, xlab=False, ylab=False):
        if title is not None:
            ax.set_title(title, fontsize=10, pad=4)
        if ylab:
            ax.set_ylabel('Radial [µm]', fontsize=8)
        if xlab:
            ax.set_xlabel('Time [$s$]', fontsize=8)
        ax.tick_params(axis='both', labelsize=7)

    # column titles (only once, first row) --------------------------
    col_titles = ['Anode', 'Cathode', 'Anode |err|', 'Cathode |err|']

    # row‑0 : ground truth + c‑bars & e‑bars -------------------------
    row_title_x = 0.05   # figure‑fraction x‑position for row labels
    def add_row_label(row, text):
        ax0 = fig.add_subplot(cell(row, 0))
        bbox = ax0.get_position()
        fig.text(row_title_x, (bbox.y0 + bbox.y1)/2, text,
                 va='center', ha='left', fontsize=10, rotation=90)
        ax0.remove()  # we only used it for placement

    add_row_label(0, 'Ground Truth')

    # Ground‑truth Anode
    ax = fig.add_subplot(cell(0, 0))
    _contourf_no_cracks(ax, TT*t_max, RR*Ran*1e6,
                        c_true_anode, levels=100, cmap=cmap, norm=a_norm)
    style(ax, col_titles[0], xlab=False, ylab=True)
    axes['true_anode'] = ax

    # Ground‑truth Cathode
    ax = fig.add_subplot(cell(0, 1))
    _contourf_no_cracks(ax, TT*t_max, RR*Rca*1e6,
                        c_true_cathode, levels=100, cmap=cmap, norm=c_norm)
    style(ax, col_titles[1], xlab=False, ylab=False)
    axes['true_cathode'] = ax

    # concentration bars (row‑0, col‑2 cell bbox)
    tmp = fig.add_subplot(cell(0, 2)); bbox_c = tmp.get_position(); tmp.remove()
    bar_h, gap = 0.022, 0.05
    y = bbox_c.y0 + bbox_c.height - bar_h
    cax = fig.add_axes([bbox_c.x0, y, bbox_c.width, bar_h])
    cb  = fig.colorbar(ScalarMappable(norm=a_norm, cmap=cmap),
                       cax=cax, orientation='horizontal')
    cb.ax.set_title(r'Concentration Anode [$\mathrm{mol\,m^{-3}}$]', fontsize=7)
    cb.ax.tick_params(labelsize=7)
    y -= bar_h + gap
    cax = fig.add_axes([bbox_c.x0, y, bbox_c.width, bar_h])
    cb  = fig.colorbar(ScalarMappable(norm=c_norm, cmap=cmap),
                       cax=cax, orientation='horizontal')
    cb.ax.set_title(r'Concentration Cathode [$\mathrm{mol\,m^{-3}}$]', fontsize=7)
    cb.ax.tick_params(labelsize=7)

    # error bars (row‑0, col‑3 cell bbox)
    tmp = fig.add_subplot(cell(0, 3)); bbox_e = tmp.get_position(); tmp.remove()
    bar_h, spacing = 0.014, 0.035
    for i, lbl in enumerate(reversed(pred_labels)):
        cax = fig.add_axes([bbox_e.x0,
                            bbox_e.y0 + i*(bar_h + spacing),
                            bbox_e.width, bar_h])
        cb = fig.colorbar(ScalarMappable(norm=err_norm[lbl], cmap=err_cmap),
                          cax=cax, orientation='horizontal')
        cb.ax.set_title(fr'{lbl} $|err|$ [$\mathrm{{mol\,m^{{-3}}}}$]', fontsize=7)
        cb.ax.tick_params(labelsize=7)

    # ----------------------------------------------------------------
    # prediction rows (rows 1 … n_pred)
    for r, (p, lbl) in enumerate(zip(pred_sets, pred_labels), start=1):
        add_row_label(r, lbl)

        # Anode
        ax = fig.add_subplot(cell(r, 0))
        _contourf_no_cracks(ax, TT*t_max, RR*Ran*1e6,
                            p['anode'], levels=100, cmap=cmap, norm=a_norm)
        style(ax, None, xlab=False, ylab=True)
        axes[f'{lbl}_anode'] = ax

        # Cathode
        ax = fig.add_subplot(cell(r, 1))
        _contourf_no_cracks(ax, TT*t_max, RR*Rca*1e6,
                            p['cathode'], levels=100, cmap=cmap, norm=c_norm)
        style(ax, None, xlab=False, ylab=False)
        axes[f'{lbl}_cathode'] = ax

        # Anode error
        ax = fig.add_subplot(cell(r, 2))
        _contourf_no_cracks(ax, TT*t_max, RR*Ran*1e6,
                            np.abs(p['anode'] - c_true_anode),
                            levels=100, cmap=err_cmap, norm=err_norm[lbl])
        style(ax, None, xlab=False, ylab=False)
        axes[f'{lbl}_err_anode'] = ax

        # Cathode error
        ax = fig.add_subplot(cell(r, 3))
        _contourf_no_cracks(ax, TT*t_max, RR*Rca*1e6,
                            np.abs(p['cathode'] - c_true_cathode),
                            levels=100, cmap=err_cmap, norm=err_norm[lbl])
        style(ax, None, xlab=False, ylab=False)
        axes[f'{lbl}_err_cathode'] = ax

    # ----------------------------------------------------------------
    # bottom row (row n_rows-1) : line plots & legend
    last = n_rows - 1
    add_row_label(last, '')   # keep left margin aligned

    ax = fig.add_subplot(cell(last, 0))
    ax.plot(t*t_max, func_I, color=gt_colour)
    style(ax, 'Applied Current', xlab=True, ylab=True)
    axes['current'] = ax

    ax = fig.add_subplot(cell(last, 1))
    ax.plot(t*t_max, V_true, lw=2, color=gt_colour, label='Ground Truth')
    for p, lbl in zip(pred_sets, pred_labels):
        ax.plot(t*t_max, p['V'], '--', color=label_col[lbl], label=lbl)
    style(ax, 'Cell Voltage', xlab=True, ylab=True)
    axes['voltage'] = ax

    ax = fig.add_subplot(cell(last, 2))
    for p, lbl in zip(pred_sets, pred_labels):
        ax.plot(t*t_max, np.abs(p['V'] - V_true)*1e3,
                color=label_col[lbl], label=f'{lbl} |V-err|')
    style(ax, 'Absolute Voltage Error', xlab=True, ylab=True)
    axes['voltage_error'] = ax

    # legend
    ax_leg = fig.add_subplot(cell(last, 3))
    ax_leg.axis('off')
    handles = ([Line2D([], [], lw=2, color=gt_colour, label='Ground Truth')] +
               [Line2D([], [], lw=2, color=label_col[lbl], label=lbl)
                for lbl in pred_labels])
    ax_leg.legend(handles, [h.get_label() for h in handles],
                  loc='center', frameon=False, fontsize=8, handlelength=3)
    axes['legend'] = ax_leg

    # ----------------------------------------------------------------
    fig.tight_layout(rect=[0.06, 0.03, 1, 1])  # leave space for row labels
    return fig, axes

# ------------------------------------------------------------------ #
def _contourf_no_cracks(ax, X, Y, Z, **kw):
    """Contourf helper that removes white polygon edges."""
    cf = ax.contourf(X, Y, Z, antialiased=True, **kw)
    for coll in cf.collections:
        coll.set_edgecolor("face")
    return cf


# ------------------------------------------------------------------ #
def create_plot_paperc(
        pred_sets,
        c_true_anode, c_true_cathode,
        t_max, Ran, Rca,
        pred_labels=None,
        cmap='viridis', err_cmap='plasma'):
    """
    Compact Elsevier‑width multi‑panel figure (no line plots).

    Layout:  (n_pred + 1) rows  ×  4 columns
             rows 0 … n_pred-1  -> prediction models
             last row           -> Ground‑Truth + colour‑bars
    """

    # ───────────────────── set‑up & sanity ─────────────────────────
    n_pred = len(pred_sets)
    if pred_labels is None:
        pred_labels = [f'Pred-{i+1}' for i in range(n_pred)]
    if len(pred_labels) != n_pred:
        raise ValueError('pred_labels length must match pred_sets length')

    gt_colour = 'k'
    pred_cols = sns.color_palette("colorblind", n_pred)
    label_col = {'Ground Truth': gt_colour,
                 **dict(zip(pred_labels, pred_cols))}

    # meshes --------------------------------------------------------
    R, T = c_true_anode.shape
    t  = np.linspace(0, 1, T)
    r  = np.linspace(0, 1, R)
    TT, RR = np.meshgrid(t, r)

    # normalisations ------------------------------------------------
    a_norm = Normalize(np.min([c_true_anode] + [p['anode'] for p in pred_sets]),
                       np.max([c_true_anode] + [p['anode'] for p in pred_sets]))
    c_norm = Normalize(np.min([c_true_cathode] + [p['cathode'] for p in pred_sets]),
                       np.max([c_true_cathode] + [p['cathode'] for p in pred_sets]))

    err_norm = {lbl: Normalize(0,
                               np.max([np.abs(p['anode'] - c_true_anode),
                                       np.abs(p['cathode'] - c_true_cathode)]))
                for p, lbl in zip(pred_sets, pred_labels)}

    # figure & GridSpec --------------------------------------------
    mm = 1 / 25.4
    fig_w = 190 * mm                  # Elsevier full column width
    row_h = 38  * mm                  # ~38 mm per data row
    n_rows = n_pred + 1               # predictions + ground truth
    fig = plt.figure(figsize=(fig_w, row_h * n_rows))

    gs = gridspec.GridSpec(
        n_rows, 4, width_ratios=[1, 1, 1, 1], height_ratios=[1]*n_rows,
        wspace=0.25, hspace=0.35, figure=fig
    )
    cell = lambda r, c: gs[r, c]
    axes = {}

    # helpers --------------------------------------------------------
    def style_axis(ax, *, show_xlabel=False, show_ylabel=False, title=None):
        if title is not None:
            ax.set_title(title, fontsize=10, pad=4)
        if show_xlabel:
            ax.set_xlabel('Time [$s$]', fontsize=8)
        if show_ylabel:
            ax.set_ylabel('Radial [µm]', fontsize=8)
        ax.tick_params(axis='both', labelsize=7)
        if not show_xlabel:
            ax.tick_params(labelbottom=False)

    # fixed column titles (only on row 0) ---------------------------
    col_titles = ['Anode', 'Cathode', 'Anode |err|', 'Cathode |err|']

    # function for left‑hand row labels -----------------------------
    def add_row_label(r, text):
        # grab bbox of leftmost cell to compute y position
        dummy = fig.add_subplot(cell(r, 0)); bbox = dummy.get_position(); dummy.remove()
        fig.text(0.05, (bbox.y0 + bbox.y1)/2, text, va='center',
                 ha='left', rotation=90, fontsize=10)

    # ----------------------------------------------------------------
    # -------  prediction rows  (0 … n_pred-1)  ----------------------
    # ----------------------------------------------------------------
    for r, (p, lbl) in enumerate(zip(pred_sets, pred_labels)):
        add_row_label(r, lbl)

        # ---- Anode
        ax = fig.add_subplot(cell(r, 0))
        _contourf_no_cracks(ax, TT*t_max, RR*Ran*1e6,
                            p['anode'], levels=100, cmap=cmap, norm=a_norm)
        style_axis(ax, show_ylabel=True,
                   title=col_titles[0] if r == 0 else None)
        axes[f'{lbl}_anode'] = ax

        # ---- Cathode
        ax = fig.add_subplot(cell(r, 1))
        _contourf_no_cracks(ax, TT*t_max, RR*Rca*1e6,
                            p['cathode'], levels=100, cmap=cmap, norm=c_norm)
        style_axis(ax, title=col_titles[1] if r == 0 else None)
        axes[f'{lbl}_cathode'] = ax

        # ---- Anode error
        ax = fig.add_subplot(cell(r, 2))
        _contourf_no_cracks(ax, TT*t_max, RR*Ran*1e6,
                            np.abs(p['anode'] - c_true_anode),
                            levels=100, cmap=err_cmap, norm=err_norm[lbl])
        style_axis(ax, title=col_titles[2] if r == 0 else None, show_xlabel=True if lbl == pred_labels[-1] else False)
        axes[f'{lbl}_err_anode'] = ax

        # ---- Cathode error
        ax = fig.add_subplot(cell(r, 3))
        _contourf_no_cracks(ax, TT*t_max, RR*Rca*1e6,
                            np.abs(p['cathode'] - c_true_cathode),
                            levels=100, cmap=err_cmap, norm=err_norm[lbl])
        style_axis(ax, title=col_titles[3] if r == 0 else None, show_xlabel=True if lbl == pred_labels[-1] else False)
        axes[f'{lbl}_err_cathode'] = ax

    # ----------------------------------------------------------------
    # ---------------  last row : Ground‑Truth  ----------------------
    # ----------------------------------------------------------------
    gt_row = n_rows - 1
    add_row_label(gt_row, 'Ground Truth')

    # Ground‑Truth Anode
    ax = fig.add_subplot(cell(gt_row, 0))
    _contourf_no_cracks(ax, TT*t_max, RR*Ran*1e6,
                        c_true_anode, levels=100, cmap=cmap, norm=a_norm)
    style_axis(ax, show_xlabel=True, show_ylabel=True)
    axes['true_anode'] = ax

    # Ground‑Truth Cathode
    ax = fig.add_subplot(cell(gt_row, 1))
    _contourf_no_cracks(ax, TT*t_max, RR*Rca*1e6,
                        c_true_cathode, levels=100, cmap=cmap, norm=c_norm)
    style_axis(ax, show_xlabel=True)
    axes['true_cathode'] = ax

    # concentration colour‑bars (reuse bbox of gt_row / col‑2)
    tmp = fig.add_subplot(cell(gt_row, 2)); bbox_c = tmp.get_position(); tmp.remove()
    bar_h, gap = 0.023, 0.1
    y = bbox_c.y0 - 0.045 + bbox_c.height - bar_h
    cax = fig.add_axes([bbox_c.x0, y, bbox_c.width, bar_h])
    cb  = fig.colorbar(ScalarMappable(norm=a_norm, cmap=cmap),
                       cax=cax, orientation='horizontal')
    cb.ax.set_title(r'Concentration Anode [$\mathrm{mol\,m^{-3}}$]', fontsize=7)
    cb.ax.tick_params(labelsize=7)
    y -= (bar_h + gap)
    cax = fig.add_axes([bbox_c.x0, y, bbox_c.width, bar_h])
    cb  = fig.colorbar(ScalarMappable(norm=c_norm, cmap=cmap),
                       cax=cax, orientation='horizontal')
    cb.ax.set_title(r'Concentration Cathode [$\mathrm{mol\,m^{-3}}$]', fontsize=7)
    cb.ax.tick_params(labelsize=7)
    # hide axes rectangle that would otherwise appear
    cax.set_frame_on(True)

    # error colour‑bars (reuse bbox of gt_row / col‑3)
    tmp = fig.add_subplot(cell(gt_row, 3)); bbox_e = tmp.get_position(); tmp.remove()
    bar_h, spacing = 0.01, 0.062
    for i, lbl in enumerate(reversed(pred_labels)):
        cax = fig.add_axes([bbox_e.x0,
                            bbox_e.y0 - 0.038 + i*(bar_h + spacing),
                            bbox_e.width, bar_h])
        cb = fig.colorbar(ScalarMappable(norm=err_norm[lbl], cmap=err_cmap),
                          cax=cax, orientation='horizontal')
        cb.ax.set_title(fr'{lbl} $|err|\,$[$\mathrm{{mol\,m^{{-3}}}}$]', fontsize=7, pad =-0.01)
        cb.ax.tick_params(labelsize=7)
        cax.set_frame_on(True)

    # ----------------------------------------------------------------
    fig.tight_layout(rect=[0.05, 0.03, 1, 1])  # left margin for row labels
    return fig, axes

# ------------------------------------------------------------------ #
def create_plot_paperd(
        pred_sets,
        c_true_anode, c_true_cathode,
        t_max, Ran, Rca,
        pred_labels=None,
        cmap='viridis', err_cmap='plasma'):
    """
    Compact Elsevier‑width multi‑panel figure (no line plots).

    Layout:  (n_pred + 1) rows  ×  4 columns
             rows 0 … n_pred-1  -> prediction models
             last row           -> Ground‑Truth + colour‑bars
    """

    # ───────────────────── set‑up & sanity ─────────────────────────
    n_pred = len(pred_sets)
    if pred_labels is None:
        pred_labels = [f'Pred-{i+1}' for i in range(n_pred)]
    if len(pred_labels) != n_pred:
        raise ValueError('pred_labels length must match pred_sets length')

    gt_colour = 'k'
    pred_cols = sns.color_palette("colorblind", n_pred)
    label_col = {'Ground Truth': gt_colour,
                 **dict(zip(pred_labels, pred_cols))}

    # meshes --------------------------------------------------------
    R, T = c_true_anode.shape
    t  = np.linspace(0, 1, T)
    r  = np.linspace(0, 1, R)
    TT, RR = np.meshgrid(t, r)

    # normalisations ------------------------------------------------
    a_norm = Normalize(np.min([c_true_anode] + [p['anode'] for p in pred_sets]),
                       np.max([c_true_anode] + [p['anode'] for p in pred_sets]))
    c_norm = Normalize(np.min([c_true_cathode] + [p['cathode'] for p in pred_sets]),
                       np.max([c_true_cathode] + [p['cathode'] for p in pred_sets]))

    err_norm = {lbl: Normalize(0,
                               np.max([np.abs(p['anode'] - c_true_anode),
                                       np.abs(p['cathode'] - c_true_cathode)]))
                for p, lbl in zip(pred_sets, pred_labels)}

    # figure & GridSpec --------------------------------------------
    mm = 1 / 25.4
    fig_w = 190 * mm                  # Elsevier full column width
    row_h = 34  * mm                  # ~30 mm per data row
    n_rows = n_pred + 1               # predictions + ground truth
    fig = plt.figure(figsize=(fig_w, row_h * n_rows))

    gs = gridspec.GridSpec(
        n_rows, 4, width_ratios=[1, 1, 1, 1], height_ratios=[1]*n_rows,
        wspace=0.25, hspace=0.35, figure=fig
    )
    cell = lambda r, c: gs[r, c]
    axes = {}

    # helpers --------------------------------------------------------
    def style_axis(ax, *, show_xlabel=False, show_ylabel=False, title=None):
        ax.yaxis.offsetText.set_fontsize(7.2)
        if title is not None:
            ax.set_title(title, fontsize=10, pad=4)
        if show_xlabel:
            ax.set_xlabel('Time [$s$]', fontsize=8.2)
        if show_ylabel:
            ax.set_ylabel('Radial [m]', fontsize=8.2)
        ax.tick_params(axis='both', labelsize=7.2)
        if not show_xlabel:
            ax.tick_params(labelbottom=False)

    # fixed column titles (only on row 0) ---------------------------
    col_titles = ['   Anode   ', '   Cathode   ', 'Anode |err|', 'Cathode |err|']

    # function for left‑hand row labels -----------------------------
    def add_row_label(r, text):
        # grab bbox of leftmost cell to compute y position
        dummy = fig.add_subplot(cell(r, 0)); bbox = dummy.get_position(); dummy.remove()
        fig.text(0.05, (bbox.y0 + bbox.y1)/2, text, va='center',
                 ha='left', rotation=90, fontsize=10)

    # ----------------------------------------------------------------
    # -------  prediction rows  (0 … n_pred-1)  ----------------------
    # ----------------------------------------------------------------
    for r, (p, lbl) in enumerate(zip(pred_sets, pred_labels)):
        add_row_label(r, lbl)

        # ---- Anode
        ax = fig.add_subplot(cell(r, 0))
        _contourf_no_cracks(ax, TT*t_max, RR*Ran,
                            p['anode'], levels=100, cmap=cmap, norm=a_norm)
        style_axis(ax, show_ylabel=True,
                   title=col_titles[0] if r == 0 else None)
        axes[f'{lbl}_anode'] = ax

        # ---- Cathode
        ax = fig.add_subplot(cell(r, 1))
        _contourf_no_cracks(ax, TT*t_max, RR*Rca,
                            p['cathode'], levels=100, cmap=cmap, norm=c_norm)
        style_axis(ax, title=col_titles[1] if r == 0 else None, show_ylabel=False)
        axes[f'{lbl}_cathode'] = ax

        # ---- Anode error
        ax = fig.add_subplot(cell(r, 2))
        _contourf_no_cracks(ax, TT*t_max, RR*Ran,
                            np.abs(p['anode'] - c_true_anode),
                            levels=100, cmap=err_cmap, norm=err_norm[lbl])
        style_axis(ax, title=col_titles[2] if r == 0 else None, show_xlabel=True if lbl == pred_labels[-1] else False, show_ylabel=False)
        axes[f'{lbl}_err_anode'] = ax

        # ---- Cathode error
        ax = fig.add_subplot(cell(r, 3))
        _contourf_no_cracks(ax, TT*t_max, RR*Rca,
                            np.abs(p['cathode'] - c_true_cathode),
                            levels=100, cmap=err_cmap, norm=err_norm[lbl])
        style_axis(ax, title=col_titles[3] if r == 0 else None, show_xlabel=True if lbl == pred_labels[-1] else False, show_ylabel=False)
        axes[f'{lbl}_err_cathode'] = ax

    # ----------------------------------------------------------------
    # ---------------  last row : Ground‑Truth  ----------------------
    # ----------------------------------------------------------------
    gt_row = n_rows - 1
    add_row_label(gt_row, 'Ground Truth')

    # Ground‑Truth Anode
    ax = fig.add_subplot(cell(gt_row, 0))
    _contourf_no_cracks(ax, TT*t_max, RR*Ran,
                        c_true_anode, levels=100, cmap=cmap, norm=a_norm)
    style_axis(ax, show_xlabel=True, show_ylabel=True)
    # ax.yaxis.offsetText.set_fontsize(7)
    axes['true_anode'] = ax

    # Ground‑Truth Cathode
    ax = fig.add_subplot(cell(gt_row, 1))
    _contourf_no_cracks(ax, TT*t_max, RR*Rca,
                        c_true_cathode, levels=100, cmap=cmap, norm=c_norm)
    style_axis(ax, show_xlabel=True)
    # ax.yaxis.offsetText.set_fontsize(7)
    axes['true_cathode'] = ax

    # concentration colour‑bars (reuse bbox of gt_row / col‑2)
    tmp = fig.add_subplot(cell(gt_row, 2)); bbox_c = tmp.get_position(); tmp.remove()
    bar_h, gap = 0.023, 0.1
    y = bbox_c.y0 - 0.045 + bbox_c.height - bar_h
    cax = fig.add_axes([bbox_c.x0, y, bbox_c.width, bar_h])
    cb  = fig.colorbar(ScalarMappable(norm=a_norm, cmap=cmap),
                       cax=cax, orientation='horizontal')
    cb.ax.set_title(r'Concentration Anode [$\mathrm{mol\,m^{-3}}$]', fontsize=7.4)
    cb.ax.tick_params(labelsize=7.2)
    y -= (bar_h + gap)
    cax = fig.add_axes([bbox_c.x0, y, bbox_c.width, bar_h])
    cb  = fig.colorbar(ScalarMappable(norm=c_norm, cmap=cmap),
                       cax=cax, orientation='horizontal')
    cb.ax.set_title(r'Concentration Cathode [$\mathrm{mol\,m^{-3}}$]', fontsize=7.4)
    cb.ax.tick_params(labelsize=7.2)
    # hide axes rectangle that would otherwise appear
    cax.set_frame_on(True)

    # error colour‑bars (reuse bbox of gt_row / col‑3)
    tmp = fig.add_subplot(cell(gt_row, 3)); bbox_e = tmp.get_position(); tmp.remove()
    bar_h, spacing = 0.01, 0.062
    for i, lbl in enumerate(reversed(pred_labels)):
        cax = fig.add_axes([bbox_e.x0,
                            bbox_e.y0 - 0.038 + i*(bar_h + spacing),
                            bbox_e.width, bar_h])
        cb = fig.colorbar(ScalarMappable(norm=err_norm[lbl], cmap=err_cmap),
                          cax=cax, orientation='horizontal')
        cb.ax.set_title(fr'{lbl} $|err|\,$[$\mathrm{{mol\,m^{{-3}}}}$]', fontsize=7.4, pad =-0.01)
        cb.ax.tick_params(labelsize=7)
        cax.set_frame_on(True)

    # ----------------------------------------------------------------
    fig.tight_layout(rect=[0.05, 0.03, 1, 1])  # left margin for row labels
    return fig, axes

def create_line_plot_panel(
        pred_sets,
        func_I, V_true,
        t_max,
        pred_labels=None,
        gt_colour='k'):
    """
    190 mm‑wide Elsevier‑style figure with three line plots (side by side)
    and a *very* slim legend column.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : dict[str, matplotlib.axes.Axes]
    """

    # ─────────── inputs & colours ──────────────────────────────────
    n_pred = len(pred_sets)
    if pred_labels is None:
        pred_labels = [f'Pred-{i+1}' for i in range(n_pred)]
    if len(pred_labels) != n_pred:
        raise ValueError('pred_labels length must match pred_sets length')

    pred_cols = sns.color_palette("colorblind", n_pred)
    label_col = {'Ground Truth': gt_colour,
                 **dict(zip(pred_labels, pred_cols))}

    # time axis
    T = len(V_true)
    t = np.linspace(0, t_max, T)

    # ─────────── figure geometry ───────────────────────────────────
    mm          = 1 / 25.4
    fig_w_mm    = 190          # Elsevier full width
    fig_h_mm    = 34           # lower height → more elongated look
    fig         = plt.figure(figsize=(fig_w_mm*mm, fig_h_mm*mm))

    # 3 plots + VERY slim legend column
    # each plot gets ratio 2, legend gets 0.1 → legend ≈ 1.6 % of width
    gs = gridspec.GridSpec(
        1, 4,
        width_ratios=[2, 2, 2, 0.1],
        wspace=0.30, hspace=0, figure=fig
    )
    cell = lambda c: gs[0, c]
    axes = {}

    # font sizes
    fs_tick, fs_label, fs_title, fs_leg = 7, 8, 10, 8

    def stylise(ax, title, ylabel):
        ax.set_title(title, fontsize=fs_title, pad=4)
        ax.set_xlabel('Time [$s$]', fontsize=fs_label)
        ax.set_ylabel(ylabel,  fontsize=fs_label)
        ax.tick_params(axis='both', labelsize=fs_tick)

    # ─────────── Applied Current ───────────────────────────────────
    ax_I = fig.add_subplot(cell(0))
    ax_I.plot(t, func_I, color=gt_colour)
    stylise(ax_I, 'Applied Current', r'$I\;[\mathrm{A}]$')
    axes['current'] = ax_I

    # ─────────── Cell Voltage ──────────────────────────────────────
    ax_V = fig.add_subplot(cell(1))
    ax_V.plot(t, V_true, lw=2, color=gt_colour, label='Ground Truth')
    for p, lbl in zip(pred_sets, pred_labels):
        ax_V.plot(t, p['V'], '--', color=label_col[lbl], label=lbl)
    stylise(ax_V, 'Cell Voltage', r'$V\;[\mathrm{V}]$')
    axes['voltage'] = ax_V

    # ─────────── |Voltage Error| ───────────────────────────────────
    ax_E = fig.add_subplot(cell(2))
    for p, lbl in zip(pred_sets, pred_labels):
        ax_E.plot(t, np.abs(p['V'] - V_true)*1e3,
                  color=label_col[lbl], label=f'{lbl} |V-err|')
    stylise(ax_E, 'Absolute Voltage Error', 'Error [mV]')
    axes['voltage_error'] = ax_E

    # ─────────── Legend (ultra‑slim column) ────────────────────────
    ax_leg = fig.add_subplot(cell(3))
    ax_leg.axis('off')
    handles = ([Line2D([], [], lw=2, color=gt_colour, label='Ground Truth')]
               + [Line2D([], [], lw=2, color=label_col[lbl], label=lbl)
                  for lbl in pred_labels])
    ax_leg.legend(handles,
                  [h.get_label() for h in handles],
                  loc='center', frameon=False,
                  fontsize=fs_leg, handlelength=3)
    axes['legend'] = ax_leg

    fig.tight_layout(rect=[0, 0, 1, 1])
    return fig, axes
