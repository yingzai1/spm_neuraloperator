import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import jax.numpy as jnp
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

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