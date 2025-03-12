import random
import matplotlib
matplotlib.use("TkAgg")  # Enables interactive plotting
import matplotlib.pyplot as plt

from debug_func import debug_L12, debug_L3, debug_L4, debug_L5, debug_L6
from help_func import self_feeding, enc_self_feeding

def plot_losses(Lgx_Array, Lgu_Array, L3_Array, L4_Array, L5_Array, L6_Array, Lowest_test_loss_index):

    # Set fontsizes for title, labels, and legend.
    title_fontsize = 14
    label_fontsize = 12
    legend_fontsize = 10

    # Retrieve default color cycle from matplotlib.
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Increase figure size to accommodate the additional subplots.
    fig = plt.figure(figsize=(18, 8))

    # Convert index to integer.
    idx = int(Lowest_test_loss_index)

    # Top subplot: spans both columns of a 4x2 grid (first row)
    ax_top = plt.subplot2grid((4, 2), (0, 0), colspan=2)
    ax_top.plot(Lgx_Array[idx], label='Lgx', color=colors[0])
    ax_top.plot(Lgu_Array[idx], label='Lgu', color=colors[1])
    ax_top.plot(L3_Array[idx], label='L3', color=colors[2])
    ax_top.plot(L4_Array[idx], label='L4', color=colors[3])
    ax_top.plot(L5_Array[idx], label='L5', color=colors[4])
    ax_top.plot(L6_Array[idx], label='L6', color=colors[5])
    ax_top.legend(loc='upper right', fontsize=legend_fontsize)
    ax_top.set_xlabel('Epochs', fontsize=label_fontsize)
    ax_top.set_ylabel('Loss', fontsize=label_fontsize)
    ax_top.set_title('Losses', fontsize=title_fontsize)

    # Lower subplots: arranged as a 3x2 grid.
    ax1 = plt.subplot2grid((4, 2), (1, 0))
    ax1.plot(Lgx_Array[idx], label='Lgx', color=colors[0])
    ax1.legend(loc='upper right', fontsize=legend_fontsize)
    ax1.set_xlabel('Epochs', fontsize=label_fontsize)
    ax1.set_ylabel('Loss', fontsize=label_fontsize)
    ax1.set_title('Lgx', fontsize=title_fontsize)

    ax2 = plt.subplot2grid((4, 2), (1, 1))
    ax2.plot(Lgu_Array[idx], label='Lgu', color=colors[1])
    ax2.legend(loc='upper right', fontsize=legend_fontsize)
    ax2.set_xlabel('Epochs', fontsize=label_fontsize)
    ax2.set_ylabel('Loss', fontsize=label_fontsize)
    ax2.set_title('Lgu', fontsize=title_fontsize)

    ax3 = plt.subplot2grid((4, 2), (2, 0))
    ax3.plot(L3_Array[idx], label='L3', color=colors[2])
    ax3.legend(loc='upper right', fontsize=legend_fontsize)
    ax3.set_xlabel('Epochs', fontsize=label_fontsize)
    ax3.set_ylabel('Loss', fontsize=label_fontsize)
    ax3.set_title('L3', fontsize=title_fontsize)

    ax4 = plt.subplot2grid((4, 2), (2, 1))
    ax4.plot(L4_Array[idx], label='L4', color=colors[3])
    ax4.legend(loc='upper right', fontsize=legend_fontsize)
    ax4.set_xlabel('Epochs', fontsize=label_fontsize)
    ax4.set_ylabel('Loss', fontsize=label_fontsize)
    ax4.set_title('L4', fontsize=title_fontsize)

    ax5 = plt.subplot2grid((4, 2), (3, 0))
    ax5.plot(L5_Array[idx], label='L5', color=colors[4])
    ax5.legend(loc='upper right', fontsize=legend_fontsize)
    ax5.set_xlabel('Epochs', fontsize=label_fontsize)
    ax5.set_ylabel('Loss', fontsize=label_fontsize)
    ax5.set_title('L5', fontsize=title_fontsize)

    ax6 = plt.subplot2grid((4, 2), (3, 1))
    ax6.plot(L6_Array[idx], label='L6', color=colors[5])
    ax6.legend(loc='upper right', fontsize=legend_fontsize)
    ax6.set_xlabel('Epochs', fontsize=label_fontsize)
    ax6.set_ylabel('Loss', fontsize=label_fontsize)
    ax6.set_title('L6', fontsize=title_fontsize)

    plt.tight_layout()
    plt.show()

def plot_losses_mixed(Lgx_unforced_Array, Lgu_forced_Array, L3_forced_Array, L4_forced_Array, L5_forced_Array, L6_forced_Array, 
                      L3_unforced_Array, L4_unforced_Array, L5_unforced_Array, L6_unforced_Array, Lowest_loss_index):
    # Set fontsizes for title, labels, and legend.
    title_fontsize = 14
    label_fontsize = 12
    legend_fontsize = 10

    # Retrieve default color cycle from matplotlib.
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    idx = int(Lowest_loss_index)

    fig = plt.figure(figsize=(18, 8))

    # Top subplot: spans both columns of a 4x2 grid (first row)
    ax_top = plt.subplot2grid((4, 2), (0, 0), colspan=2)
    ax_top.plot(Lgx_unforced_Array[idx], label='Lgx', color=colors[0])
    ax_top.plot(L3_unforced_Array[idx], label='L3', color=colors[2])
    ax_top.plot(L4_unforced_Array[idx], label='L4', color=colors[3])
    ax_top.plot(L5_unforced_Array[idx], label='L5', color=colors[4])
    ax_top.plot(L6_unforced_Array[idx], label='L6', color=colors[5])
    ax_top.legend(loc='upper right', fontsize=legend_fontsize)
    ax_top.set_xlabel('Epochs', fontsize=label_fontsize)
    ax_top.set_ylabel('Loss', fontsize=label_fontsize)
    ax_top.set_title('Losses Unforced', fontsize=title_fontsize)

    # Lower subplots: arranged as a 3x2 grid.
    ax1 = plt.subplot2grid((4, 2), (1, 0))
    ax1.plot(Lgx_unforced_Array[idx], label='Lgx', color=colors[0])
    ax1.legend(loc='upper right', fontsize=legend_fontsize)
    ax1.set_xlabel('Epochs', fontsize=label_fontsize)
    ax1.set_ylabel('Loss', fontsize=label_fontsize)
    ax1.set_title('Lgx', fontsize=title_fontsize)

    ax2 = plt.subplot2grid((4, 2), (1, 1))
    ax2.plot(Lgx_unforced_Array[idx], label='Lgx', color=colors[1])
    ax2.legend(loc='upper right', fontsize=legend_fontsize)
    ax2.set_xlabel('Epochs', fontsize=label_fontsize)
    ax2.set_ylabel('Loss', fontsize=label_fontsize)
    ax2.set_title('Lgx', fontsize=title_fontsize)

    ax3 = plt.subplot2grid((4, 2), (2, 0))
    ax3.plot(L3_unforced_Array[idx], label='L3', color=colors[2])
    ax3.legend(loc='upper right', fontsize=legend_fontsize)
    ax3.set_xlabel('Epochs', fontsize=label_fontsize)
    ax3.set_ylabel('Loss', fontsize=label_fontsize)
    ax3.set_title('L3', fontsize=title_fontsize)

    ax4 = plt.subplot2grid((4, 2), (2, 1))
    ax4.plot(L4_unforced_Array[idx], label='L4', color=colors[3])
    ax4.legend(loc='upper right', fontsize=legend_fontsize)
    ax4.set_xlabel('Epochs', fontsize=label_fontsize)
    ax4.set_ylabel('Loss', fontsize=label_fontsize)
    ax4.set_title('L4', fontsize=title_fontsize)

    ax5 = plt.subplot2grid((4, 2), (3, 0))
    ax5.plot(L5_unforced_Array[idx], label='L5', color=colors[4])
    ax5.legend(loc='upper right', fontsize=legend_fontsize)
    ax5.set_xlabel('Epochs', fontsize=label_fontsize)
    ax5.set_ylabel('Loss', fontsize=label_fontsize)
    ax5.set_title('L5', fontsize=title_fontsize)

    ax6 = plt.subplot2grid((4, 2), (3, 1))
    ax6.plot(L6_unforced_Array[idx], label='L6', color=colors[5])
    ax6.legend(loc='upper right', fontsize=legend_fontsize)
    ax6.set_xlabel('Epochs', fontsize=label_fontsize)
    ax6.set_ylabel('Loss', fontsize=label_fontsize)
    ax6.set_title('L6', fontsize=title_fontsize)

    plt.tight_layout()
    plt.show() 

    fig = plt.figure(figsize=(18, 8))

    # Top subplot: spans both columns of a 4x2 grid (first row)
    ax_top = plt.subplot2grid((4, 2), (0, 0), colspan=2)
    ax_top.plot(Lgu_forced_Array[idx], label='Lgu', color=colors[0])
    ax_top.plot(L3_forced_Array[idx], label='L3', color=colors[2])
    ax_top.plot(L4_forced_Array[idx], label='L4', color=colors[3])
    ax_top.plot(L5_forced_Array[idx], label='L5', color=colors[4])
    ax_top.plot(L6_forced_Array[idx], label='L6', color=colors[5])
    ax_top.legend(loc='upper right', fontsize=legend_fontsize)
    ax_top.set_xlabel('Epochs', fontsize=label_fontsize)
    ax_top.set_ylabel('Loss', fontsize=label_fontsize)
    ax_top.set_title('Losses Forced', fontsize=title_fontsize)

    # Lower subplots: arranged as a 3x2 grid.
    ax1 = plt.subplot2grid((4, 2), (1, 0))
    ax1.plot(Lgu_forced_Array[idx], label='Lgu', color=colors[0])
    ax1.legend(loc='upper right', fontsize=legend_fontsize)
    ax1.set_xlabel('Epochs', fontsize=label_fontsize)
    ax1.set_ylabel('Loss', fontsize=label_fontsize)
    ax1.set_title('Lgu', fontsize=title_fontsize)

    ax2 = plt.subplot2grid((4, 2), (1, 1))
    ax2.plot(Lgu_forced_Array[idx], label='Lgu', color=colors[1])
    ax2.legend(loc='upper right', fontsize=legend_fontsize)
    ax2.set_xlabel('Epochs', fontsize=label_fontsize)
    ax2.set_ylabel('Loss', fontsize=label_fontsize)
    ax2.set_title('Lgu', fontsize=title_fontsize)

    ax3 = plt.subplot2grid((4, 2), (2, 0))
    ax3.plot(L3_forced_Array[idx], label='L3', color=colors[2])
    ax3.legend(loc='upper right', fontsize=legend_fontsize)
    ax3.set_xlabel('Epochs', fontsize=label_fontsize)
    ax3.set_ylabel('Loss', fontsize=label_fontsize)
    ax3.set_title('L3', fontsize=title_fontsize)

    ax4 = plt.subplot2grid((4, 2), (2, 1))
    ax4.plot(L4_forced_Array[idx], label='L4', color=colors[3])
    ax4.legend(loc='upper right', fontsize=legend_fontsize)
    ax4.set_xlabel('Epochs', fontsize=label_fontsize)
    ax4.set_ylabel('Loss', fontsize=label_fontsize)
    ax4.set_title('L4', fontsize=title_fontsize)

    ax5 = plt.subplot2grid((4, 2), (3, 0))
    ax5.plot(L5_forced_Array[idx], label='L5', color=colors[4])
    ax5.legend(loc='upper right', fontsize=legend_fontsize)
    ax5.set_xlabel('Epochs', fontsize=label_fontsize)
    ax5.set_ylabel('Loss', fontsize=label_fontsize)
    ax5.set_title('L5', fontsize=title_fontsize)

    ax6 = plt.subplot2grid((4, 2), (3, 1))
    ax6.plot(L6_forced_Array[idx], label='L6', color=colors[5])
    ax6.legend(loc='upper right', fontsize=legend_fontsize)
    ax6.set_xlabel('Epochs', fontsize=label_fontsize)
    ax6.set_ylabel('Loss', fontsize=label_fontsize)
    ax6.set_title('L6', fontsize=title_fontsize)

    plt.tight_layout()
    plt.show() 

def plot_debug(model, val_tensor, train_tensor, S_p, Num_meas, Num_x_Obsv, T):
    # Set font sizes
    title_fontsize = 14
    label_fontsize = 12
    legend_fontsize = 10

    # Use the validation tensor as xuk
    xuk = val_tensor

    # Get the debug outputs (assumes debug_* functions are defined)
    actual_L1, predicted_L1 = debug_L12(xuk[:, :, :Num_meas], model.x_Encoder, model.x_Decoder)
    actual_L2, predicted_L2 = debug_L12(xuk, model.u_Encoder, model.u_Decoder)
    actual_L3, predicted_L3 = debug_L3(xuk, Num_meas, model)
    actual_L4, predicted_L4 = debug_L4(xuk, Num_meas, model)
    actual_L5, predicted_L5 = debug_L5(xuk, Num_meas, S_p, model)
    actual_L6, predicted_L6 = debug_L6(xuk, Num_meas, Num_x_Obsv, T, model)

    # Select three random sample indices
    sample_indices = random.sample(range(xuk.shape[0]), 3)

    # ---------------------------
    # ENCODER/DECODER PLOT (L1)
    # ---------------------------
    fig, axs = plt.subplots(2, 3, figsize=(18, 8), sharex=True, sharey=True)
    for i, idx in enumerate(sample_indices):
        predicted_traj = predicted_L1[idx]
        actual_traj = actual_L1[idx]
        time_steps = range(actual_L1.shape[1])

        # Plot x1 in first row
        axs[0, i].plot(time_steps, actual_traj[:, 0].cpu().numpy(), 'o-', label='True $\\mathrm{x_{1,m+1}}$')
        axs[0, i].plot(time_steps, predicted_traj[:, 0].detach().cpu().numpy(), 'x--',
                     label='Predicted $\\mathrm{\\phi^{-1}(K^m\\phi(x_{1,0}))}$')
        axs[0, i].set_title(f"gx validation, Sample {idx} (x1)", fontsize=title_fontsize)
        axs[0, i].set_xlabel("Time step", fontsize=label_fontsize)
        axs[0, i].set_ylabel("x1", fontsize=label_fontsize)
        axs[0, i].legend(fontsize=legend_fontsize)

        # Plot x2 in second row
        axs[1, i].plot(time_steps, actual_traj[:, 1].cpu().numpy(), 'o-', label='True $\\mathrm{x_{2,m+1}}$')
        axs[1, i].plot(time_steps, predicted_traj[:, 1].detach().cpu().numpy(), 'x--',
                     label='Predicted $\\mathrm{\\phi^{-1}(K^m\\phi(x_{2,0}))}$')
        axs[1, i].set_title(f"gx validation, Sample {idx} (x2)", fontsize=title_fontsize)
        axs[1, i].set_xlabel("Time step", fontsize=label_fontsize)
        axs[1, i].set_ylabel("x2", fontsize=label_fontsize)
        axs[1, i].legend(fontsize=legend_fontsize)
    plt.tight_layout()
    plt.show()

    # ---------------------------
    # GU VALIDATION PLOT (L2)
    # ---------------------------
    fig, axs = plt.subplots(3, 3, figsize=(18, 8), sharex=True, sharey=True)
    for i, idx in enumerate(sample_indices):
        predicted_traj = predicted_L2[idx]
        actual_traj = actual_L2[idx]
        time_steps = range(actual_L2.shape[1])

        # First row: x1
        axs[0, i].plot(time_steps, actual_traj[:, 0].cpu().numpy(), 'o-', label='True $\\mathrm{x_{1,m+1}}$')
        axs[0, i].plot(time_steps, predicted_traj[:, 0].detach().cpu().numpy(), 'x--',
                     label='Predicted $\\mathrm{\\phi^{-1}(K^m\\phi(x_{1,0}))}$')
        axs[0, i].set_title(f"gu validation, Sample {idx} (x1)", fontsize=title_fontsize)
        axs[0, i].set_xlabel("Time step", fontsize=label_fontsize)
        axs[0, i].set_ylabel("x1", fontsize=label_fontsize)
        axs[0, i].legend(fontsize=legend_fontsize)

        # Second row: x2
        axs[1, i].plot(time_steps, actual_traj[:, 1].cpu().numpy(), 'o-', label='True $\\mathrm{x_{2,m+1}}$')
        axs[1, i].plot(time_steps, predicted_traj[:, 1].detach().cpu().numpy(), 'x--',
                     label='Predicted $\\mathrm{\\phi^{-1}(K^m\\phi(x_{2,0}))}$')
        axs[1, i].set_title(f"gu validation, Sample {idx} (x2)", fontsize=title_fontsize)
        axs[1, i].set_xlabel("Time step", fontsize=label_fontsize)
        axs[1, i].set_ylabel("x2", fontsize=label_fontsize)
        axs[1, i].legend(fontsize=legend_fontsize)

        # Third row: u
        axs[2, i].plot(time_steps, actual_traj[:, 2].cpu().numpy(), 'o-', label='True $\\mathrm{x_{2,m+1}}$')
        axs[2, i].plot(time_steps, predicted_traj[:, 2].detach().cpu().numpy(), 'x--',
                     label='Predicted $\\mathrm{\\phi^{-1}(K^m\\phi(x_{2,0}))}$')
        axs[2, i].set_title(f"gu validation, Sample {idx} (u)", fontsize=title_fontsize)
        axs[2, i].set_xlabel("Time step", fontsize=label_fontsize)
        axs[2, i].set_ylabel("u", fontsize=label_fontsize)
        axs[2, i].legend(fontsize=legend_fontsize)
    plt.tight_layout()
    plt.show()

    # ---------------------------
    # L3 VALIDATION PLOT
    # ---------------------------
    zoom_start, zoom_end = 100, 200
    fig, axs = plt.subplots(2, 2, figsize=(18, 6))
    # First row: x1
    axs[0, 0].plot(actual_L3[:, 0].cpu().numpy(), label='True x1')
    axs[0, 0].plot(predicted_L3[:, 0].detach().cpu().numpy(), label='Predicted x1')
    axs[0, 0].set_title("L3 validation x1", fontsize=title_fontsize)
    axs[0, 0].set_xlabel("Time step", fontsize=label_fontsize)
    axs[0, 0].set_ylabel("x1", fontsize=label_fontsize)
    axs[0, 0].legend(fontsize=legend_fontsize)

    axs[0, 1].plot(actual_L3[zoom_start:zoom_end, 0].cpu().numpy(), label='True x1')
    axs[0, 1].plot(predicted_L3[zoom_start:zoom_end, 0].detach().cpu().numpy(), label='Predicted x1')
    axs[0, 1].set_title("L3 validation x1 (Zoom In)", fontsize=title_fontsize)
    axs[0, 1].set_xlabel("Time step", fontsize=label_fontsize)
    axs[0, 1].set_ylabel("x1", fontsize=label_fontsize)
    axs[0, 1].legend(fontsize=legend_fontsize)

    # Second row: x2
    axs[1, 0].plot(actual_L3[:, 1].cpu().numpy(), label='True x2')
    axs[1, 0].plot(predicted_L3[:, 1].detach().cpu().numpy(), label='Predicted x2')
    axs[1, 0].set_title("L3 validation x2 (Whole Data)", fontsize=title_fontsize)
    axs[1, 0].set_xlabel("Time step", fontsize=label_fontsize)
    axs[1, 0].set_ylabel("x2", fontsize=label_fontsize)
    axs[1, 0].legend(fontsize=legend_fontsize)

    axs[1, 1].plot(actual_L3[zoom_start:zoom_end, 1].cpu().numpy(), label='True x2')
    axs[1, 1].plot(predicted_L3[zoom_start:zoom_end, 1].detach().cpu().numpy(), label='Predicted x2')
    axs[1, 1].set_title("L3 validation x2 (Zoom In)", fontsize=title_fontsize)
    axs[1, 1].set_xlabel("Time step", fontsize=label_fontsize)
    axs[1, 1].set_ylabel("x2", fontsize=label_fontsize)
    axs[1, 1].legend(fontsize=legend_fontsize)
    plt.tight_layout()
    plt.show()

    # ---------------------------
    # L4 VALIDATION PLOT
    # ---------------------------
    fig, axs = plt.subplots(3, 2, figsize=(18, 6))
    # First row: Y1
    axs[0, 0].plot(actual_L4[:, 0].detach().cpu().numpy(), label='True Y1')
    axs[0, 0].plot(predicted_L4[:, 0].detach().cpu().numpy(), label='Predicted Y1')
    axs[0, 0].set_title("L4 validation Y1", fontsize=title_fontsize)
    axs[0, 0].set_xlabel("Time step", fontsize=label_fontsize)
    axs[0, 0].set_ylabel("Y1", fontsize=label_fontsize)
    axs[0, 0].legend(fontsize=legend_fontsize)

    axs[0, 1].plot(actual_L4[zoom_start:zoom_end, 0].detach().cpu().numpy(), label='True Y1')
    axs[0, 1].plot(predicted_L4[zoom_start:zoom_end, 0].detach().cpu().numpy(), label='Predicted Y1')
    axs[0, 1].set_title("L4 validation Y1 (Zoom In)", fontsize=title_fontsize)
    axs[0, 1].set_xlabel("Time step", fontsize=label_fontsize)
    axs[0, 1].set_ylabel("Y1", fontsize=label_fontsize)
    axs[0, 1].legend(fontsize=legend_fontsize)

    # Second row: Y2
    axs[1, 0].plot(actual_L4[:, 1].detach().cpu().numpy(), label='True Y2')
    axs[1, 0].plot(predicted_L4[:, 1].detach().cpu().numpy(), label='Predicted Y2')
    axs[1, 0].set_title("L4 validation Y2 (Whole Data)", fontsize=title_fontsize)
    axs[1, 0].set_xlabel("Time step", fontsize=label_fontsize)
    axs[1, 0].set_ylabel("Y2", fontsize=label_fontsize)
    axs[1, 0].legend(fontsize=legend_fontsize)

    axs[1, 1].plot(actual_L4[zoom_start:zoom_end, 1].detach().cpu().numpy(), label='True Y2')
    axs[1, 1].plot(predicted_L4[zoom_start:zoom_end, 1].detach().cpu().numpy(), label='Predicted Y2')
    axs[1, 1].set_title("L4 validation Y2 (Zoom In)", fontsize=title_fontsize)
    axs[1, 1].set_xlabel("Time step", fontsize=label_fontsize)
    axs[1, 1].set_ylabel("Y2", fontsize=label_fontsize)
    axs[1, 1].legend(fontsize=legend_fontsize)

    # Third row: Y3
    axs[2, 0].plot(actual_L4[:, 2].detach().cpu().numpy(), label='True Y3')
    axs[2, 0].plot(predicted_L4[:, 2].detach().cpu().numpy(), label='Predicted Y3')
    axs[2, 0].set_title("L4 validation Y3 (Whole Data)", fontsize=title_fontsize)
    axs[2, 0].set_xlabel("Time step", fontsize=label_fontsize)
    axs[2, 0].set_ylabel("Y3", fontsize=label_fontsize)
    axs[2, 0].legend(fontsize=legend_fontsize)

    axs[2, 1].plot(actual_L4[zoom_start:zoom_end, 2].detach().cpu().numpy(), label='True Y3')
    axs[2, 1].plot(predicted_L4[zoom_start:zoom_end, 2].detach().cpu().numpy(), label='Predicted Y3')
    axs[2, 1].set_title("L4 validation Y3 (Zoom In)", fontsize=title_fontsize)
    axs[2, 1].set_xlabel("Time step", fontsize=label_fontsize)
    axs[2, 1].set_ylabel("Y3", fontsize=label_fontsize)
    axs[2, 1].legend(fontsize=legend_fontsize)
    plt.tight_layout()
    plt.show()

    # ---------------------------
    # L5 VALIDATION PLOT
    # ---------------------------
    fig, axs = plt.subplots(2, 3, figsize=(18, 8), sharex=True, sharey=True)
    for i, idx in enumerate(sample_indices):
        predicted_traj = predicted_L5[idx]
        actual_traj = actual_L5[idx]
        time_steps = range(actual_L5.shape[1])

        axs[0, i].plot(time_steps, actual_traj[:, 0].cpu().numpy(), 'o-', label='True $\\mathrm{x_{1,m+1}}$')
        axs[0, i].plot(time_steps, predicted_traj[:, 0].detach().cpu().numpy(), 'x--',
                     label='Predicted $\\mathrm{\\phi^{-1}(K^m\\phi(x_{1,0}))}$')
        axs[0, i].set_title(f"L5 validation, Sample {idx} (x1)", fontsize=title_fontsize)
        axs[0, i].set_xlabel("Time step", fontsize=label_fontsize)
        axs[0, i].set_ylabel("x1", fontsize=label_fontsize)
        axs[0, i].legend(fontsize=legend_fontsize)

        axs[1, i].plot(time_steps, actual_traj[:, 1].cpu().numpy(), 'o-', label='True $\\mathrm{x_{2,m+1}}$')
        axs[1, i].plot(time_steps, predicted_traj[:, 1].detach().cpu().numpy(), 'x--',
                     label='Predicted $\\mathrm{\\phi^{-1}(K^m\\phi(x_{2,0}))}$')
        axs[1, i].set_title(f"L5 validation, Sample {idx} (x2)", fontsize=title_fontsize)
        axs[1, i].set_xlabel("Time step", fontsize=label_fontsize)
        axs[1, i].set_ylabel("x2", fontsize=label_fontsize)
        axs[1, i].legend(fontsize=legend_fontsize)
    plt.tight_layout()
    plt.show()

    # ---------------------------
    # L6 VALIDATION PLOT
    # ---------------------------
    fig, axs = plt.subplots(3, 3, figsize=(18, 8), sharex=True, sharey=True)
    for i, idx in enumerate(sample_indices):
        predicted_traj = predicted_L6[idx]
        actual_traj = actual_L6[idx]
        time_steps = range(actual_L6.shape[1])

        axs[0, i].plot(time_steps, actual_traj[:, 0].detach().cpu().numpy(), 'o-', label='True Y1')
        axs[0, i].plot(time_steps, predicted_traj[:, 0].detach().cpu().numpy(), 'x--', label='Predicted Y1')
        axs[0, i].set_title(f"L6 validation, Sample {idx} (Y1)", fontsize=title_fontsize)
        axs[0, i].set_xlabel("Time step", fontsize=label_fontsize)
        axs[0, i].set_ylabel("Y1", fontsize=label_fontsize)
        axs[0, i].legend(fontsize=legend_fontsize)

        axs[1, i].plot(time_steps, actual_traj[:, 1].detach().cpu().numpy(), 'o-', label='True Y2')
        axs[1, i].plot(time_steps, predicted_traj[:, 1].detach().cpu().numpy(), 'x--', label='Predicted Y2')
        axs[1, i].set_title(f"L6 validation, Sample {idx} (Y2)", fontsize=title_fontsize)
        axs[1, i].set_xlabel("Time step", fontsize=label_fontsize)
        axs[1, i].set_ylabel("Y2", fontsize=label_fontsize)
        axs[1, i].legend(fontsize=legend_fontsize)

        axs[2, i].plot(time_steps, actual_traj[:, 2].detach().cpu().numpy(), 'o-', label='True Y3')
        axs[2, i].plot(time_steps, predicted_traj[:, 2].detach().cpu().numpy(), 'x--', label='Predicted Y3')
        axs[2, i].set_title(f"L6 validation, Sample {idx} (Y3)", fontsize=title_fontsize)
        axs[2, i].set_xlabel("Time step", fontsize=label_fontsize)
        axs[2, i].set_ylabel("Y3", fontsize=label_fontsize)
        axs[2, i].legend(fontsize=legend_fontsize)
    plt.tight_layout()
    plt.show()

def plot_results(model, val_tensor, train_tensor, S_p, Num_meas, Num_x_Obsv, T):
    # Set font sizes
    title_fontsize = 14
    label_fontsize = 12
    legend_fontsize = 10
    # ---------------------------
    # VALIDATION PREDICTIONS (enc_self_feeding)
    # ---------------------------
    sample_indices_val = random.sample(range(val_tensor.shape[0]), 3)
    Val_pred_traj, val_loss = enc_self_feeding(model, val_tensor, Num_meas)
    print(f"Running loss for validation: {val_loss:.3e}")

    fig, axs = plt.subplots(2, 3, figsize=(18, 8), sharex=True, sharey=True)
    for i, idx in enumerate(sample_indices_val):
        predicted_traj = Val_pred_traj[idx]
        actual_traj = val_tensor[idx]
        time_steps = range(val_tensor.shape[1])

        axs[0, i].plot(time_steps, actual_traj[:, 0].cpu().numpy(), 'o-', label='True x1')
        axs[0, i].plot(time_steps, predicted_traj[:, 0].detach().cpu().numpy(), 'x--', label='Predicted x1')
        axs[0, i].set_title(f"Validation Sample {idx} (x1)", fontsize=title_fontsize)
        axs[0, i].set_xlabel("Time step", fontsize=label_fontsize)
        axs[0, i].set_ylabel("x1", fontsize=label_fontsize)
        axs[0, i].legend(fontsize=legend_fontsize)

        axs[1, i].plot(time_steps, actual_traj[:, 1].cpu().numpy(), 'o-', label='True x2')
        axs[1, i].plot(time_steps, predicted_traj[:, 1].detach().cpu().numpy(), 'x--', label='Predicted x2')
        axs[1, i].set_title(f"Validation Sample {idx} (x2)", fontsize=title_fontsize)
        axs[1, i].set_xlabel("Time step", fontsize=label_fontsize)
        axs[1, i].set_ylabel("x2", fontsize=label_fontsize)
        axs[1, i].legend(fontsize=legend_fontsize)
    plt.tight_layout()
    plt.show()

    # ---------------------------
    # TRAINING PREDICTIONS (enc_self_feeding)
    # ---------------------------
    sample_indices_train = random.sample(range(train_tensor.shape[0]), 3)
    train_pred_traj, train_loss = enc_self_feeding(model, train_tensor, Num_meas)
    print(f"Running loss for training: {train_loss:.3e}")

    fig, axs = plt.subplots(2, 3, figsize=(18, 8), sharex=True, sharey=True)
    for i, idx in enumerate(sample_indices_train):
        predicted_traj = train_pred_traj[idx]
        actual_traj = train_tensor[idx]
        time_steps = range(train_tensor.shape[1])

        axs[0, i].plot(time_steps, actual_traj[:, 0].cpu().numpy(), 'o-', label='True x1')
        axs[0, i].plot(time_steps, predicted_traj[:, 0].detach().cpu().numpy(), 'x--', label='Predicted x1')
        axs[0, i].set_title(f"Train Sample {idx} (x1)", fontsize=title_fontsize)
        axs[0, i].set_xlabel("Time step", fontsize=label_fontsize)
        axs[0, i].set_ylabel("x1", fontsize=label_fontsize)
        axs[0, i].legend(fontsize=legend_fontsize)

        axs[1, i].plot(time_steps, actual_traj[:, 1].cpu().numpy(), 'o-', label='True x2')
        axs[1, i].plot(time_steps, predicted_traj[:, 1].detach().cpu().numpy(), 'x--', label='Predicted x2')
        axs[1, i].set_title(f"Train Sample {idx} (x2)", fontsize=title_fontsize)
        axs[1, i].set_xlabel("Time step", fontsize=label_fontsize)
        axs[1, i].set_ylabel("x2", fontsize=label_fontsize)
        axs[1, i].legend(fontsize=legend_fontsize)
    plt.tight_layout()
    plt.show()
