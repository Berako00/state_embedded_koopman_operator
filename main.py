import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import random as r
import time

from help_func import self_feeding, enc_self_feeding
from nn_structure import AUTOENCODER
from training import trainingfcn
from data_generation import DataGenerator
from debug_func import debug_L12, debug_L3, debug_L4, debug_L5, debug_L6

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

start_time = time.time()
# Data Generation

numICs = 10000
x1range = (-0.5, 0.5)
x2range = x1range
T_step = 50
dt = 0.02
mu = -0.05
lam = -1
seed = 1

[train_tensor, test_tensor, val_tensor] = DataGenerator(x1range, x2range, numICs, mu, lam, T_step, dt)

print(f"Train tensor shape: {train_tensor.shape}")
print(f"Test tensor shape: {test_tensor.shape}")
print(f"Validation tensor shape: {val_tensor.shape}")
# NN Structure

Num_meas = 2
Num_inputs = 1
Num_x_Obsv = 3
Num_u_Obsv = 3
Num_x_Neurons = 30
Num_u_Neurons = 30
Num_hidden_x_encoder = 2
Num_hidden_x_decoder = 2
Num_hidden_u_encoder = 2
Num_hidden_u_decoder = 2

# Instantiate the model and move it to the GPU (if available)
model = AUTOENCODER(Num_meas, Num_inputs, Num_x_Obsv, Num_x_Neurons, Num_u_Obsv, Num_u_Neurons, Num_hidden_x_encoder, Num_hidden_x_decoder, Num_hidden_u_encoder, Num_hidden_u_decoder)

# Training Loop
start_training_time = time.time()

eps = 5000       # Number of epochs per batch size
lr = 1e-3        # Learning rate
batch_size = 256
S_p = 30
T = len(train_tensor[0, :, :])
alpha = [0.1, 10e-7, 10e-15] 
W = 0
M = 1 # Amount of models you want to run

[Lowest_loss, Lowest_test_loss, Best_Model] = trainingfcn(eps, lr, batch_size, S_p, T, alpha, Num_meas, Num_inputs, Num_x_Obsv, Num_x_Neurons, Num_u_Obsv, Num_u_Neurons, Num_hidden_x_encoder, Num_hidden_x_decoder, Num_hidden_u_encoder, Num_hidden_u_decoder, train_tensor, test_tensor, M)

# Load the parameters of the best model
model.load_state_dict(torch.load(Best_Model))
print(f"Loaded model parameters from Model: {Best_Model}")

end_time =  time.time()

total_time = end_time - start_time
total_training_time = end_time - start_training_time

print(f"Total time is: {total_time}")
print(f"Total training time is: {total_training_time}")

# Result Plotting

xuk = val_tensor

[actual_L1, predicted_L1] = debug_L12(xuk[:,:,:Num_meas], model.x_Encoder, model.x_Decoder)
[actual_L2, predicted_L2] = debug_L12(xuk, model.u_Encoder, model.u_Decoder)
[actual_L3, predicted_L3] = debug_L3(xuk, Num_meas, model)
[actual_L4, predicted_L4] = debug_L4(xuk, Num_meas, model)
[actual_L5, predicted_L5] = debug_L5(xuk, Num_meas, S_p, model)
[actual_L6, predicted_L6] = debug_L6(xuk, Num_meas, Num_x_Obsv, T, model)

sample_indices = r.sample(range(xuk.shape[0]), 3)


title_fontsize = 14
label_fontsize = 12
legend_fontsize = 10

# ENCODER DECODER PLOT

zoom_start, zoom_end = 100, 200

fig, axs = plt.subplots(2, 3, figsize=(18, 8), sharex=True, sharey=True)

for i, idx in enumerate(sample_indices):

    predicted_traj = predicted_L1[idx]
    actual_traj = actual_L1[idx]

    time_steps = range(actual_L1.shape[1])

    # Plot x1 in the first row
    axs[0, i].plot(time_steps, actual_traj[:, 0].cpu().numpy(), 'o-', label='True $\mathrm{x_{1,m+1}}$')
    axs[0, i].plot(time_steps, predicted_traj[:, 0].detach().cpu().numpy(), 'x--', label='Predicted $\mathrm{\phi^{-1}(K^m\phi(x_{1,0}))}$')
    axs[0, i].set_title(f"gx validation, Sample {idx} (x1)")
    axs[0, i].set_xlabel("Time step")
    axs[0, i].set_ylabel("x1")
    axs[0, i].legend()

    # Plot x2 in the second row
    axs[1, i].plot(time_steps, actual_traj[:, 1].cpu().numpy(), 'o-', label='True $\mathrm{x_{2,m+1}}$')
    axs[1, i].plot(time_steps, predicted_traj[:, 1].detach().cpu().numpy(), 'x--', label='Predicted $\mathrm{\phi^{-1}(K^m\phi(x_{2,0}))}$')
    axs[1, i].set_title(f"gx validation, Sample {idx} (x2)")
    axs[1, i].set_xlabel("Time step")
    axs[1, i].set_ylabel("x2")
    axs[1, i].legend()

plt.tight_layout()
plt.show()

fig, axs = plt.subplots(3, 3, figsize=(18, 8), sharex=True, sharey=True)

for i, idx in enumerate(sample_indices):

    predicted_traj = predicted_L2[idx]
    actual_traj = actual_L2[idx]

    time_steps = range(actual_L2.shape[1])

    # Plot x1 in the first row
    axs[0, i].plot(time_steps, actual_traj[:, 0].cpu().numpy(), 'o-', label='True $\mathrm{x_{1,m+1}}$')
    axs[0, i].plot(time_steps, predicted_traj[:, 0].detach().cpu().numpy(), 'x--', label='Predicted $\mathrm{\phi^{-1}(K^m\phi(x_{1,0}))}$')
    axs[0, i].set_title(f"gu validation, Sample {idx} (x1)")
    axs[0, i].set_xlabel("Time step")
    axs[0, i].set_ylabel("x1")
    axs[0, i].legend()

    # Plot x2 in the second row
    axs[1, i].plot(time_steps, actual_traj[:, 1].cpu().numpy(), 'o-', label='True $\mathrm{x_{2,m+1}}$')
    axs[1, i].plot(time_steps, predicted_traj[:, 1].detach().cpu().numpy(), 'x--', label='Predicted $\mathrm{\phi^{-1}(K^m\phi(x_{2,0}))}$')
    axs[1, i].set_title(f"gu validation, Sample {idx} (x2)")
    axs[1, i].set_xlabel("Time step")
    axs[1, i].set_ylabel("x2")
    axs[1, i].legend()

    # Plot x2 in the second row
    axs[2, i].plot(time_steps, actual_traj[:, 2].cpu().numpy(), 'o-', label='True $\mathrm{x_{2,m+1}}$')
    axs[2, i].plot(time_steps, predicted_traj[:, 2].detach().cpu().numpy(), 'x--', label='Predicted $\mathrm{\phi^{-1}(K^m\phi(x_{2,0}))}$')
    axs[2, i].set_title(f"gu validation, Sample {idx} (u)")
    axs[2, i].set_xlabel("Time step")
    axs[2, i].set_ylabel("u")
    axs[2, i].legend()

plt.tight_layout()
plt.show()

fig, axs = plt.subplots(2, 2, figsize=(18, 6))

# First row: variable x1
axs[0, 0].plot(actual_L3[:, 0].cpu().numpy(), label='True x1')
axs[0, 0].plot(predicted_L3[:, 0].detach().cpu().numpy(), label='Predicted x1')
axs[0, 0].set_title("L3 validation x1")
axs[0, 0].set_xlabel("Time step")
axs[0, 0].set_ylabel("x1")
axs[0, 0].legend()

axs[0, 1].plot(actual_L3[zoom_start:zoom_end, 0].cpu().numpy(), label='True x1')
axs[0, 1].plot(predicted_L3[zoom_start:zoom_end, 0].detach().cpu().numpy(), label='Predicted x1')
axs[0, 1].set_title("L3 validation x1 (Zoom In)")
axs[0, 1].set_xlabel("Time step")
axs[0, 1].set_ylabel("x1")
axs[0, 1].legend()

# Second row: variable x2
axs[1, 0].plot(actual_L3[:, 1].cpu().numpy(), label='True x2')
axs[1, 0].plot(predicted_L3[:, 1].detach().cpu().numpy(), label='Predicted x2')
axs[1, 0].set_title("L3 validation x2 (Whole Data)")
axs[1, 0].set_xlabel("Time step")
axs[1, 0].set_ylabel("x2")
axs[1, 0].legend()

axs[1, 1].plot(actual_L3[zoom_start:zoom_end, 1].cpu().numpy(), label='True x2')
axs[1, 1].plot(predicted_L3[zoom_start:zoom_end, 1].detach().cpu().numpy(), label='Predicted x2')
axs[1, 1].set_title("L3 validation x2 (Zoom In)")
axs[1, 1].set_xlabel("Time step")
axs[1, 1].set_ylabel("x2")
axs[1, 1].legend()

plt.tight_layout()
plt.show()

fig, axs = plt.subplots(3, 2, figsize=(18, 6))

# First row: variable Y1
axs[0, 0].plot(actual_L4[:, 0].detach().cpu().numpy(), label='True Y1')
axs[0, 0].plot(predicted_L4[:, 0].detach().cpu().numpy(), label='Predicted Y1')
axs[0, 0].set_title("L4 validation Y1")
axs[0, 0].set_xlabel("Time step")
axs[0, 0].set_ylabel("Y1")
axs[0, 0].legend()

axs[0, 1].plot(actual_L4[zoom_start:zoom_end, 0].detach().cpu().numpy(), label='True Y1')
axs[0, 1].plot(predicted_L4[zoom_start:zoom_end, 0].detach().cpu().numpy(), label='Predicted Y1')
axs[0, 1].set_title("L4 validation Y1 (Zoom In)")
axs[0, 1].set_xlabel("Time step")
axs[0, 1].set_ylabel("Y1")
axs[0, 1].legend()

# Second row: variable Y2
axs[1, 0].plot(actual_L4[:, 1].detach().cpu().numpy(), label='True Y2')
axs[1, 0].plot(predicted_L4[:, 1].detach().cpu().numpy(), label='Predicted Y2')
axs[1, 0].set_title("L4 validation Y2 (Whole Data)")
axs[1, 0].set_xlabel("Time step")
axs[1, 0].set_ylabel("Y2")
axs[1, 0].legend()

axs[1, 1].plot(actual_L4[zoom_start:zoom_end, 1].detach().cpu().numpy(), label='True Y2')
axs[1, 1].plot(predicted_L4[zoom_start:zoom_end, 1].detach().cpu().numpy(), label='Predicted Y2')
axs[1, 1].set_title("L4 validation Y2 (Zoom In)")
axs[1, 1].set_xlabel("Time step")
axs[1, 1].set_ylabel("Y2")
axs[1, 1].legend()

axs[2, 0].plot(actual_L4[:, 2].detach().cpu().numpy(), label='True Y3')
axs[2, 0].plot(predicted_L4[:, 2].detach().cpu().numpy(), label='Predicted Y3')
axs[2, 0].set_title("L4 validation Y3 (Whole Data)")
axs[2, 0].set_xlabel("Time step")
axs[2, 0].set_ylabel("Y3")
axs[2, 0].legend()

axs[2, 1].plot(actual_L4[zoom_start:zoom_end, 2].detach().cpu().numpy(), label='True Y3')
axs[2, 1].plot(predicted_L4[zoom_start:zoom_end, 2].detach().cpu().numpy(), label='Predicted Y3')
axs[2, 1].set_title("L4 validation Y3 (Zoom In)")
axs[2, 1].set_xlabel("Time step")
axs[2, 1].set_ylabel("Y3")
axs[2, 1].legend()

plt.tight_layout()
plt.show()

# LOSS PRED PLOT

fig, axs = plt.subplots(2, 3, figsize=(18, 8), sharex=True, sharey=True)

for i, idx in enumerate(sample_indices):

    predicted_traj = predicted_L5[idx]
    actual_traj = actual_L5[idx]

    time_steps = range(actual_L5.shape[1])

    # Plot x1 in the first row
    axs[0, i].plot(time_steps, actual_traj[:, 0].cpu().numpy(), 'o-', label='True $\mathrm{x_{1,m+1}}$')
    axs[0, i].plot(time_steps, predicted_traj[:, 0].detach().cpu().numpy(), 'x--', label='Predicted $\mathrm{\phi^{-1}(K^m\phi(x_{1,0}))}$')
    axs[0, i].set_title(f"L5 validation, Sample {idx} (x1)")
    axs[0, i].set_xlabel("Time step")
    axs[0, i].set_ylabel("x1")
    axs[0, i].legend()

    # Plot Y2 in the second row
    axs[1, i].plot(time_steps, actual_traj[:, 1].cpu().numpy(), 'o-', label='True $\mathrm{x_{2,m+1}}$')
    axs[1, i].plot(time_steps, predicted_traj[:, 1].detach().cpu().numpy(), 'x--', label='Predicted $\mathrm{\phi^{-1}(K^m\phi(x_{2,0}))}$')
    axs[1, i].set_title(f"L5 validation, Sample {idx} (x2)")
    axs[1, i].set_xlabel("Time step")
    axs[1, i].set_ylabel("x2")
    axs[1, i].legend()

plt.tight_layout()
plt.show()

fig, axs = plt.subplots(3, 3, figsize=(18, 8), sharex=True, sharey=True)

for i, idx in enumerate(sample_indices):

    predicted_traj = predicted_L6[idx]
    actual_traj = actual_L6[idx]

    time_steps = range(actual_L6.shape[1])

    # Plot x1 in the first row
    axs[0, i].plot(time_steps, actual_traj[:, 0].detach().cpu().numpy(), 'o-', label='True Y1$')
    axs[0, i].plot(time_steps, predicted_traj[:, 0].detach().cpu().numpy(), 'x--', label='Predicted Y1')
    axs[0, i].set_title(f"L6 validation, Sample {idx} (Y1)")
    axs[0, i].set_xlabel("Time step")
    axs[0, i].set_ylabel("Y1")
    axs[0, i].legend()

    # Plot x2 in the second row
    axs[1, i].plot(time_steps, actual_traj[:, 1].detach().cpu().numpy(), 'o-', label='True Y2$')
    axs[1, i].plot(time_steps, predicted_traj[:, 1].detach().cpu().numpy(), 'x--', label='Predicted Y2$')
    axs[1, i].set_title(f"L6 validation, Sample {idx} (Y2)")
    axs[1, i].set_xlabel("Time step")
    axs[1, i].set_ylabel("Y2")
    axs[1, i].legend()

    # Plot x2 in the second row
    axs[2, i].plot(time_steps, actual_traj[:, 2].detach().cpu().numpy(), 'o-', label='True Y3$')
    axs[2, i].plot(time_steps, predicted_traj[:, 2].detach().cpu().numpy(), 'x--', label='Predicted Y3$')
    axs[2, i].set_title(f"L6 validation, Sample {idx} (Y3)")
    axs[2, i].set_xlabel("Time step")
    axs[2, i].set_ylabel("Y3")
    axs[2, i].legend()

plt.tight_layout()
plt.show()

sample_indices = r.sample(range(val_tensor.shape[0]), 3)
[Val_pred_traj, val_loss] = enc_self_feeding(model, val_tensor, Num_meas)

print(f"Running loss for validation: {val_loss:.3e}")

fig, axs = plt.subplots(2, 3, figsize=(18, 8), sharex=True, sharey=True)

for i, idx in enumerate(sample_indices):

    predicted_traj = Val_pred_traj[idx]
    actual_traj = val_tensor[idx]

    time_steps = range(val_tensor.shape[1])

    # Plot x1 in the first row
    axs[0, i].plot(time_steps, actual_traj[:, 0].cpu().numpy(), 'o-', label='True x1')
    axs[0, i].plot(time_steps, predicted_traj[:, 0].detach().cpu().numpy(), 'x--', label='Predicted x1')
    axs[0, i].set_title(f"Validation Sample {idx} (x1)")
    axs[0, i].set_xlabel("Time step")
    axs[0, i].set_ylabel("x1")
    axs[0, i].legend()

    # Plot x2 in the second row
    axs[1, i].plot(time_steps, actual_traj[:, 1].cpu().numpy(), 'o-', label='True x2')
    axs[1, i].plot(time_steps, predicted_traj[:, 1].detach().cpu().numpy(), 'x--', label='Predicted x2')
    axs[1, i].set_title(f"Validation Sample {idx} (x2)")
    axs[1, i].set_xlabel("Time step")
    axs[1, i].set_ylabel("x2")
    axs[1, i].legend()

plt.tight_layout()
plt.show()

# Choose three distinct sample indices
sample_indices = r.sample(range(train_tensor.shape[0]), 3)
[train_pred_traj, train_loss] = enc_self_feeding(model, train_tensor, Num_meas)

print(f"Running loss for training: {train_loss:.3e}")

fig, axs = plt.subplots(2, 3, figsize=(18, 8), sharex=True, sharey=True)

for i, idx in enumerate(sample_indices):

    predicted_traj = train_pred_traj[idx]
    actual_traj = train_tensor[idx]

    time_steps = range(train_tensor.shape[1])

    # Plot x1 in the first row
    axs[0, i].plot(time_steps, actual_traj[:, 0].cpu().numpy(), 'o-', label='True x1')
    axs[0, i].plot(time_steps, predicted_traj[:, 0].detach().cpu().numpy(), 'x--', label='Predicted x1')
    axs[0, i].set_title(f"Train Sample {idx} (x1)")
    axs[0, i].set_xlabel("Time step")
    axs[0, i].set_ylabel("x1")
    axs[0, i].legend()

    # Plot x2 in the second row
    axs[1, i].plot(time_steps, actual_traj[:, 1].cpu().numpy(), 'o-', label='True x2')
    axs[1, i].plot(time_steps, predicted_traj[:, 1].detach().cpu().numpy(), 'x--', label='Predicted x2')
    axs[1, i].set_title(f"Train Sample {idx} (x2)")
    axs[1, i].set_xlabel("Time step")
    axs[1, i].set_ylabel("x2")
    axs[1, i].legend()

plt.tight_layout()
plt.show()
