import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random as r
import time

from help_func import self_feeding, enc_self_feeding
from nn_structure import AUTOENCODER
from training import trainingfcn
from data_generation import DataGenerator, TwoLinkRobotDataGenerator
from debug_func import debug_L1, debug_L2, debug_L3, debug_L4, debug_L5, debug_L6
from plotting import plot_results, plot_losses, plot_debug

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

start_time = time.time()
# Data Generation

Setup = 'Twolink'# Simple, Twolink

numICs = 20000
T_step = 50
dt = 0.02

if Setup == 'Simple':
    x1range = (-0.5, 0.5)
    x2range = x1range
    mu = -0.05
    lam = -1
    seed = 1
    [train_tensor, test_tensor, val_tensor] = DataGenerator(x1range, x2range, numICs, mu, lam, T_step, dt)
    # NN Structure

    Num_meas = 2
    Num_inputs = 1
    Num_x_Obsv = 3
    Num_u_Obsv = 3
    Num_x_Neurons = 30
    Num_u_Neurons = 30
    Num_hidden_x_encoder = 2
    Num_hidden_u_encoder = 2
    Num_hidden_u_decoder = 2

elif Setup == 'Twolink':
    q1_range = (-np.pi, np.pi)
    q2_range = q1_range
    dq1_range = (-1, 1)
    dq2_range = dq1_range
    [train_tensor, test_tensor, val_tensor] = TwoLinkRobotDataGenerator(q1_range, q2_range, dq1_range, dq2_range, numICs, T_step, dt)

    # NN Structure

    Num_meas = 4
    Num_inputs = 2
    Num_x_Obsv = 17
    Num_u_Obsv = 18
    Num_x_Neurons = 45
    Num_u_Neurons = 50
    Num_hidden_x_encoder = 1
    Num_hidden_u_encoder = 1
    Num_hidden_u_decoder = 1

print(f"Train tensor shape: {train_tensor.shape}")
print(f"Test tensor shape: {test_tensor.shape}")
print(f"Validation tensor shape: {val_tensor.shape}")


# Instantiate the model and move it to the GPU (if available)
model = AUTOENCODER(Num_meas, Num_inputs, Num_x_Obsv, Num_x_Neurons, Num_u_Obsv, Num_u_Neurons, Num_hidden_x_encoder, Num_hidden_u_encoder, Num_hidden_u_decoder)

# Training Loop
start_training_time = time.time()

eps = 2      # Number of epochs per batch size
breakout = 10
lr = 1e-3        # Learning rate
batch_size = 256
S_p = 30
T = len(train_tensor[0, :, :])
alpha = [0.001, 10e-9, 10e-14]
W = 0
M = 1 # Amount of models you want to run
check_epoch = 2

[Lowest_loss,Models_loss_list, Best_Model, Lowest_loss_index, Running_Losses_Array, Lgx_Array, Lgu_Array, L3_Array, L4_Array, L5_Array, L6_Array] = trainingfcn(eps, breakout, check_epoch, lr, batch_size, S_p, T, alpha, Num_meas, Num_inputs, Num_x_Obsv, Num_x_Neurons, Num_u_Obsv, Num_u_Neurons, Num_hidden_x_encoder, Num_hidden_u_encoder, Num_hidden_u_decoder, train_tensor, test_tensor, M, device=device)

ind_loss = int(Lowest_loss_index)
Lgx_Array = np.array(Lgx_Array[ind_loss])
Lgu_Array = np.array(Lgu_Array[ind_loss])
L3_Array = np.array(L3_Array[ind_loss])
L4_Array = np.array(L4_Array[ind_loss])
L5_Array = np.array(L5_Array[ind_loss])
L6_Array = np.array(L6_Array[ind_loss])
Running_Losses_Array = np.array(Running_Losses_Array[ind_loss])


# Create a dictionary with each array as a column
data = {
    "Lgx": Lgx_Array,
    "Lgu": Lgu_Array,
    "L3": L3_Array,
    "L4": L4_Array,
    "L5": L5_Array,
    "L6": L6_Array,
    "Running_Losses": Running_Losses_Array
}

# Create a DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to an Excel file (all data in one sheet, different columns)
df.to_excel("training_results.xlsx", index=False)

# Load the parameters of the best model
checkpoint = torch.load(Best_Model, map_location=device)

if 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
else:
    state_dict = checkpoint
model.load_state_dict(state_dict)

print(f"Loaded model parameters from Model: {Best_Model}")

end_time =  time.time()

total_time = end_time - start_time
total_training_time = end_time - start_training_time

print(f"Total time is: {total_time}")
print(f"Total training time is: {total_training_time}")

# Result Plotting

plot_losses(Lgx_Array, Lgu_Array, L3_Array, L4_Array, L5_Array, L6_Array, Lowest_loss_index)
plot_debug(model, val_tensor, train_tensor, S_p, Num_meas, Num_x_Obsv, T)
plot_results(model, val_tensor, train_tensor, S_p, Num_meas, Num_x_Obsv, T)
