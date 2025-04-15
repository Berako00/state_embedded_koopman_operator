import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import random as r
import time

from help_func import self_feeding, enc_self_feeding, load_model
from nn_structure import AUTOENCODER
from training import trainingfcn_mixed
from data_generation import DataGenerator_mixed
from debug_func import debug_L1, debug_L2, debug_L3, debug_L4, debug_L5, debug_L6
from plotting import plot_results, plot_losses_mixed, plot_debug

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

start_time = time.time()

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
    [train_tensor_unforced, train_tensor_forced, test_tensor_unforced, test_tensor_forced, val_tensor] = DataGenerator_mixed(x1range, x2range, numICs, mu, lam, T_step, dt)
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
    dq1_range = (-6, 6)
    dq2_range = dq1_range
    seed = 1
    [train_tensor_unforced, train_tensor_forced, test_tensor_unforced, test_tensor_forced, val_tensor] = TwoLinkRobotDataGenerator_mixed(q1_range, q2_range, dq1_range, dq2_range, numICs, T_step, dt, tau_max = 7.5)

    # NN Structure

    Num_meas = 4
    Num_inputs = 2
    Num_x_Obsv = 29
    Num_u_Obsv = 48
    Num_x_Neurons = 128
    Num_u_Neurons = 128
    Num_hidden_x_encoder = 3
    Num_hidden_u_encoder = 3
    Num_hidden_u_decoder = 3

print(f"Train tensor for unforced system shape: {train_tensor_unforced.shape}")
print(f"Train tensor with force shape: {train_tensor_forced.shape}")
print(f"Test tensor for unforced system shape: {test_tensor_unforced.shape}")
print(f"Test tensor with force shape: {test_tensor_forced.shape}")
print(f"Validation tensor shape: {val_tensor.shape}")

# Instantiate the model and move it to the GPU (if available)
model = AUTOENCODER(Num_meas, Num_inputs, Num_x_Obsv, Num_x_Neurons, Num_u_Obsv, Num_u_Neurons, Num_hidden_x_encoder,  Num_hidden_u_encoder, Num_hidden_u_decoder)

# Training Loop
start_training_time = time.time()

eps = 700 # Number of epochs per batch size
lr = 1e-3 # Learning rate
batch_size = 256
S_p = 30
T = len(train_tensor_unforced[0, :, :])
alpha = [0.1, 10e-7, 10e-15]
W = 0
M = 1 # Amount of models you want to run
check_epoch = 2

[Lowest_loss, Models_loss_list, Best_Model, Lowest_loss_index,
          Running_Losses_Array, Lgu_forced_Array,
          L4_unforced_Array, L6_unforced_Array,
          L4_forced_Array, L6_forced_Array] = trainingfcn_mixed(eps, check_epoch, lr, batch_size, S_p, T, alpha,
                                                                        Num_meas, Num_inputs, Num_x_Obsv, Num_x_Neurons,
                                                                        Num_u_Obsv, Num_u_Neurons, Num_hidden_x_encoder,
                                                                         Num_hidden_u_encoder, Num_hidden_u_decoder,
                                                                        train_tensor_unforced, train_tensor_forced, test_tensor_unforced,
                                                                        test_tensor_forced, M)
# Load the parameters of the best model
load_model(model, Best_Model, device)
print(f"Loaded model parameters from Model: {Best_Model}")


end_time =  time.time()

total_time = end_time - start_time
total_training_time = end_time - start_training_time


print(f"Total time is: {total_time}")
print(f"Total training time is: {total_training_time}")

# Result Plotting


#plot_debug(model, val_tensor, train_tensor_forced, S_p, Num_meas, Num_x_Obsv, T)
#plot_results(model, val_tensor, train_tensor_forced, S_p, Num_meas, Num_x_Obsv, T)
