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
from data_generation import DataGenerator_mixed
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

[train_tensor_unforced, train_tensor_forced, test_tensor, val_tensor] = DataGenerator_mixed(x1range, x2range, numICs, mu, lam, T_step, dt)

print(f"Train tensor for unforced system shape: {train_tensor_unforced.shape}")       # Expected: [10000, 101, 3]
print(f"Train tensor with force shape: {train_tensor_forced.shape}")       # Expected: [10000, 101, 3]
print(f"Test tensor shape: {test_tensor.shape}")          # Expected: [5000, 101, 3]
print(f"Validation tensor shape: {val_tensor.shape}")     # Expected: [5000, 101, 3]

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

eps = 2 # Number of epochs per batch size
lr = 1e-3 # Learning rate
batch_size = 256
S_p = 30
T = len(train_tensor_unforced[0, :, :])
alpha = [0.1, 10e-7, 10e-15]
W = 0
M = 2 # Amount of models you want to run

[Lowest_loss, Lowest_test_loss, Best_Model] = trainingfcn(eps, lr, batch_size, S_p, T, alpha, Num_meas, Num_inputs, Num_x_Obsv, Num_x_Neurons, Num_u_Obsv, Num_u_Neurons, Num_hidden_x_encoder, Num_hidden_x_decoder, Num_hidden_u_encoder, Num_hidden_u_decoder, train_tensor_unforced, train_tensor_forced, test_tensor, M)

# Load the parameters of the best model
model.load_state_dict(torch.load(Best_Model))
print(f"Loaded model parameters from Model: {Best_Model}")

end_time =  time.time()

total_time = end_time - start_time
total_training_time = end_time - start_training_time


print(f"Total time is: {total_time}")
print(f"Total training time is: {total_training_time}")
