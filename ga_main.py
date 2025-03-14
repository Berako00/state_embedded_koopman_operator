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
from training import trainingfcn, trainingfcn_mixed
from data_generation import DataGenerator, DataGenerator_mixed

from plotting import plot_results
from ga_optimizer import run_genetic_algorithm

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
# GA parameters
generations = 1
pop_size = 1
eps = 10

# Define training type
training_type = 'normal' # 'mixed' or 'normal' training

[train_tensor, test_tensor, val_tensor] = DataGenerator(x1range, x2range, numICs, mu, lam, T_step, dt)

[train_tensor_unforced, train_tensor_forced, test_tensor_unforced, test_tensor_forced, val_tensor] = DataGenerator_mixed(x1range, x2range, numICs, mu, lam, T_step, dt)

print(f"Train tensor shape: {train_tensor.shape}")
print(f"Test tensor shape: {test_tensor.shape}")
print(f"Validation tensor shape: {val_tensor.shape}")

print(f"Train tensor unforced shape: {train_tensor_unforced.shape}")
print(f"Test tensor unforced shape: {test_tensor_unforced.shape}")
print(f"Train tensor forced shape: {train_tensor_forced.shape}")
print(f"Test tensor forced shape: {test_tensor_forced.shape}")

# Define parameter ranges
param_ranges = {
    "Num_x_Obsv": (1, 5),
    "Num_u_Obsv": (1, 5),
    "Num_x_Neurons": (10, 50),
    "Num_u_Neurons": (10, 50),
    "Num_hidden_x": (1, 3),  # Shared for both x encoder and decoder
    "Num_hidden_u": (1, 3),  # Shared for both u encoder and decoder
    "alpha0": (0.01, 1.0),
    "alpha1": (1e-9, 1e-5),
    "alpha2": (1e-18, 1e-12)
}

# --- Genetic Algorithm Hyperparameter Optimization ---
use_ga = True
if use_ga:
    # For speed, use a lower number of epochs for evaluation (eps) and fewer generations/population size.
    best_params = run_genetic_algorithm(Num_meas, Num_inputs, training_type, train_tensor, test_tensor, train_tensor_unforced, train_tensor_forced, test_tensor_unforced, test_tensor_forced, generations, pop_size, eps, param_ranges=param_ranges, elitism_count=1)

    Num_meas      = best_params['Num_meas']
    Num_inputs    = best_params['Num_inputs']
    Num_x_Obsv    = best_params['Num_x_Obsv']
    Num_u_Obsv    = best_params['Num_u_Obsv']
    Num_x_Neurons = best_params['Num_x_Neurons']
    Num_u_Neurons = best_params['Num_u_Neurons']
    # Use the same value for both encoder and decoder hidden layers
    Num_hidden_x  = best_params['Num_hidden_x']
    Num_hidden_u  = best_params['Num_hidden_u']
    alpha         = [best_params['alpha0'], best_params['alpha1'], best_params['alpha2']]
else:
    # Default hyperparameters
    Num_meas             = 2
    Num_inputs           = 1
    Num_x_Obsv           = 3
    Num_u_Obsv           = 2
    Num_x_Neurons        = 30
    Num_u_Neurons        = 30
    Num_hidden_x_encoder = 2
    Num_hidden_x_decoder = 2
    Num_hidden_u_encoder = 2
    Num_hidden_u_decoder = 2
    alpha                = [0.1, 10e-7, 10e-15]
    
model = AUTOENCODER(Num_meas, Num_inputs, Num_x_Obsv, Num_x_Neurons,
                    Num_u_Obsv, Num_u_Neurons, Num_hidden_x,
                    Num_hidden_x, Num_hidden_u, Num_hidden_u)

# Training Loop Parameters
start_training_time = time.time()

eps = 50       # Number of epochs for final training
lr = 1e-3       # Learning rate
batch_size = 256
S_p = 30
T = len(train_tensor[0, :, :])
W = 0
M = 1  # Amount of models you want to run

if training_type == 'normal':
  [Lowest_loss,Models_loss_list, Best_Model, Lowest_loss_index, Running_Losses_Array, Lgx_Array, Lgu_Array, L3_Array, L4_Array, L5_Array, L6_Array] = trainingfcn(eps, lr, batch_size, S_p, T, alpha, Num_meas, Num_inputs, Num_x_Obsv, Num_x_Neurons, Num_u_Obsv, Num_u_Neurons, Num_hidden_x, Num_hidden_x, Num_hidden_u, Num_hidden_u, train_tensor, test_tensor, M, device=None)

elif training_type == 'mixed':
  [Lowest_loss, Models_loss_list, Best_Model, Lowest_loss_index,
  Running_Losses_Array, Lgx_unforced_Array, Lgu_forced_Array,
  L3_unforced_Array, L4_unforced_Array, L5_unforced_Array, L6_unforced_Array,
  L3_forced_Array, L4_forced_Array, L5_forced_Array, L6_forced_Array] = trainingfcn_mixed(eps, lr, batch_size, S_p, T, alpha,
                                                                          Num_meas, Num_inputs, Num_x_Obsv, Num_x_Neurons,
                                                                          Num_u_Obsv, Num_u_Neurons, Num_hidden_x,
                                                                          Num_hidden_x, Num_hidden_u, Num_hidden_u,
                                                                          train_tensor_unforced, train_tensor_forced, test_tensor_unforced,
                                                                          test_tensor_forced, M)

# Load the parameters of the best model
model.load_state_dict(torch.load(Best_Model))
print(f"Loaded model parameters from Model: {Best_Model}")

end_time = time.time()
total_time = end_time - start_time
total_training_time = end_time - start_training_time

print(f"Total time is: {total_time}")
print(f"Total training time is: {total_training_time}")

# ----- Result Plotting and Further Analysis -----
if training_type == 'normal':
  plot_results(model, val_tensor, train_tensor, S_p, Num_meas, Num_x_Obsv, T)

elif training_type == 'mixed':
  plot_results(model, val_tensor, train_tensor_forced, S_p, Num_meas, Num_x_Obsv, T)
