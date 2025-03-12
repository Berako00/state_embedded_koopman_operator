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
from Data_Generation import DataGenerator

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

# --- Genetic Algorithm Hyperparameter Optimization ---
use_ga = True
if use_ga:
    from ga_optimizer import run_genetic_algorithm
    # For speed, use a lower number of epochs for evaluation (eps) and fewer generations/population size.
    best_params = run_genetic_algorithm(train_tensor, test_tensor, generations=3, pop_size=5, eps=50)
    
    Num_meas             = best_params['Num_meas']
    Num_inputs           = best_params['Num_inputs']
    Num_x_Obsv           = best_params['Num_x_Obsv']
    Num_u_Obsv           = best_params['Num_u_Obsv']
    Num_x_Neurons        = best_params['Num_x_Neurons']
    Num_u_Neurons        = best_params['Num_u_Neurons']
    Num_hidden_x_encoder = best_params['Num_hidden_x_encoder']
    Num_hidden_x_decoder = best_params['Num_hidden_x_decoder']
    Num_hidden_u_encoder = best_params['Num_hidden_u_encoder']
    Num_hidden_u_decoder = best_params['Num_hidden_u_decoder']
    alpha                = [best_params['alpha0'], best_params['alpha1'], best_params['alpha2']]
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

# --- Instantiate the model using the (possibly optimized) hyperparameters ---
model = AUTOENCODER(Num_meas, Num_inputs, Num_x_Obsv, Num_x_Neurons,
                    Num_u_Obsv, Num_u_Neurons, Num_hidden_x_encoder,
                    Num_hidden_x_decoder, Num_hidden_u_encoder, Num_hidden_u_decoder)
                    
# Training Loop Parameters
start_training_time = time.time()

eps = 500       # Number of epochs for final training
lr = 1e-3       # Learning rate
batch_size = 256
S_p = 30
T = len(train_tensor[0, :, :])
W = 0
M = 1  # Amount of models you want to run

[Lowest_loss, Lowest_test_loss, Best_Model, Lowest_test_loss_index, 
 Lgx_Array, Lgu_Array, L3_Array, L4_Array, L5_Array, L6_Array] = trainingfcn(
    eps, lr, batch_size, S_p, T, alpha, 
    Num_meas, Num_inputs, Num_x_Obsv, Num_x_Neurons,
    Num_u_Obsv, Num_u_Neurons, Num_hidden_x_encoder, 
    Num_hidden_x_decoder, Num_hidden_u_encoder, Num_hidden_u_decoder,
    train_tensor, test_tensor, M)

# Load the parameters of the best model
model.load_state_dict(torch.load(Best_Model))
print(f"Loaded model parameters from Model: {Best_Model}")

end_time = time.time()
total_time = end_time - start_time
total_training_time = end_time - start_training_time

print(f"Total time is: {total_time}")
print(f"Total training time is: {total_training_time}")

# ----- Result Plotting and Further Analysis -----
# [Your plotting code remains unchanged below...]
