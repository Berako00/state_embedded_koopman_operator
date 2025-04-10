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
import math
import openpyxl

from help_func import self_feeding, enc_self_feeding, load_model
from nn_structure import AUTOENCODER
from training import trainingfcn
from data_generation import DataGenerator, TwoLinkRobotDataGenerator

from plotting import plot_results, plot_debug, plot_losses
from ga_optimizer import run_genetic_algorithm

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

start_time = time.time()

# ---- System Params ----------
Num_meas = 4
Num_inputs = 2
system = 'two_link'     # 'two_link' or 'simple'
# ----------------------------

# ------- Data Generation Params ----------
numICs = 20000
T_step = 50
dt = 0.02
seed = 1

if system == 'simple':
  x1range = (-0.5, 0.5)
  x2range = x1range
  mu = -0.05
  lam = -1
elif system == 'two_link':
  q1_range = (-math.pi, math.pi)
  q2_range = (-math.pi, math.pi)
  dq1_range = (-1, 1)
  dq2_range = dq1_range
  tau_max = 7.5
# -----------------------------------------

if system == 'simple':
  [train_tensor, test_tensor, val_tensor] = DataGenerator(x1range, x2range, numICs, mu, lam, T_step, dt)

elif system == 'two_link':
  [train_tensor, test_tensor, val_tensor] = TwoLinkRobotDataGenerator(q1_range, q2_range, dq1_range, dq2_range, numICs, T_step, dt, tau_max)

# ---- GA Params -------------
use_ga = True
generations = 2
pop_size = 2
eps = 10
tournament_size = 2
mutation_rate = 0.2

# Define parameter ranges For GA
param_ranges = {
    "Num_x_Obsv": (10, 100),
    "Num_u_Obsv": (10, 100),
    "Num_x_Neurons": (128, 128),
    "Num_u_Neurons": (128, 128),
    "Num_hidden_x": (3, 3),  # Shared for both x encoder and decoder
    "Num_hidden_u": (3, 3),  # Shared for both u encoder and decoder
    "alpha0": (0.001, 0.1),
    "alpha1": (1e-9, 1e-5),
    "alpha2": (1e-18, 1e-12)
}
# ------------------------------

# ---- Define last training param -------
eps_final = 200      # Number of epochs for final training
breakout = 10
check_epoch = 10
lr = 1e-3       # Learning rate
batch_size = 256
S_p = 30
T = len(train_tensor[0, :, :])
W = 0
M = 1  # Amount of models you want to run

if not use_ga:
    Num_x_Obsv    = 17
    Num_u_Obsv    = 18
    Num_x_Neurons = 45
    Num_u_Neurons = 50
    Num_hidden_x  = 1
    Num_hidden_u  = 1
    alpha         = [0.656327, 5.1215e-6, 5.7252e-13]
# ---------------------------------------


print(f"Train tensor shape: {train_tensor.shape}")
print(f"Test tensor shape: {test_tensor.shape}")
print(f"Validation tensor shape: {val_tensor.shape}")

# --- Genetic Algorithm Hyperparameter Optimization ---
if use_ga:
    # For speed, use a lower number of epochs for evaluation (eps) and fewer generations/population size.
    best_params = run_genetic_algorithm(check_epoch, Num_meas, Num_inputs, train_tensor, test_tensor, tournament_size, mutation_rate, generations, pop_size, eps, lr, batch_size, S_p, T, dt, param_ranges=param_ranges, elitism_count=1)

    Num_meas      = best_params['Num_meas']
    Num_inputs    = best_params['Num_inputs']
    Num_x_Obsv    = best_params['Num_x_Obsv']
    Num_u_Obsv    = best_params['Num_u_Obsv']
    Num_x_Neurons = best_params['Num_x_Neurons']
    Num_u_Neurons = best_params['Num_u_Neurons']
    Num_hidden_x  = best_params['Num_hidden_x']
    Num_hidden_u  = best_params['Num_hidden_u']
    alpha         = [best_params['alpha0'], best_params['alpha1'], best_params['alpha2']]



model = AUTOENCODER(Num_meas, Num_inputs, Num_x_Obsv, Num_x_Neurons,
                    Num_u_Obsv, Num_u_Neurons, Num_hidden_x,
                    Num_hidden_u, Num_hidden_u)

# Training Loop Parameters
start_training_time = time.time()


[Lowest_loss, Models_loss_list, Best_Model, Lowest_loss_index, Running_Losses_Array, Lgu_Array, L4_Array, L6_Array] = trainingfcn(eps_final, check_epoch, lr, batch_size, S_p, T, dt, alpha, Num_meas, Num_inputs, Num_x_Obsv, Num_x_Neurons, Num_u_Obsv, Num_u_Neurons, Num_hidden_x, Num_hidden_u, Num_hidden_u, train_tensor, test_tensor, M, device=None)

ind_loss = int(Lowest_loss_index)
Lgu = np.asarray(Lgu_Array[ind_loss])
L4 = np.asarray(L4_Array[ind_loss])
L6 = np.asarray(L6_Array[ind_loss])
Running_Losses = np.asarray(Running_Losses_Array[ind_loss])

# Create a dictionary with each array as a column
data = {
    "Lgu": Lgu,
    "L4": L4,
    "L6": L6,
    "Running_Losses": Running_Losses
}

# Create a DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to an Excel file (all data in one sheet, different columns)
df.to_excel("training_results.xlsx", index=False)

# Load the parameters of the best model
load_model(model, Best_Model, device)
print(f"Loaded model parameters from Model: {Best_Model}")

end_time = time.time()
total_time = end_time - start_time
total_training_time = end_time - start_training_time

print(f"Total time is: {total_time}")
print(f"Total training time is: {total_training_time}")

# ----- Result Plotting and Further Analysis -----
plot_debug(model, val_tensor, train_tensor, S_p, Num_meas, Num_x_Obsv, T)
plot_results(model, val_tensor, train_tensor, S_p, Num_meas, Num_x_Obsv, T)

