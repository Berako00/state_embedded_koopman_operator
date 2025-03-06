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


