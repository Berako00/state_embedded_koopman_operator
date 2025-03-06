import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import os

from help_func import self_feeding, enc_self_feeding, set_requires_grad
from loss_func import total_loss, total_loss_forced, total_loss_unforced
from nn_structure import AUTOENCODER


