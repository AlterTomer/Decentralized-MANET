from utils import *
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
import numpy as np
import torch

# === DEVICE CONFIGURATION ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === CONFIGURATION ===
channel_path = None
num_samples = 10
n_list = [8] * num_samples
tx_list = [0] * num_samples
rx_list = [n - 1 for n in n_list]
sigma_list = [1] * num_samples
B = 6

# === LOAD DATASET ===
dataset = generate_graph_data(n_list, tx_list, rx_list, sigma_list, B, channel_path)

# If dataset is a list of Data objects, move each to device
for data in dataset:
    if hasattr(data, "to"):
        data.to(device)

train_size = int(0.8 * len(dataset))
train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
train_loader = DataLoader(train_dataset, batch_size=1)
val_loader = DataLoader(val_dataset, batch_size=1)

# === EVALUATE WITH CENTRALIZED ADAM ===
rate_train, p_opt_train = evaluate_centralized_adam(train_loader, B, device=device)
mean_train_rate = np.mean(rate_train)

rate_val, p_opt_val = evaluate_centralized_adam(val_loader, B, device=device)
mean_val_rate = np.mean(rate_val)

print(f"✅ Centralized Adam Train Rate: {mean_train_rate:.4f}")
print(f"✅ Centralized Adam Val Rate:   {mean_val_rate:.4f}")
