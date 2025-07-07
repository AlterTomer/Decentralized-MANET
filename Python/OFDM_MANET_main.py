import os
from time import time
from torch_geometric.loader import DataLoader
from models import GraphNet, ChainedGNN
from GraphNetAux import train_GraphNet, validate_GraphNet, train_chained, validate_chained
from GraphingAux import plot_train_valid_loss
from utils import *
from datetime import datetime
from torch.utils.data import random_split
import torch
import numpy as np

# === CONFIG ===
use_chained = True  # 🔁 Set to False to use GraphNet (regular), True to use ChainedGNN (layer-wise)
channel_path = None  # Path to a .mat file with pre-generated channels
model_dir = r"C:\Users\User\Desktop\MS.c\Ph.D\Distributed OFDMA MANETs\Trained Models\GCN"
figs_dir = r'C:\Users\User\Desktop\MS.c\Ph.D\Distributed OFDMA MANETs\figs'
num_samples = 1000
n_list = [8] * num_samples
tx_list = [0] * num_samples
rx_list = [n - 1 for n in n_list]
sigma_list = [1] * num_samples
B = 6
num_epochs = 50

# === DATASET ===
dataset = generate_graph_data(n_list, tx_list, rx_list, sigma_list, B, channel_path)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
layer_sizes = [B] * 10
layer_types = ['gated'] * (len(layer_sizes) - 1)

# === MODEL & OPTIMIZER ===
if use_chained:
    model = ChainedGNN(layer_sizes=layer_sizes, layer_types=layer_types, B=B, dropout=0.2).to(device)
    optimizers = [torch.optim.Adam(layer.parameters(), lr=1e-3) for layer in model.layers]
else:
    model = GraphNet(layer_sizes=layer_sizes, layer_types=layer_types, B=B, dropout=0.2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# === INIT POWER (unused in ChainedGNN) ===
n = n_list[0]
p_init = torch.full((B, n, n), 0.0, device=device)
for i in range(n):
    for b in range(B):
        p_init[b, i, i] = 1.0 / (B ** 0.5)

# === TRAINING ===
train_loss_arr = np.zeros(num_epochs)
val_rate_arr = np.zeros(num_epochs)
t0 = time()
best_val_rate = -float('inf')
best_checkpoint_path = None

for epoch in range(num_epochs):
    if use_chained:
        train_loss = train_chained(model, train_loader, optimizers, device)
        val_rate = validate_chained(model, val_loader, device)
    else:
        train_loss = train_GraphNet(model, train_loader, optimizer, device)
        val_rate = validate_GraphNet(model, val_loader, device)

    train_loss_arr[epoch] = train_loss
    val_rate_arr[epoch] = val_rate
    print(f"Epoch {epoch:02d} | Loss: {train_loss:.4f} | Val Rate: {val_rate:.4f}")

    if val_rate > best_val_rate:
        if best_checkpoint_path is not None:
            try:
                os.remove(best_checkpoint_path)
                print(f"🗑️ Deleted previous checkpoint: {best_checkpoint_path}")
            except Exception as e:
                print(f"⚠️ Failed to delete {best_checkpoint_path}: {e}")

        best_val_rate = val_rate
        formatted = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        model_type = "Chained" if use_chained else "GraphNet"
        filename = f"{model_type}_{len(layer_sizes)} layers {layer_types[0]} network {formatted}.pth"
        checkpoint_path = os.path.join(model_dir, filename)

        torch.save({
            'model_state_dict': model.state_dict(),
            'epoch': epoch + 1,
            'dataset': dataset,
            'layer sizes': layer_sizes,
            'layer types': layer_types,
            'B': B
        }, checkpoint_path)

        best_checkpoint_path = checkpoint_path
        print(f"✅ New best model saved: {checkpoint_path}")

t1 = time()
print(f'Training time = {(t1 - t0) / 60:.2f} minutes')

# === PLOT RESULTS ===
os.chdir(figs_dir)
model_type = "Chained" if use_chained else "GraphNet"
plot_train_valid_loss(train_loss_arr, -val_rate_arr, filename=f'{model_type}_{len(layer_sizes)} layers {layer_types[0]} network.png')
