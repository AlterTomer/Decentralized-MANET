import os
from DataUtils import generate_graph_data
from CentralizedUtils import compute_lower_bound_rate
import csv
import torch
import numpy as np

# === DEVICE CONFIGURATION ===
device = 'cpu'

# === DATA PREPARATION ===
channel_path = None
num_samples = 3
n_list = [8] * num_samples
tx_list = [0] * num_samples
rx_list = [n - 1 for n in n_list]
sigma_list = [1] * num_samples
B = 6

# === GENERATE DATASET ===
dataset = generate_graph_data(n_list, tx_list, rx_list, sigma_list, B, seed=1000, channel_path=None, device=device)

# === COMPUTE LOWER BOUNDS ===
print('lower bound')
lower_bounds, _ = compute_lower_bound_rate(dataset)
print(lower_bounds)
print(np.mean(lower_bounds))
# === OPTIONAL: SAVE RESULTS TO CSV ===
filename = f'NEW lower_bounds num samples {num_samples} starter seed 1000.csv'
os.chdir(r'C:\Users\alter\Desktop\PhD')
with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Graph Index", "Lower Bound Rate"])
    for i, rate in enumerate(lower_bounds):
        writer.writerow([i, rate])
