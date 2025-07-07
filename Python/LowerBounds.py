import os
from utils import *
import csv

# Prepare data
num_samples = 1000
# n_list = random.sample(range(4, 10), num_samples)
n_list = [8] * num_samples
tx_list = [0] * num_samples
rx_list = []
sigma_list = [1] * num_samples
B = 6
for n in n_list:
    rx_list.append(n - 1)
dataset = generate_graph_data(n_list, tx_list, rx_list, sigma_list, B)
lower_bounds = compute_lower_bound_rate(dataset)
print(lower_bounds)
filename = 'lower_bounds.csv'
os.chdir(r"C:\Users\User\Desktop\MS.c\Ph.D\Distributed OFDMA MANETs")
with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Graph Index", "Lower Bound Rate"])
    for i, rate in enumerate(lower_bounds):
        writer.writerow([i, rate])