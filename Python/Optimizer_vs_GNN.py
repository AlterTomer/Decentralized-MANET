import os
from torch.utils.data import Subset
import random
from models import GraphNet
from utils import *
from time import time

lower_bound_rates = []
gnn_rates = []
adam_rates = []


sigma_db = np.array(range(-10, 11))  # AWGN variance dB
sigma_values = 10 ** (sigma_db / 10)  # AWGN variance linear
channel_power = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
chk_path = r"C:\Users\User\Desktop\MS.c\Ph.D\Distributed OFDMA MANETs\Trained Models\GCN\13 layers gated network 2025-06-26_00-22-42.pth"
model_data = torch.load(chk_path, weights_only=False)
state_dict = model_data['model_state_dict']
dataset = model_data['dataset']
layer_sizes = model_data['layer sizes']
layer_types = model_data['layer types']
B = model_data['B']

# Create a test dataset
n = 50
indices = random.sample(range(len(dataset)), n)  # random n indices
test_dataset = Subset(dataset, indices)

# Load pretrained model
model = GraphNet(layer_sizes=layer_sizes, layer_types=layer_types, B=B, dropout=0.2).to(device)
model.load_state_dict(state_dict)
model.eval()

for s_idx, noise_var in enumerate(sigma_values):
    print(f'AWGN var = {10 * np.log10(noise_var)} dB')
    sigma = noise_var ** 0.5
    adam_rates_sigma = []
    gnn_rates_sigma = []

    # Lower Bounds
    lb_rates = compute_lower_bound_rate(test_dataset, sigma)
    lower_bound_rates.append(np.mean(lb_rates))
    t0 = time()
    for idx, data in enumerate(test_dataset):
        if s_idx == 0:
            channel_power += torch.std(data.links_matrix).item()
        print(idx)
        adj = data.adj_matrix
        links = data.links_matrix
        tx = data.tx
        rx = data.rx
        paths = find_all_paths(adj, tx, rx)
        if len(paths) < 1:
            continue

        p_arr = torch.stack([create_normalized_tensor(adj.shape[0], adj.shape[1], mask=adj) for _ in range(B)])
        p_arr = nn.Parameter(p_arr, requires_grad=True)
        data = data.to(device)

        # ADAM Optimizer (Centralized)
        optimizer = optim.Adam([p_arr], lr=0.1)
        p_opt = classic_opt(50, optimizer, links, p_arr, sigma, paths, B)
        adam_rate = calc_sum_rate(links, p_opt, sigma, paths, model.B)
        adam_rates_sigma.append(adam_rate.item())

        # GNN (Distributed)
        with torch.no_grad():
            out = model(data)
            p_opt_gnn = expand_power_allocation(out, adj)
            gnn_rate = calc_sum_rate(links, p_opt_gnn, sigma, paths, B)
            gnn_rates_sigma.append(gnn_rate.item())


    adam_rates.append(np.mean(adam_rates_sigma))
    gnn_rates.append(np.mean(gnn_rates_sigma))
    print(f'{(time() - t0) / 60} mins')
    print('-----------------------------')



channel_power = np.log10(channel_power)
snr_db = channel_power - sigma_db
plt.plot(snr_db, lower_bound_rates, marker='*', label='Lower Bounds')
plt.plot(snr_db, gnn_rates, marker='o', label='Learned Decentralized Optimization', linestyle='dashed')
plt.plot(snr_db, adam_rates, marker='d', label='Learned Centralized Optimization', linestyle='dotted')

plt.grid()
plt.xlabel('SNR [dB]', fontsize=14)
plt.yscale('log')
plt.ylabel('Mean Rate (log scale)', fontsize=14)
plt.legend(loc='best', fontsize=14)

plt.tight_layout()
os.chdir(r"C:\Users\User\Desktop\MS.c\Ph.D\Distributed OFDMA MANETs\figs")
plt.savefig('Comparison.png')
plt.close()

print(f'Lower Bounds: {lower_bound_rates}')
print(f'GNN Rates: {gnn_rates}')
print(f'ADAM Rates: {adam_rates}')