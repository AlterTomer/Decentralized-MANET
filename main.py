import torch
import random
import numpy as np
from utils.DataUtils import generate_graph_data
from utils.CentralizedUtils import evaluate_centralized_adam_single
from models.models import ChainedGNN
from utils.PathUtils import find_all_paths, paths_to_tensor
from models.GraphNetAux import _compute_rates_per_layer
# === DATA PREPARATION ===
channel_path = None
num_samples = 4000
n_list = [8] * num_samples
tx_list = [0] * num_samples
rx_list = [n - 1 for n in n_list]
sigma_list = [1] * num_samples
B = 6
SEED = 1337

# ====== seeding ======
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
# === GENERATE DATASET ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ChainedGNN(num_layers=3, B=B, dropout=0.2, use_jk=True, jk_mode="concat").to(device).eval()
model_path = r"C:\Users\alter\Desktop\PhD\Decentralized MANET\Trained Models\Quadriga_Lin_TAU_ChainedGNN_L3_val0.152_2025-08-24_09-27-58.pth"
ckpt = torch.load(model_path, map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
channel_path = r"C:\Users\alter\Desktop\PhD\Decentralized MANET\Quadriga channels\H_seed_1337_n_8_B_6_samples_4000.mat"
dataset = generate_graph_data(n_list, tx_list, rx_list, sigma_list, B, seed=SEED, channel_path=channel_path, device='cpu')
lb_wins = 0
adam_wins = 0
gnn_wins = 0
tie = 0

gnn_rates = []
adamw_rates = []
for data in dataset:
    rates = []
    sigma = data.sigma
    adj = data.adj_matrix
    links = data.links_matrix
    tx = data.tx
    rx = data.rx
    rate, _ = evaluate_centralized_adam_single(data, B, lr=0.05, num_iterations=100)
    d = data.to(device)
    paths = find_all_paths(d.adj_matrix, d.tx, d.rx)
    paths = paths_to_tensor(paths, device)
    gnn_rates, _ = _compute_rates_per_layer(model, d, paths)
    gnn_rate = max(gnn_rates)
    if gnn_rate == rate:
        tie += 1
    rates.append(rate)
    rates.append(gnn_rate)
    argmax = rates.index(max(rates))
    if argmax == 0:
        adam_wins += 1
    else:
        gnn_wins += 1
    gnn_rates.append(gnn_rate.item())
    adamw_rates.append(rate.item())
    print(f'ADAM rate: {rate: .4f}| GNN rate: {gnn_rate: .4f}')
mean_gnn = sum(gnn_rates) / len(gnn_rates)
mean_adam = sum(adamw_rates) / len(adamw_rates)
print(f'gnn mean: {mean_gnn: .4f}, adam mean: {mean_adam: .4f}')
print(f'adam wins: {adam_wins}, gnn wins: {gnn_wins}, lb wins: {lb_wins} ,ties: {tie}')


