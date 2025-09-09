import torch
from configparser import ConfigParser
from models.models import ChainedGNN
from utils.DataUtils import generate_graph_data
from utils.ComparisonUtils import est_true_model_compare
from utils.EstimationUtils import masked_band_variance_from_dataset, precompute_csi_estimates
from utils.ConfigUtils import parse_args, load_ini_config
from visualization.GraphingAux import est_true_model_compare_plot

# ====== config ======
# try:
#     args = parse_args()
#     cfg_path = args.config.resolve()
#     parser = load_ini_config(cfg_path)
#     print(f"Loaded config from CLI: {cfg_path}")

# except Exception as e:
    # print(f"⚠️ Failed to load CLI config ({e}), falling back to default path...")
cfg_path = r"C:\Users\alter\Desktop\PhD\Decentralized MANET\Config Files\comp B_6 L_3 seed_1337 n_8 est_vs_true.ini"
parser = ConfigParser()
parser.read_file(open(cfg_path))
print(f"Loaded default config: {cfg_path}")

USE_AMP = torch.cuda.is_available()
# Training Parameters
train_params = parser["Train Parameters"]
SEED = int(train_params["SEED"])
B = int(train_params["B"])
L = int(train_params["L"])
n = int(train_params["n"])
tx = int(train_params["tx"])
sigma = float(train_params["sigma"])
DROPOUT = float(train_params["dropout"])
num_samples = int(train_params["num samples"])

# Files Parameters
files_params = parser["Files"]
try:
    channel_path = files_params["channel path"]
except KeyError:
    channel_path = None
fig_path = files_params["fig path"]
true_model_path = files_params["true model path"]
est_model_path = files_params["est model path"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset & models (dataset based on the larger topology)
n_list = [n] * num_samples
tx_list = [tx] * num_samples
rx_list = [n - 1 for n in n_list]
sigma_list = [sigma] * num_samples

true_dataset = generate_graph_data(
    n_list=n_list,
    tx_list=tx_list,
    rx_list=rx_list,
    sigma_list=sigma_list,
    B=B,
    seed=SEED,
    channel_path=channel_path,
    device='cpu'
)

prior_var = masked_band_variance_from_dataset(true_dataset)
est_dataset = precompute_csi_estimates(
    true_dataset,
    pilots_M=4,
    pilot_power=1,
    prior_var=prior_var,
    est_noise_std=None,
    seed=SEED,
    device=device,
)
# Trained model for a given topology
true_model = ChainedGNN(num_layers=L, B=B, dropout=DROPOUT, use_jk=True, jk_mode="concat").to(device).eval()

ckpt = torch.load(true_model_path, map_location=device, weights_only=False)
true_model.load_state_dict(ckpt["model_state_dict"])

# Trained model of a different topology
est_model = ChainedGNN(num_layers=L, B=B, dropout=DROPOUT, use_jk=True, jk_mode="concat").to(device).eval()

ckpt = torch.load(est_model_path, map_location=device, weights_only=False)
est_model.load_state_dict(ckpt["model_state_dict"])

snr_db_list = list(range(-10, 12, 1))
results = est_true_model_compare(true_dataset, est_dataset, true_model, est_model, snr_db_list)
est_true_model_compare_plot(snr_db_list, results, save_path=fig_path)
