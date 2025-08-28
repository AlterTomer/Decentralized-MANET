import torch
from torch.optim.swa_utils import AveragedModel
from models.models import ChainedGNN
from utils.DataUtils import generate_graph_data
from utils.ComparisonUtils import evaluate_across_snr
from configparser import ConfigParser
from visualization.GraphingAux import plot_mean_rate_vs_snr
from utils.CentralizedUtils import compute_equal_power_bound

# ====== config ======
# args = parse_args()
# cfg_path = args.config.resolve()
# parser = load_ini_config(cfg_path)
cfg_path = r"C:\Users\alter\Desktop\PhD\Decentralized MANET\Config Files\comp LinTAU Quadriga B_6 L_3 seed_1337.ini"
parser = ConfigParser()
parser.read_file(open(cfg_path))

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
model_path = files_params["model path"]
swa_path = files_params["swa path"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset & model (adapt to your paths)
n_list = [n] * num_samples
tx_list = [tx] * num_samples
rx_list = [n - 1 for n in n_list]
sigma_list = [sigma] * num_samples

dataset = generate_graph_data(
    n_list=n_list,
    tx_list=tx_list,
    rx_list=rx_list,
    sigma_list=sigma_list,
    B=B,
    seed=SEED,
    channel_path=channel_path,
    device='cpu',
)
rate_bounds, p_arr = compute_equal_power_bound(dataset)
model = ChainedGNN(num_layers=L, B=B, dropout=DROPOUT, use_jk=True, jk_mode="concat").to(device).eval()
# swa = AveragedModel(model).to(device)
# swa.num_layers = L
# swa.B = B
swa = None

ckpt = torch.load(model_path, map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])

# swa_ckpt = torch.load(swa_path, map_location=device, weights_only=False)
# swa.load_state_dict(swa_ckpt["model_state_dict"])

snr_db_list = list(range(-10, 12, 1))  # -10, -8, ..., 10
# snr_db_list = [-4, -2]
results = evaluate_across_snr(dataset, model, swa, B, snr_db_list)
plot_mean_rate_vs_snr(snr_db_list, results, save_path=fig_path)
a = 0


