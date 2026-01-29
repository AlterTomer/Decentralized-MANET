import torch
from configparser import ConfigParser
from models.models import ChainedGNN
from utils.DataUtils import generate_graph_data
from utils.ComparisonUtils import evaluate_across_snr
from utils.EstimationUtils import masked_band_variance_from_dataset, precompute_csi_estimates
from utils.ConfigUtils import parse_args, load_ini_config
from visualization.GraphingAux import plot_mean_rate_vs_snr
import pickle

# ====== config ======
# args = parse_args()
# cfg_path = args.config.resolve()
# parser = load_ini_config(cfg_path)
# print(f"Loaded config from CLI: {cfg_path}")

cfg_path = r"C:\Users\alter\Desktop\PhD\Decentralized MANET\Config Files\Multicommodity\comp_multicommodity.ini"
parser = ConfigParser()
parser.read_file(open(cfg_path))
print(f"Loaded default config: {cfg_path}")

USE_AMP = torch.cuda.is_available()
# Training Parameters
train_params = parser["Train Parameters"]
SEED = int(train_params["SEED"])
MODE = train_params["mode"]  # "single" | "multicast" | "multi"
B = int(train_params["B"])
L = int(train_params["L"])
n = int(train_params["n"])
tx = int(train_params["tx"])
sigma = float(train_params["sigma"])
DROPOUT = float(train_params["dropout"])
num_samples = int(train_params["num samples"])
est_csi = True if int(train_params["LMMSE estimation"]) == 1 else False
# Files Parameters
files_params = parser["Files"]
try:
    channel_path = files_params["channel path"]
except KeyError:
    channel_path = None
fig_path = files_params["fig path"]
model_path = files_params["model path"]
fig_data_path = files_params["fig data path"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset & model (adapt to your paths)
n_list = [n] * num_samples
tx_list = [tx] * num_samples
# rx can be int OR a list in the ini; handle both
_raw_rx = train_params["rx"].strip()
if _raw_rx.startswith("[") and _raw_rx.endswith("]"):
    rx = [int(x) for x in _raw_rx[1:-1].replace(" ", "").split(",") if x]
elif "," in _raw_rx:
    rx = [int(x) for x in _raw_rx.replace(" ", "").split(",") if x]
else:
    rx = int(_raw_rx)

# replicate rx per sample (int or list)
if isinstance(rx, list):
    rx_list = [rx] * num_samples  # each sample may have a list of receivers
    K_cfg = len(rx)
else:
    rx_list = [rx] * num_samples
    K_cfg = 1

sigma_list = [sigma] * num_samples

dataset = generate_graph_data(
    n_list=n_list,
    tx_list=tx_list,
    rx_list=rx_list,
    sigma_list=sigma_list,
    B=B,
    K=K_cfg,
    problem=MODE,
    seed=SEED,
    channel_path=channel_path,
    device='cpu'
)

if est_csi:
    print("Using estimated CSI")
    prior_var = masked_band_variance_from_dataset(dataset)
    dataset = precompute_csi_estimates(
        dataset,
        pilots_M=4,
        pilot_power=1,
        prior_var=prior_var,
        est_noise_std=None,
        seed=SEED,
        device=torch.device('cpu'),
    )
else:
    print("Using True CSI")

# Choose K for the model:
# - single: K_model = 1
# - multicast: K_model = K_cfg (to enable per-receiver role channels; still one shared message)
# - multi: K_model = K_cfg (distinct messages, produces [B,K,n,n] + Z)
if MODE == "single":
    K_model = 1
elif MODE == "multicast":
    K_model = K_cfg
elif MODE == "multi":
    K_model = max(1, K_cfg)
else:
    raise ValueError("MODE must be 'single', 'multicast', or 'multi'.")
model = ChainedGNN(
    num_layers=L,
    B=B,
    K=K_model,
    problem=MODE,
    dropout=DROPOUT,
    use_jk=True,
    jk_mode="concat"
).to(device).eval()
ckpt = torch.load(model_path, map_location=device, weights_only=False)

state_dict = ckpt["model_state_dict"]
new_state_dict = {}

for k, v in state_dict.items():
    # remap old names -> new names
    if k.startswith("head."):
        new_k = k.replace("head.", "p_head.")
        new_state_dict[new_k] = v
    else:
        new_state_dict[k] = v
# model.load_state_dict(ckpt["model_state_dict"])
model.load_state_dict(new_state_dict, strict=False)


g = torch.Generator().manual_seed(SEED)

snr_db_list = list(range(0, 21, 1))
results = evaluate_across_snr(dataset, model, B, snr_db_list,problem=MODE)
with open(fig_data_path, "wb") as file:
    pickle.dump(results, file)

plot_mean_rate_vs_snr(snr_db_list, results, save_path=fig_path)

