import os
import random
import numpy as np
import torch
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from torch.amp import GradScaler
from torch.optim.lr_scheduler import LambdaLR
from pathlib import Path
from configparser import ConfigParser
from utils.ConfigUtils import parse_args, load_ini_config
from models.models import ChainedGNN
from models.GraphNetAux import train_chained, validate_chained, tau_linear
from utils.DataUtils import generate_graph_data
from utils.TrainUtils import cosine_warm_restart_lambda
from utils.FilesUtils import save_best_ckpt
from utils.EstimationUtils import masked_band_variance_from_dataset, precompute_csi_estimates, build_estimate_lookup
from visualization.GraphingAux import plot_train_valid_loss
from time import time

# ====== config ======
# args = parse_args()
# cfg_path = args.config.resolve()
# parser = load_ini_config(cfg_path)
# print(f"Loaded config from CLI: {cfg_path}")

cfg_path = r"C:\Users\alter\Desktop\PhD\Decentralized MANET\Config Files\ChainedGNN Estimated Rayleigh B_6 L_3 seed_1337.ini"
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
LR = float(train_params["lr"])
WEIGHT_DECAY = float(train_params["wd"])
GRAD_CLIP = float(train_params["grad clip"])
MAX_EPOCHS = int(train_params["epochs"])
supervised_epochs = int(train_params["supervised epochs"])
num_samples = int(train_params["num samples"])
grad_batch = int(train_params["grad batch"])
MONO = float(train_params["mono"])
sup_loss_mode = train_params["supervised loss mode"]
est_csi = True if int(train_params["LMMSE estimation"]) == 1 else False

# Files Parameters
files_params = parser["Files"]
try:
    channel_path = files_params["channel path"]
except KeyError:
    channel_path = None
CKPT_DIR = files_params["ckpt dir"]
figs_dir = files_params["figs dir"]
include_training_state = True if int(files_params["include training state"]) == 1 else False
ckpt_prefix = files_params["prefix"]

CKPT_DIR = Path(CKPT_DIR)
figs_dir = Path(figs_dir)
CKPT_DIR.mkdir(parents=True, exist_ok=True)
figs_dir.mkdir(parents=True, exist_ok=True)


# ====== seeding ======
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ====== dataset ======
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


# splits
g = torch.Generator().manual_seed(SEED)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size], generator=g)
train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_set,   batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

if est_csi:
    print("Estimated CSI Model")
    # Train
    prior_var = masked_band_variance_from_dataset(train_set)
    est_train = precompute_csi_estimates(
        train_set,
        pilots_M=4,
        pilot_power=1,
        prior_var=prior_var,
        est_noise_std=None,
        seed=SEED,
        device=device,
    )
    train_est_lookup = build_estimate_lookup(est_train)

    # Validation
    prior_var = masked_band_variance_from_dataset(val_set)
    est_val = precompute_csi_estimates(
        val_set,
        pilots_M=4,
        pilot_power=1,
        prior_var=prior_var,
        est_noise_std=None,
        seed=SEED,
        device=device,
    )
    val_est_lookup = build_estimate_lookup(est_val)
else:
    train_est_lookup = None
    val_est_lookup = None
# ====== model / optim / amp ======
model = ChainedGNN(num_layers=L, B=B, dropout=DROPOUT, use_jk=True, jk_mode="concat").to(device)

# parameter groups (decay vs no_decay)
decay, no_decay = [], []
for name, p in model.named_parameters():
    if not p.requires_grad:
        continue
    if name.endswith(".bias") or "norm" in name.lower() or "layernorm" in name.lower():
        no_decay.append(p)
    else:
        decay.append(p)

optimizer = torch.optim.AdamW(
    [{"params": decay, "weight_decay": WEIGHT_DECAY},
     {"params": no_decay, "weight_decay": 0.0}],
    lr=LR, betas=(0.9, 0.999), eps=1e-8
)

# cosine warm restart schedule
base_lr = LR
total_epochs = MAX_EPOCHS
eta_min = 1e-5
lr_warmup_epochs = 5
scheduler = LambdaLR(optimizer, lr_lambda=cosine_warm_restart_lambda(lr_warmup_epochs, total_epochs, eta_min, base_lr))

scaler = GradScaler(enabled=USE_AMP)

# ====== checkpoints ======
print(f">>> {num_samples} samples simulation | seed={SEED} | AMP={USE_AMP} | grad_clip={GRAD_CLIP}")

os.makedirs(CKPT_DIR, exist_ok=True)
best_val = -float("inf")
best_ckpt = None

train_loss_arr = np.zeros(MAX_EPOCHS)
val_rate_arr = np.zeros(MAX_EPOCHS)
t0 = time()

# ====== training loop ======
for epoch in range(MAX_EPOCHS):
    epoch_tau = tau_linear(epoch, MAX_EPOCHS)
    print(f">>> Epoch {epoch} | tau={epoch_tau}")
    t1 = time()
    # --- train ---
    stats = train_chained(
        model,
        train_loader,          # loader over TRUE-CSI dataset
        optimizer,
        epoch,
        mono_weight=MONO,
        use_amp=True,
        grad_clip=GRAD_CLIP,
        grad_accum_steps=grad_batch,
        tau=epoch_tau,
        est_dataset=train_est_lookup # or est_ds; pass None for full-CSI training
    )


    # --- step LR ---
    scheduler.step()
    val_stats = validate_chained(
    model,
    val_loader,
    device=device,
    est_dataset=val_est_lookup,
    verbose=True,
    )

    # --- log ---
    train_loss_arr[epoch] = stats["loss"]
    val_rate_arr[epoch] = val_stats["best_rate"]
    print(
        f"E{epoch:02d} | "
        f"Time {(time() - t1) / 60: .3f} mins"
        f"train_loss={stats['loss']:.6f} | "
        f"val_best={val_stats['best_rate']:.6f} | "
    )

    # --- track best ---
    if val_stats["best_rate"] > best_val:
        old = best_ckpt
        best_val = val_stats["best_rate"]
        best_ckpt = save_best_ckpt(
            model=model, epoch=epoch, best_val=best_val, cfg_path=cfg_path,
            ckpt_dir=CKPT_DIR, prefix=ckpt_prefix, include_training_state=include_training_state,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )
        if old and os.path.exists(old):
            try:
                os.remove(old)
                print(f"🗑️ Deleted previous checkpoint: {old}")
            except Exception as e:
                print(f"⚠️ Failed to delete {old}: {e}")
        print(f"✅ New best model saved: {best_ckpt}")

# ====== final evals ======
print(f'Training time = {(time() - t0) / 60} mins')

os.chdir(figs_dir)
plot_train_valid_loss(train_loss_arr, val_rate_arr, filename=f'Train {ckpt_prefix}_{L} layers {B} bands network.png')