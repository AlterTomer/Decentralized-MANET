import os
import random
import numpy as np
import torch
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from torch.amp import GradScaler
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import LambdaLR
from pathlib import Path
from configparser import ConfigParser
from models import ChainedGNN
from GraphNetAux import train_chained, validate_chained, tau_linear
from DataUtils import generate_graph_data
from TrainUtils import cosine_warm_restart_lambda
from FilesUtils import save_best_ckpt
from EstimationUtils import masked_band_variance_from_dataset
from GraphingAux import plot_train_valid_loss
from time import time

# ====== config ======
# args = parse_args()
# cfg_path = args.config.resolve()
# parser = load_ini_config(cfg_path)
cfg_path = r"C:\Users\alter\Desktop\PhD\Decentralized MANET\Config Files\ChainedGNN Estimated Rayleigh B_6 L_3 seed_1337.ini"
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
LR = float(train_params["lr"])
WEIGHT_DECAY = float(train_params["wd"])
GRAD_CLIP = float(train_params["grad clip"])
MAX_EPOCHS = int(train_params["epochs"])
supervised_epochs = int(train_params["supervised epochs"])
SWA_START_FRAC = float(train_params["swa start frac"])
num_samples = int(train_params["num samples"])
grad_batch = int(train_params["grad batch"])
MONO = float(train_params["mono"])
sup_loss_mode = train_params["supervised loss mode"]
swa_enabled = True if int(train_params["swa enabled"]) == 1 else False
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

# sanity: B matches
sample0 = dataset[0]
assert sample0.x.shape[0] == B, f"B mismatch: data {sample0.x.shape[0]} vs config {B}"

# splits
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_set,   batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

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

# SWA
swa_start = int(SWA_START_FRAC * MAX_EPOCHS)
swa_model = AveragedModel(model).to(device)
swa_model.B = getattr(model, "B", B)
swa_model.num_layers = getattr(model, "num_layers", L)
swa_scheduler = None  # created once we hit swa_start

# ====== checkpoints ======
print(f">>> {num_samples} samples simulation | seed={SEED} | AMP={USE_AMP} | grad_clip={GRAD_CLIP}")

os.makedirs(CKPT_DIR, exist_ok=True)
best_val = -float("inf")
best_ckpt = None

best_swa_val = -float("inf")
best_swa_ckpt = None

train_loss_arr = np.zeros(MAX_EPOCHS)
val_rate_arr = np.zeros(MAX_EPOCHS)
t0 = time()
prior_var = masked_band_variance_from_dataset(train_set)
# ====== training loop ======
for epoch in range(MAX_EPOCHS):
    epoch_tau = tau_linear(epoch, MAX_EPOCHS)
    print(f">>> Epoch {epoch} | tau={epoch_tau}")
    t1 = time()
    # --- train ---
    stats = train_chained(
        model,
        train_loader,
        optimizer,
        epoch,
        warmup=supervised_epochs,                 # no supervised warmup in this run
        mono_weight=MONO,                         # set >0.0 to encourage monotonicity
        use_amp=USE_AMP,
        scaler=scaler,
        grad_clip=GRAD_CLIP,
        grad_accum_steps=grad_batch,
        tau=epoch_tau,
        use_csi_estimation=True,
        est_noise_std=None,
        pilots_M=4,
        pilot_power=1,
        prior_var=prior_var,
    )


    # --- step LR or do SWA ---
    if epoch < swa_start:
        scheduler.step()
    else:
        if swa_scheduler is None:
            swa_scheduler = SWALR(optimizer, swa_lr=2e-4, anneal_epochs=5, anneal_strategy="cos")
        swa_scheduler.step()
        swa_model.update_parameters(model)
        swa_enabled = True

    # --- validation (rate-based) ---
    val_stats = validate_chained(
        model,
        val_loader,
        device=device,
        csv_path=None,
        epoch=epoch,
        log_interval=1000,
    )

    # --- optional: also evaluate SWA snapshot mid-training ---
    if swa_enabled:
        swa_model.eval()
        swa_stats = validate_chained(
            swa_model,
            val_loader,
            device=device,
            csv_path=None,
            epoch=epoch,
            log_interval=1000,
        )
        swa_best = swa_stats["best_rate"]

        # --- track best SWA ---
        if swa_best > best_swa_val:
            old_swa = best_swa_ckpt
            best_swa_val = swa_best
            best_swa_ckpt = save_best_ckpt(
                model=swa_model,
                epoch=epoch,
                best_val=best_swa_val,
                cfg_path=cfg_path,
                ckpt_dir=CKPT_DIR,
                prefix=f"{ckpt_prefix}_SWA",
                include_training_state=False
            )
            if old_swa and os.path.exists(old_swa):
                try:
                    os.remove(old_swa)
                    print(f"🗑️ Deleted previous SWA checkpoint: {old_swa}")
                except Exception as e:
                    print(f"⚠️ Failed to delete {old_swa}: {e}")
            print(f"✅ New best SWA model saved: {best_swa_ckpt}")
    else:
        swa_best = float("nan")

    # --- log ---
    train_loss_arr[epoch] = stats["loss"]
    val_rate_arr[epoch] = val_stats["best_rate"]
    print(
        f"E{epoch:02d} | "
        f"Time {(time() - t1) / 60: .3f} mins"
        f"train_loss={stats['loss']:.6f} | "
        f"val_best={val_stats['best_rate']:.6f} | "
        f"swa_best={swa_best:.6f} | "
        f"val_norm_dev={val_stats['max_norm_dev']:.2e}"
    )

    # --- track best ---
    if val_stats["best_rate"] > best_val:
        old = best_ckpt
        best_val = val_stats["best_rate"]
        best_ckpt = save_best_ckpt(
            model=model, epoch=epoch, best_val=best_val, cfg_path=cfg_path,
            ckpt_dir=CKPT_DIR, prefix=ckpt_prefix, include_training_state=include_training_state,
            optimizer=optimizer,
            scheduler=(swa_scheduler if (swa_enabled and epoch >= swa_start) else scheduler),
            scaler=scaler,
            swa_model=(swa_model if (swa_enabled and epoch >= swa_start) else None),
            swa_scheduler=(swa_scheduler if (swa_enabled and epoch >= swa_start) else None),
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
final_raw = validate_chained(model, val_loader, device=device, epoch=MAX_EPOCHS, log_interval=1000)

swa_model.to(device)
swa_model.eval()
final_swa = validate_chained(swa_model, val_loader, device=device, epoch=MAX_EPOCHS, log_interval=1000)

print(
    f"Final compare -> raw: {final_raw['best_rate']:.6f} | "
    f"SWA: {final_swa['best_rate']:.6f}"
)

os.chdir(figs_dir)
plot_train_valid_loss(train_loss_arr, val_rate_arr, filename=f'Train {ckpt_prefix}_{L} layers {B} bands network.png')