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
from utils.EstimationUtils import (
    masked_band_variance_from_dataset,
    precompute_csi_estimates,
    build_estimate_lookup,
)
from utils.ParseUtils import parse_tx_rx_data
from visualization.GraphingAux import plot_train_valid_loss
from time import time

# ====== config ======
# args = parse_args()
# cfg_path = args.config.resolve()
# parser = load_ini_config(cfg_path)
# print(f"Loaded config from CLI: {cfg_path}")
cfg_arr = [r"C:\Users\alter\OneDrive\Desktop\PhD\Decentralized MANET\Config Files\Multicast\ChainedGNN_multicast.ini"]

# cfg_path =   # os.environ.get("MANET_CFG") or
for cfg_path in cfg_arr:
    parser = ConfigParser()
    parser.read_file(open(cfg_path))
    print(f"Loaded default config: {cfg_path}")

    USE_AMP = torch.cuda.is_available()

    # ------- Training Parameters -------
    train_params = parser["Train Parameters"]
    SEED = int(train_params["seed"])
    MODE = train_params["mode"]  # "single" | "multicast" | "multi" | "converge" | "multiunicast"
    B = int(train_params["B"])
    L = int(train_params["L"])
    n = int(train_params["n"])

    # SNR can be an int OR a list in the ini; handle both
    _raw_snr = train_params["SNR"].strip()
    if _raw_snr.startswith("[") and _raw_snr.endswith("]"):
        SNR = [int(x) for x in _raw_snr[1:-1].replace(" ", "").split(",") if x]
    elif "," in _raw_snr:
        SNR = [int(x) for x in _raw_snr.replace(" ", "").split(",") if x]
    else:
        SNR = int(_raw_snr)

    sigma = float(train_params["sigma"])
    DROPOUT = float(train_params["dropout"])
    LR = float(train_params["lr"])
    WEIGHT_DECAY = float(train_params["wd"])
    GRAD_CLIP = float(train_params["grad clip"])
    MAX_EPOCHS = int(train_params["epochs"])
    num_samples = int(train_params["num samples"])
    grad_batch = int(train_params["grad batch"])
    MONO = float(train_params["mono"])
    # Concentration penalty weight (0 = off). Encourages the GNN to route power onto few
    # outgoing edges (a clean multi-band route) instead of spreading it and drowning in
    # self-interference. Optional key; defaults to 0 so existing configs are unchanged.
    SPARSITY = float(train_params.get("sparsity", 0))
    # Late-sparsity ramp (for brittle multi-message "fair"): applying concentration from
    # epoch 0 starves the min-commodity's route BEFORE it forms and collapses training. So
    # keep sparsity OFF for the first SPARSITY_START epochs (routes form on the pure fair
    # objective), then ramp the weight linearly 0 -> SPARSITY over SPARSITY_RAMP epochs and
    # hold. Defaults (0, 0) => sparsity is constant from epoch 0, i.e. unchanged behavior.
    SPARSITY_START = int(train_params.get("sparsity start epoch", 0))
    SPARSITY_RAMP = int(train_params.get("sparsity ramp epochs", 0))
    # Budget-utilization weight (0 = off). Pushes ACTIVE transmitters to saturate their
    # per-node power budget (normalize_power only scales down, so nothing else does).
    # Complements sparsity: sparsity picks WHICH edges, utilization sets them to full power.
    UTILIZATION = float(train_params.get("utilization", 0))
    # Restrict utilization to routes with <= this many hops (0 = no restriction). Full power
    # is near-optimal only on short routes; on 3+ hops the optimal backs power off, so gating
    # to short routes nudges the GNN up where the greedy gap lives without hurting long routes.
    UTIL_MAX_HOPS = int(train_params.get("utilization max hops", 0))
    # Positive init bias on the power head (0 = default). Starts edges near full power so the
    # per-node budget saturates from the start (fights the 1-hop under-powering) — an init
    # shift, not a loss term, so it does not corrupt routing like the utilization penalty did.
    P_HEAD_BIAS_INIT = float(train_params.get("p head bias init", 0))
    # Noise conditioning (0 = off). Appends a global SNR feature (-log10(sigma^2)/5) to every node
    # so the model can adapt its per-band power split to the operating SNR (water-fill across bands
    # at high SNR, concentrate at low SNR). Without it the forward is noise-blind -> one fixed
    # allocation at all SNR. Changes node input dim, so it requires a retrain (default off).
    NOISE_CONDITIONING = int(train_params.get("noise conditioning", 0)) == 1
    # Routing mode: "edge" = per-edge softmax Z (original); "candidate" = route-consistent Z
    # assembled from candidate paths (scores options, path-incidence). Multi family only.
    # Candidate mode guarantees off-route power ~0 by construction (fixes transmit-everywhere).
    Z_MODE = train_params.get("z mode", "edge").strip().lower()
    # Annealed selection temperature for candidate routing: soft (explore paths) -> near-hard
    # (commit to one route). Eval always uses hard argmax (model.eval()).
    ROUTING_TAU_START = float(train_params.get("routing tau start", 1.0))
    ROUTING_TAU_END = float(train_params.get("routing tau end", 0.1))
    # Multi-message (K>1) message reduction: "fair" = eq (10) max-min over messages (the paper);
    # "sum"/"mean" = throughput surrogate (gradient reaches every commodity, trains far better).
    # Ignored for single/multicast (no message axis). Optional key; defaults to the paper's "fair".
    REDUCE = train_params.get("reduce", "sum").strip().lower()
    # Validation/model-selection reduction: default to the REPORTED metric ("fair" = eq 10
    # max-min), even when TRAINING on a surrogate like "pf". So val_best is the positive
    # max-min rate we actually report, and the best checkpoint is chosen by that — not by the
    # surrogate's (negative) value. Override with a `val reduce` key if you report something else.
    VAL_REDUCE = train_params.get("val reduce", "fair").strip().lower()
    # Candidate-head decentralized validation/inference. Training remains centralized.
    DECENTRALIZED_VAL = int(train_params.get("decentralized validation", 1)) == 1
    MAX_ROUTE_HOPS = int(train_params.get("max route hops", L))
    # Cap the soft->hard tau schedule: once tau_linear reaches TAU_CAP, hold it there for the
    # rest of training. At very high tau the soft-min gradient concentrates on a single
    # bottleneck edge (near-hard, high variance) and one sharp step can drop a commodity ->
    # val collapses to 0 (seen in F5 at tau~24). Capping ~16-18 keeps training in the stable
    # zone (tau that high already approximates hard-min well). Default = no cap (1e9).
    TAU_CAP = float(train_params.get("tau cap", 1e9))
    if MODE in {"multi", "converge", "multiunicast"}:
        print(f"REDUCE={REDUCE} | VAL_REDUCE={VAL_REDUCE} | TAU_CAP={TAU_CAP} | DECENTRALIZED_VAL={DECENTRALIZED_VAL} | MAX_ROUTE_HOPS={MAX_ROUTE_HOPS}")

    # Force fp32 (disable AMP) for the numerically fragile regimes:
    #   - reduce=pf: steep 1/(R_k+eps) gradient overflows fp16.
    #   - ANY multi-message framework (multi/converge/multiunicast): at high SNR the SINR
    #     (~desired/sigma^2 ~ 1e10 at 50 dB) OVERFLOWS fp16 -> inf; the backward through it +
    #     grad-accum + clip_grad_norm (one NaN grad makes total_norm NaN -> NaN clip coeff ->
    #     ALL grads NaN) can slip a NaN step past the GradScaler and permanently NaN the weights
    #     (observed: loss=nan from ~ep58, never recovers). fp32 represents 1e10 and stays finite.
    # single/multicast keep AMP (they train fine and are less interference-heavy).
    if REDUCE == "pf" or MODE in {"multi", "converge", "multiunicast"}:
        USE_AMP = False
        print(f"[amp] reduce={REDUCE}/mode={MODE} -> AMP disabled (fp32) for numerical stability")

    est_csi = True if int(train_params["LMMSE estimation"]) == 1 else False
    include_training_state = True if int(train_params["include training state"]) == 1 else False

    # ------- Files Parameters -------
    files_params = parser["Files"]
    channel_path = files_params.get("channel path", None)
    CKPT_DIR = files_params["ckpt dir"]
    figs_dir = files_params["figs dir"]
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

    # tx, rx can be int OR a list in the ini; handle both
    raw_tx = train_params["tx"].strip()
    raw_rx = train_params["rx"].strip()
    tx, rx = parse_tx_rx_data(raw_tx, raw_rx)

    # replicate tx, rx per sampl
    tx_list = [tx] * num_samples  # each sample may have a list of receivers
    rx_list = [rx] * num_samples  # each sample may have a list of receivers

    if MODE == "single":
        K_cfg = 1
    elif MODE in {"multicast", "multi"}:
        K_cfg = len(rx)
    elif MODE == "converge":
        K_cfg = len(tx)
    else:  # "multiunicast"
        if len(tx) != len(rx):
            raise ValueError("tx and rx must have the same length for multiunicast.")
        K_cfg = len(tx)

    # SNR_dB = 10*log10(1/sigma^2)  =>  sigma = 10^(-SNR/20)  (channel variance ~= 1).
    # Matches the benchmark convention in utils/ComparisonUtils.py (sigma = sqrt(var/snr)).
    sigma_vals = np.array([10 ** (-s / 20) for s in SNR])
    base = num_samples // len(sigma_vals)
    remainder = num_samples % len(sigma_vals)
    sigma_list = np.repeat(sigma_vals, base)
    if remainder > 0:
        extra_indices = np.random.choice(len(sigma_vals), size=remainder, replace=True)
        extra_values = sigma_vals[extra_indices]
        sigma_list = np.concatenate([sigma_list, extra_values])
    np.random.shuffle(sigma_list)
    sigma_list.tolist()

    dataset = generate_graph_data(
        n_list=n_list,
        tx_list=tx_list,
        rx_list=rx_list,          # supports int OR list per sample
        sigma_list=sigma_list,
        B=B,
        K=K_cfg,
        problem=MODE,
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
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    # ====== CSI estimation (optional) ======
    if est_csi:
        print("Estimated CSI Model")
        # Train
        prior_var = masked_band_variance_from_dataset(train_set)
        est_train = precompute_csi_estimates(
            train_set, pilots_M=4, pilot_power=1, prior_var=prior_var,
            est_noise_std=None, seed=SEED, device=device,
        )
        train_est_lookup = build_estimate_lookup(est_train)

        # Validation
        prior_var = masked_band_variance_from_dataset(val_set)
        est_val = precompute_csi_estimates(
            val_set, pilots_M=4, pilot_power=1, prior_var=prior_var,
            est_noise_std=None, seed=SEED, device=device,
        )
        val_est_lookup = build_estimate_lookup(est_val)
    else:
        print('True CSI Model')
        train_est_lookup = None
        val_est_lookup = None

    # ====== model / optim / amp ======
    # Choose K for the model:
    # - single: K_model = 1
    # - multicast: K_model = K_cfg (to enable per-receiver role channels; still one shared message)
    # - multi, converge, multiunicast: K_model = K_cfg (distinct messages, produces [B,K,n,n] + Z)
    if MODE == "single":
        K_model = 1
    elif MODE in {"multicast", "multi", "converge", "multiunicast"}:
        K_model = K_cfg
    else:
        raise ValueError("MODE must be 'multicast', 'multi', 'converge', or 'multiunicast'.")

    model = ChainedGNN(
        num_layers=L,
        B=B,
        K=K_model,
        problem=MODE,
        dropout=DROPOUT,
        use_jk=True,
        jk_mode="concat",
        p_head_bias_init=P_HEAD_BIAS_INIT,
        noise_conditioning=NOISE_CONDITIONING,
        z_mode=Z_MODE,
    ).to(device)

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
    scheduler = LambdaLR(optimizer, lr_lambda=cosine_warm_restart_lambda(
        lr_warmup_epochs, total_epochs, eta_min, base_lr
    ))

    scaler = GradScaler(enabled=USE_AMP)

    # ====== checkpoints ======
    _reduce_note = f" | reduce={REDUCE}" if MODE in {"multi", "converge", "multiunicast"} else ""
    print(f">>> {num_samples} samples | seed={SEED} | AMP={USE_AMP} | grad_clip={GRAD_CLIP} | mode={MODE} | K_model={K_model}{_reduce_note}")

    best_val = -float("inf")
    best_ckpt = None

    # Per-sample routing structures (paths/subgraphs) are topology-only and identical
    # every epoch, so cache them once (populated on epoch 0) and reuse.
    train_struct_cache = {}
    val_struct_cache = {}

    train_loss_arr = np.zeros(MAX_EPOCHS)
    val_rate_arr = np.zeros(MAX_EPOCHS)
    t0 = time()

    # ====== training loop ======
    for epoch in range(MAX_EPOCHS):
        epoch_tau = min(tau_linear(epoch, MAX_EPOCHS), TAU_CAP)  # capped soft-min temperature
        # Candidate-routing selection temperature: anneal soft -> hard over training so the
        # scorer explores paths early and commits to one route late. (Eval uses hard argmax.)
        if Z_MODE == "candidate":
            _rf = epoch / max(MAX_EPOCHS - 1, 1)
            model.routing_tau = ROUTING_TAU_START * (1 - _rf) + ROUTING_TAU_END * _rf
        # Late-sparsity ramp: 0 until SPARSITY_START, then linear 0->SPARSITY over
        # SPARSITY_RAMP epochs, then hold at SPARSITY. (0,0) => constant SPARSITY (unchanged).
        if epoch < SPARSITY_START:
            sparsity_w = 0.0
        elif SPARSITY_RAMP > 0:
            sparsity_w = SPARSITY * min(1.0, (epoch - SPARSITY_START) / SPARSITY_RAMP)
        else:
            sparsity_w = SPARSITY
        print(f">>> Epoch {epoch} | tau_min={epoch_tau:.3f} | sparsity={sparsity_w:.4f} | mode={MODE}")
        t1 = time()

        # --- train ---
        stats = train_chained(
            model,
            train_loader,
            optimizer,
            epoch,
            batch_size=1,
            mode=MODE,                 # unified ("single" | "multicast" | "multi")
            mono_weight=MONO,
            sparsity_weight=sparsity_w,
            utilization_weight=UTILIZATION,
            utilization_max_hops=UTIL_MAX_HOPS,
            use_amp=USE_AMP,
            grad_clip=GRAD_CLIP,
            grad_accum_steps=grad_batch,
            tau=epoch_tau,
            reduce=REDUCE,
            est_dataset=train_est_lookup,
            struct_cache=train_struct_cache,
            decentralized_inference=False,
            max_route_hops=None,
        )

        # --- step LR ---
        scheduler.step()

        # --- validate ---
        val_stats = validate_chained(
            model,
            val_loader,
            batch_size=1,
            mode=MODE,                 # unified
            device=device,
            est_dataset=val_est_lookup,
            tau=0.0,                   # hard min/max in validation
            reduce=VAL_REDUCE,         # select on the REPORTED metric (fair), not the pf surrogate
            struct_cache=val_struct_cache,
            verbose=True,
            decentralized_inference=(DECENTRALIZED_VAL and Z_MODE == "candidate" and MODE in {"multi", "converge", "multiunicast"}),
            max_route_hops=MAX_ROUTE_HOPS,
        )

        # --- log ---
        train_loss_arr[epoch] = stats["loss"]
        val_rate_arr[epoch] = val_stats["best_rate"]
        print(
            f"E{epoch:02d} | "
            f"Time {(time() - t1) / 60: .3f} mins | "
            f"train_loss={stats['loss']:.6f} | "
            f"val_best={val_stats['best_rate']:.6f}"
        )

        # --- track best ---
        if val_stats["best_rate"] > best_val:
            old = best_ckpt
            best_val = val_stats["best_rate"]
            best_ckpt = save_best_ckpt(
                model=model,
                epoch=epoch,
                best_val=best_val,
                cfg_path=cfg_path,
                ckpt_dir=CKPT_DIR,
                prefix=ckpt_prefix,
                include_training_state=include_training_state,
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
    print(f"Training time = {(time() - t0) / 60: .3f} mins")

    os.chdir(figs_dir)
    plot_train_valid_loss(
        train_loss_arr,
        val_rate_arr,
        filename=f"Train {ckpt_prefix}_{L} layers {B} bands network ({MODE}).png"
    )
    print('========================================================================')
    print(f'Done training mode: {MODE} | {ckpt_prefix}_{L} layers {B} bands network')
    print('========================================================================')