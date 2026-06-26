# Expected config sections:
#   [Train Parameters]
#       seed
#       mode
#       B
#       n
#       SNR
#       dropout
#       lr
#       wd
#       epochs
#       num samples
#       tx
#       rx
#       K                 optional; inferred if missing
#       val_ratio         optional, default 0.2
#       hidden_dim        optional, default 512
#       num_layers        optional, default 4
#       batch_size        optional, default 1
#
#   [Files]
#       channel path
#       ckpt dir
#       figs dir
#       prefix
#
# The channel .mat file is assumed to contain H_all with shape [N, B, n, n].
# Non-existing edges should already be zeroed in H_all. The adjacency matrix is
# reconstructed as adj[i,j] = 1 if any band has nonzero channel i->j.

import os
import random
import pickle
import numpy as np
import torch
from pathlib import Path
from time import time
from configparser import ConfigParser
from types import SimpleNamespace

try:
    import mat73
except ImportError:
    mat73 = None

try:
    import scipy.io
except ImportError:
    scipy = None

from utils.ParseUtils import parse_tx_rx_data
from utils.TensorUtils import normalize_power
from utils.PathUtils import find_all_paths, paths_to_tensor
from utils.MetricUtils import calc_sum_rate
from Multicast.SubGraphs import find_multicast_subgraphs
from Multicast.Objective import objective_multicast
from Multicommodity.Objective import objective_multicommodity

from MANET_FFN.MANET_FFN_Train_Validate import train_ffn


# ============================================================
# Config parsing
# ============================================================

def _parse_snr(raw_snr):
    """
    Parse SNR field from the ini file.

    Parameters
    ----------
    raw_snr : str
        Either a scalar string, e.g. "10", or a list-like string,
        e.g. "[0, 5, 10]" or "0,5,10".

    Returns
    -------
    int or list[int]
        Parsed SNR value(s).
    """
    raw_snr = raw_snr.strip()

    if raw_snr.startswith("[") and raw_snr.endswith("]"):
        return [int(x) for x in raw_snr[1:-1].replace(" ", "").split(",") if x]

    if "," in raw_snr:
        return [int(x) for x in raw_snr.replace(" ", "").split(",") if x]

    return int(raw_snr)


def parse_args(cfg_path):
    """
    Parse the same ini-style configuration used by the MANET-GNN script.

    Parameters
    ----------
    cfg_path : str or pathlib.Path
        Path to the ini configuration file.

    Returns
    -------
    SimpleNamespace
        Configuration object with attributes:
            seed, mode, B, n, SNR, dropout, lr, weight_decay,
            epochs, num_samples, batch_size, val_ratio,
            hidden_dim, num_layers, tx, rx, K_cfg,
            channel_path, ckpt_dir, figs_dir, prefix, device.
    """
    cfg_path = Path(cfg_path)

    parser = ConfigParser()
    parser.read_file(open(cfg_path))

    train_params = parser["Train Parameters"]
    files_params = parser["Files"]

    seed = int(train_params["seed"])
    mode = train_params["mode"].lower()
    B = int(train_params["B"])
    n = int(train_params["n"])
    SNR = _parse_snr(train_params["SNR"])

    dropout = float(train_params.get("dropout", 0.1))
    lr = float(train_params["lr"])
    weight_decay = float(train_params.get("wd", train_params.get("weight_decay", 0.0)))
    epochs = int(train_params["epochs"])
    num_samples = int(train_params["num samples"])

    batch_size = int(train_params.get("batch_size", 1))
    val_ratio = float(train_params.get("val_ratio", 0.2))

    # FFN-specific parameters. These can be added to the same ini file.
    hidden_dim = int(train_params.get("hidden_dim", 512))
    num_layers = int(train_params.get("num_layers", 4))

    raw_tx = train_params["tx"].strip()
    raw_rx = train_params["rx"].strip()
    tx, rx = parse_tx_rx_data(raw_tx, raw_rx)

    # Infer K exactly as in the MANET-GNN script.
    if mode == "single":
        K_cfg = 1
    elif mode in {"multicast", "multi"}:
        K_cfg = len(rx) if isinstance(rx, (list, tuple)) else 1
    elif mode == "converge":
        K_cfg = len(tx) if isinstance(tx, (list, tuple)) else 1
    elif mode == "multiunicast":
        if len(tx) != len(rx):
            raise ValueError("tx and rx must have the same length for multiunicast.")
        K_cfg = len(tx)
    else:
        raise ValueError("mode must be one of: single, multicast, multi, converge, multiunicast.")

    # Optional override from config.
    if "K" in train_params:
        K_cfg = int(train_params["K"])

    channel_path = files_params.get("channel_path", None)
    if channel_path is not None and str(channel_path).strip().lower() in {"", "none", "null"}:
        channel_path = None

    ckpt_dir = Path(files_params["ckpt dir"])
    figs_dir = Path(files_params["figs dir"])
    prefix = files_params["prefix"]

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return SimpleNamespace(
        cfg_path=cfg_path,
        seed=seed,
        mode=mode,
        B=B,
        n=n,
        SNR=SNR,
        dropout=dropout,
        lr=lr,
        weight_decay=weight_decay,
        epochs=epochs,
        num_samples=num_samples,
        batch_size=batch_size,
        val_ratio=val_ratio,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        tx=tx,
        rx=rx,
        K_cfg=K_cfg,
        channel_path=channel_path,
        ckpt_dir=ckpt_dir,
        figs_dir=figs_dir,
        prefix=prefix,
        device=device,
    )


# ============================================================
# Dataset loading / generation
# ============================================================

def build_sigma_list(SNR, num_samples, seed=None):
    """
    Build per-sample noise standard deviations from the SNR setting.

    This follows the same pattern as the MANET-GNN script:
        sigma = 10^(-SNR/10)

    Parameters
    ----------
    SNR : int or list[int]
        SNR value(s) from the config.

    num_samples : int
        Number of graph/channel samples.

    seed : int, optional
        Random seed used only for assigning remainder samples when SNR is a list.

    Returns
    -------
    list[float]
        Noise values, one per sample.
    """
    if isinstance(SNR, int):
        sigma_vals = np.array([10 ** (-SNR / 10)])
    else:
        sigma_vals = np.array([10 ** (-s / 10) for s in SNR])

    base = num_samples // len(sigma_vals)
    remainder = num_samples % len(sigma_vals)

    sigma_list = np.repeat(sigma_vals, base)

    if remainder > 0:
        rng = np.random.default_rng(seed)
        extra_indices = rng.choice(len(sigma_vals), size=remainder, replace=True)
        sigma_list = np.concatenate([sigma_list, sigma_vals[extra_indices]])

    rng = np.random.default_rng(seed)
    rng.shuffle(sigma_list)

    return sigma_list.tolist()


def _load_mat_file(channel_path):
    """
    Load a MATLAB file using mat73 when possible, with scipy.io fallback.

    Parameters
    ----------
    channel_path : str
        Path to .mat file.

    Returns
    -------
    dict
        MATLAB variable dictionary.
    """
    if mat73 is not None:
        try:
            return mat73.loadmat(channel_path)
        except Exception:
            pass

    if scipy is None:
        raise ImportError(
            "Could not load .mat file. Install mat73 for v7.3 files or scipy for older .mat files."
        )

    return scipy.io.loadmat(channel_path)


def load_channel_tensor(channel_path, mat_key="H_all"):
    """
    Load channel realizations from .mat/.pt/.pth/.pkl/.pickle.

    Parameters
    ----------
    channel_path : str
        Path to channel file.

    mat_key : str, optional
        Key containing the channel tensor. Default is "H_all".

    Returns
    -------
    torch.Tensor
        Complex tensor with shape [num_samples, B, n, n].
    """
    if channel_path is None:
        raise ValueError("channel_path is None. This FFN script expects precomputed channel realizations.")

    if not os.path.exists(channel_path):
        raise FileNotFoundError(f"Channel file does not exist: {channel_path}")

    ext = os.path.splitext(channel_path)[1].lower()

    if ext == ".mat":
        data = _load_mat_file(channel_path)

        if mat_key not in data:
            raise KeyError(
                f"Could not find key '{mat_key}' in {channel_path}. "
                f"Available keys: {list(data.keys())}"
            )

        raw_channels = data[mat_key]

    elif ext in [".pt", ".pth"]:
        data = torch.load(channel_path, map_location="cpu")

        if isinstance(data, dict) and mat_key in data:
            raw_channels = data[mat_key]
        elif isinstance(data, dict) and "links_list" in data:
            raw_channels = data["links_list"]
        else:
            raw_channels = data

    elif ext in [".pkl", ".pickle"]:
        with open(channel_path, "rb") as f:
            data = pickle.load(f)

        if isinstance(data, dict) and mat_key in data:
            raw_channels = data[mat_key]
        elif isinstance(data, dict) and "links_list" in data:
            raw_channels = data["links_list"]
        else:
            raw_channels = data

    else:
        raise ValueError("Unsupported channel file extension. Use .mat, .pt, .pth, .pkl, or .pickle.")

    # Convert list of tensors/arrays or dense numpy array to one tensor.
    if isinstance(raw_channels, list):
        raw_channels = torch.stack([torch.as_tensor(x, dtype=torch.cfloat) for x in raw_channels], dim=0)
    else:
        raw_channels = torch.as_tensor(raw_channels, dtype=torch.cfloat)

    if raw_channels.ndim != 4:
        raise ValueError(
            f"Expected channel tensor with shape [num_samples, B, n, n], "
            f"got {tuple(raw_channels.shape)}."
        )

    return raw_channels


def generate_ffn_lists_from_channels(
    channel_path,
    tx,
    rx,
    SNR,
    B,
    num_samples,
    problem,
    seed,
    device="cpu",
    mat_key="H_all",
):
    """
    Generate FFN input lists from precomputed channel realizations.

    The .mat file stores only H_all. The remaining metadata is generated here:
        - adj_list is inferred from nonzero channels.
        - tx_list and rx_list are replicated from the config.
        - sigma_list is generated from the SNR field.

    Parameters
    ----------
    channel_path : str
        Path to .mat/.pt/.pth/.pkl/.pickle channel file.

    tx :
        Transmitter definition parsed from config.

    rx :
        Receiver definition parsed from config.

    SNR : int or list[int]
        SNR value(s).

    B : int
        Number of frequency bands.

    num_samples : int
        Number of samples requested by config.

    problem : str
        Communication problem type.

    seed : int
        Random seed.

    device : str or torch.device
        Device for generated tensors. Recommended: "cpu" for datasets.

    mat_key : str
        Channel tensor key inside .mat file.

    Returns
    -------
    tuple
        adj_list, links_list, tx_list, rx_list, sigma_list
    """
    raw_channels = load_channel_tensor(channel_path, mat_key=mat_key)

    if raw_channels.shape[1] != B:
        raise ValueError(f"B mismatch: config B={B}, but channels have B={raw_channels.shape[1]}.")

    available_samples = raw_channels.shape[0]
    if num_samples > available_samples:
        raise ValueError(
            f"Requested {num_samples} samples, but channel file contains only {available_samples}."
        )

    raw_channels = raw_channels[:num_samples]

    adj_list = []
    links_list = []

    for i in range(num_samples):
        links = torch.as_tensor(raw_channels[i], dtype=torch.cfloat, device=device)

        # Recover topology from masked CSI.
        # This assumes non-edges were zeroed when the channel file was created.
        adj = (links.abs().sum(dim=0) > 0).float()
        adj.fill_diagonal_(0)

        # Safety: remove self-links from both adjacency and CSI.
        eye = torch.eye(adj.shape[0], dtype=torch.bool, device=device)
        links[:, eye] = 0.0

        adj_list.append(adj)
        links_list.append(links)

    tx_list = [tx] * num_samples
    rx_list = [rx] * num_samples
    sigma_list = build_sigma_list(SNR, num_samples, seed=seed)

    print(
        f"Loaded FFN channel data: samples={num_samples}, "
        f"B={B}, n={raw_channels.shape[-1]}, problem={problem}"
    )

    return adj_list, links_list, tx_list, rx_list, sigma_list


# ============================================================
# Objective wrappers
# ============================================================

def _unwrap_collated_scalar_or_list(x):
    """
    Normalize tx/rx values after PyTorch default collation.

    Parameters
    ----------
    x :
        Python int, tensor, list, or nested list produced by DataLoader.

    Returns
    -------
    int or list[int]
        Clean source/destination representation.
    """
    if isinstance(x, torch.Tensor):
        return int(x.item()) if x.numel() == 1 else [int(v) for v in x.view(-1).tolist()]

    if isinstance(x, tuple):
        x = list(x)

    if isinstance(x, list):
        if len(x) == 1:
            return _unwrap_collated_scalar_or_list(x[0])
        return [_unwrap_collated_scalar_or_list(v) for v in x]

    return int(x)


def objective_ffn_single(h, p, sigma, adj, tx, rx):
    """
    Single-unicast FFN objective wrapper.

    Parameters
    ----------
    h : torch.Tensor
        Complex CSI tensor [B, n, n].

    p : torch.Tensor
        Feasible power/amplitude tensor [B, n, n].

    sigma : torch.Tensor
        Noise standard deviation.

    adj : torch.Tensor
        Adjacency matrix [n, n].

    tx : int
        Source node.

    rx : int
        Destination node.

    Returns
    -------
    torch.Tensor
        Scalar achieved rate under the interference-aware calc_sum_rate.
    """
    tx = _unwrap_collated_scalar_or_list(tx)
    rx = _unwrap_collated_scalar_or_list(rx)

    paths = find_all_paths(adj.cpu(), int(tx), int(rx))
    if not paths:
        return 0.0 * p.sum()

    paths_tensor = paths_to_tensor(paths, h.device)

    return calc_sum_rate(
        h_arr=h,
        p_arr=p,
        sigma=sigma,
        paths_tensor=paths_tensor,
        B=h.shape[0],
        tau=0.0,
        eps=1e-12,
        per_band=False,
        ignore_zero_edges=False,
    )


def objective_ffn_multicast(h, p, sigma, adj, tx, rx):
    """
    Multicast FFN objective wrapper.

    Parameters
    ----------
    h : torch.Tensor
        Complex CSI tensor [B, n, n].

    p : torch.Tensor
        Feasible power/amplitude tensor [B, n, n].

    sigma : torch.Tensor
        Noise standard deviation.

    adj : torch.Tensor
        Adjacency matrix [n, n].

    tx : int
        Source node.

    rx : list[int]
        Multicast destination nodes.

    Returns
    -------
    torch.Tensor
        Scalar multicast objective value.
    """
    tx = _unwrap_collated_scalar_or_list(tx)
    rx = _unwrap_collated_scalar_or_list(rx)
    rx_list = rx if isinstance(rx, list) else [rx]

    subgraphs = find_multicast_subgraphs(adj.cpu(), int(tx), rx_list)
    if subgraphs is None or len(subgraphs) == 0:
        return 0.0 * p.sum()

    subgraphs_per_band = [subgraphs for _ in range(h.shape[0])]

    return objective_multicast(
        h=h,
        p=p,
        sigma=sigma,
        adj=adj,
        subgraphs_per_band=subgraphs_per_band,
        tau_min=0.0,
        tau_max=0.0,
        per_band=False,
        eps=1e-12,
        ignore_zero_edges=False,
    )


def objective_ffn_multicommodity(h, p, sigma, adj, tx, rx, problem):
    """
    Multi-message FFN objective wrapper for:
        - multi
        - converge
        - multiunicast

    Parameters
    ----------
    h : torch.Tensor
        Complex CSI tensor [B, n, n].

    p : torch.Tensor
        Feasible power/amplitude tensor [B, K, n, n].

    sigma : torch.Tensor
        Noise standard deviation.

    adj : torch.Tensor
        Adjacency matrix [n, n].

    tx :
        Source definition according to the selected problem.

    rx :
        Destination definition according to the selected problem.

    problem : str
        One of {"multi", "converge", "multiunicast"}.

    Returns
    -------
    torch.Tensor
        Scalar multi-commodity objective value.
    """
    tx = _unwrap_collated_scalar_or_list(tx)
    rx = _unwrap_collated_scalar_or_list(rx)

    if problem == "multi":
        rx_list = rx if isinstance(rx, list) else [rx]
        tx_list = [int(tx)] * len(rx_list)

    elif problem == "converge":
        tx_list = tx if isinstance(tx, list) else [tx]
        rx_scalar = rx[0] if isinstance(rx, list) else rx
        rx_list = [int(rx_scalar)] * len(tx_list)

    elif problem == "multiunicast":
        tx_list = tx if isinstance(tx, list) else [tx]
        rx_list = rx if isinstance(rx, list) else [rx]
        if len(tx_list) != len(rx_list):
            raise ValueError("For multiunicast, tx and rx must have the same length.")

    else:
        raise ValueError(f"Unknown multi-message problem: {problem}")

    paths_k = []
    has_path = False
    for tx_k, rx_k in zip(tx_list, rx_list):
        paths = find_all_paths(adj.cpu(), int(tx_k), int(rx_k))
        if paths:
            has_path = True
            paths_k.append(paths_to_tensor(paths, h.device))
        else:
            paths_k.append(torch.empty((0, 0), device=h.device, dtype=torch.long))

    if not has_path:
        return 0.0 * p.sum()

    B, K, n, _ = p.shape

    # FFN predicts only P. For the multi-commodity objective we use the
    # adjacency as a fixed routing-activation mask Z, as in the baseline code.
    z = adj.bool().float().unsqueeze(0).unsqueeze(0).expand(B, K, n, n).to(h.device)

    return objective_multicommodity(
        h=h,
        p=p,
        z=z,
        sigma=sigma,
        adj=adj,
        paths_k=paths_k,
        tau_min=0.0,
        tau_max=0.0,
        reduce="mean",
        per_band=False,
        outage_as_neg_inf=False,
        ignore_zero_edges=False,
    )


def select_ffn_objective(problem):
    """
    Select an FFN-compatible objective wrapper.

    Parameters
    ----------
    problem : str
        Communication framework.

    Returns
    -------
    callable
        Function with signature:
            objective_fn(h, p, sigma, adj, tx, rx) -> scalar tensor
    """
    problem = problem.lower()

    if problem == "single":
        return objective_ffn_single

    if problem == "multicast":
        return objective_ffn_multicast

    if problem in {"multi", "converge", "multiunicast"}:
        return lambda h, p, sigma, adj, tx, rx: objective_ffn_multicommodity(
            h=h,
            p=p,
            sigma=sigma,
            adj=adj,
            tx=tx,
            rx=rx,
            problem=problem,
        )

    raise ValueError(f"Unknown problem: {problem}")


# ============================================================
# Main
# ============================================================

def main(cfg_path):
    """
    Train, validate, and save the FFN model.

    Outputs
    -------
    Files under save_dir:
        - ffn_checkpoint.pt
        - ffn_history.pt
        - ffn_loss_curve.png
        - ffn_rate_curve.png
        - ffn_run_config.pt
    """
    # Set this path directly for PyCharm debugging, same style as your MANET-GNN script.


    cfg = parse_args(cfg_path)
    print(f"Loaded config: {cfg.cfg_path}")

    USE_AMP = torch.cuda.is_available()
    print(f"AMP available: {USE_AMP} -- not used in FFN helper unless you add AMP manually.")

    # ====== seeding ======
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    # ====== dataset ======
    adj_list, links_list, tx_list, rx_list, sigma_list = generate_ffn_lists_from_channels(
        channel_path=cfg.channel_path,
        tx=cfg.tx,
        rx=cfg.rx,
        SNR=cfg.SNR,
        B=cfg.B,
        num_samples=cfg.num_samples,
        problem=cfg.mode,
        seed=cfg.seed,
        device="cpu",
        mat_key="H_all",
    )

    # ====== objective ======
    objective_fn = select_ffn_objective(cfg.mode)

    # ====== train ======
    t0 = time()

    save_dir = cfg.ckpt_dir / f"{cfg.prefix}_ffn"

    print(
        f">>> {cfg.num_samples} samples | seed={cfg.seed} | "
        f"mode={cfg.mode} | B={cfg.B} | K={cfg.K_cfg} | "
        f"hidden_dim={cfg.hidden_dim} | layers={cfg.num_layers}"
    )

    model, history = train_ffn(
        adj_list=adj_list,
        links_list=links_list,
        tx_list=tx_list,
        rx_list=rx_list,
        sigma_list=sigma_list,
        B=cfg.B,
        objective_fn=objective_fn,
        normalize_power=normalize_power,
        problem=cfg.mode,
        K=cfg.K_cfg,
        val_ratio=cfg.val_ratio,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        num_epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
        device=cfg.device,
        save_dir=str(save_dir),
    )

    torch.save(vars(cfg), save_dir / "ffn_run_config.pt")

    print(f"Training time = {(time() - t0) / 60:.3f} mins")
    print(f"Best validation rate = {max(history['val_rate']):.6f}")
    print(f"Saved FFN results to: {save_dir}")


if __name__ == "__main__":
    cfg_path = r"C:\Users\alter\OneDrive\Desktop\PhD\Decentralized MANET\Config Files\FFN\Multiunicast\FFN_multiunicast.ini"
    main(cfg_path)
