"""
Off-route / interference diagnostic — GENERAL (all frameworks).

Goal: explain a GNN-vs-optimizer/greedy gap by measuring how concentrated each
method's power allocation is and how interference-limited the route(s) it uses are.

Supported `mode` (read from the config's [Train Parameters] `mode` key):
    single        1 Tx -> 1 Rx                      P: [B,n,n]
    multicast     1 Tx -> K Rx, shared message      P: [B,n,n]   (subgraph routing)
    multi         1 Tx -> K Rx, one msg per Rx      P: [B,K,n,n]
    converge      K Tx -> 1 Rx, one msg per Tx      P: [B,K,n,n]
    multiunicast  K (Tx,Rx) pairs                   P: [B,K,n,n]

For each method (GNN / centralized optimizer / greedy) it reports, averaged over a
subset of the benchmark dataset, at a few SNRs:

  off_route_frac : fraction of each band's power that sits OFF that band's best path
                   (0 = perfectly concentrated on the route, ->1 = smeared everywhere).
                   For multi-message: averaged over commodities+bands, power-weighted.
  INR_on_route   : mean interference-to-noise ratio (I / sigma^2) on the route edges
                   (>>1 at high SNR ==> interference-limited, i.e. spreading hurts).
  active_edges   : mean number of graph edges carrying power in ANY band/commodity.
  total_power    : mean sum p^2 (per-node L2<=1, so <= #tx nodes).
  rate           : mean framework objective:
                     single/multicast -> sum over bands (eq 9)
                     multi family     -> min over commodities of (sum over bands)  (eq 10)

Second table:
  single   -> rate split by Tx->Rx shortest-hop distance (relay-forwarding / depth).
  multi    -> per-commodity rate (which message is starved by the max-min).
  multicast-> skipped (single shared message, no per-commodity split).

Run:  PYTHONPATH=<repo_root> python scripts/diagnose_offroute.py
It only prints; it does not write any files.
"""
import os
import numpy as np
import torch
from pathlib import Path
from configparser import ConfigParser

from models.models import ChainedGNN
from models.GraphNetAux import _compute_rates_per_layer
from utils.DataUtils import generate_graph_data, mean_var_over_dataset
from utils.ParseUtils import parse_tx_rx_data
from utils.PathUtils import find_all_paths, paths_to_tensor
from Multicast.SubGraphs import find_multicast_subgraphs
from utils.CentralizedUtils import (
    evaluate_centralized_adam,
    compute_centralized_greedy_power_rate,
)

# ------------------------- knobs -------------------------
CFG = os.path.join(os.environ.get("MANET_DATA_ROOT", r"C:\Users\alter\OneDrive\Desktop\PhD\Decentralized MANET"), r"Config Files\Multiunicast\R2\comp_multiunicast_FullCSI_R2.ini")
SUBSET = 25            # samples to analyze (raise for tighter stats; optimizer is the slow part)
SNR_LIST = [20, 50]    # dB
OPT_ITERS = 500       # match the benchmark
PWR_THR = 1e-8         # "active edge" power^2 threshold
# ---------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

p = ConfigParser(); p.read_file(open(CFG))
tp, fp = p["Train Parameters"], p["Files"]
B, n = int(tp["B"]), int(tp["n"])
L, DROPOUT = int(tp["l"]), float(tp["dropout"])
problem = tp.get("mode", "single").strip().lower()
tx, rx = parse_tx_rx_data(tp["tx"].strip(), tp["rx"].strip())
channel_path = fp.get("channel path", None)
model_path = fp["model path"]

MULTI_FAMILY = {"multi", "converge", "multiunicast"}


def as_list(x):
    if isinstance(x, torch.Tensor):
        return x.view(-1).tolist()
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def commodity_pairs(tx_, rx_):
    """Return the ordered list of (tx_k, rx_k) commodity endpoints for the multi family.
    Order MUST match the GNN/objective K-axis (ComparisonUtils ordering)."""
    if problem == "multi":                     # 1 Tx, K receivers
        rxl = as_list(rx_)
        txv = tx_[0] if isinstance(tx_, (list, tuple)) else tx_
        return [(int(txv), int(r)) for r in rxl]
    if problem == "converge":                  # K transmitters, 1 Rx
        txl = as_list(tx_)
        rxv = rx_[0] if isinstance(rx_, (list, tuple)) else rx_
        return [(int(t), int(rxv)) for t in txl]
    # multiunicast: K (Tx,Rx) pairs
    txl, rxl = as_list(tx_), as_list(rx_)
    return [(int(t), int(r)) for t, r in zip(txl, rxl)]


# Number of commodities/messages the model must be built with.
if problem in MULTI_FAMILY:
    K = len(commodity_pairs(tx, rx))
else:
    K = 1

# ---- dataset (same generator as the benchmark) ----
dataset = generate_graph_data(
    n_list=[n] * SUBSET, tx_list=[tx] * SUBSET, rx_list=[rx] * SUBSET,
    sigma_list=[1.0] * SUBSET, B=B, K=K, problem=problem,
    seed=int(tp["seed"]), channel_path=channel_path, device="cpu",
)
mean_var = mean_var_over_dataset(dataset)

# ---- GNN ----
model = ChainedGNN(num_layers=L, B=B, K=K, problem=problem,
                   dropout=DROPOUT, use_jk=True, jk_mode="concat",
                   z_mode=tp.get("z mode", "edge").strip().lower()).to(device).eval()
ckpt = torch.load(model_path, map_location=device, weights_only=False)
sd = {(k.replace("head.", "p_head.") if k.startswith("head.") else k): v
      for k, v in ckpt["model_state_dict"].items()}
_load = model.load_state_dict(sd, strict=False)
_missing = [k for k in _load.missing_keys if "num_batches_tracked" not in k]
_unexpected = list(_load.unexpected_keys)
print(f"[ckpt] problem={problem}  K={K}  L={L}  file={Path(model_path).name}")
print(f"[ckpt] val_best in ckpt = {ckpt.get('best_val', ckpt.get('val', 'n/a'))}")
print(f"[ckpt] missing keys ({len(_missing)}): {_missing[:8]}{' ...' if len(_missing) > 8 else ''}")
print(f"[ckpt] unexpected  ({len(_unexpected)}): {_unexpected[:8]}{' ...' if len(_unexpected) > 8 else ''}")
if _missing:
    print("[ckpt] *** WARNING: missing keys run on RANDOM init -> the model is partly UNTRAINED. "
          "Check that config 'l'/'mode'/K match the trained model and 'model path' is the right file. ***")
print()


# ---------------------------------------------------------------------------
# Interference models (mirror the scoring objectives exactly)
# ---------------------------------------------------------------------------
def per_edge_rate_and_interf(P, links, sigma, eps=1e-12):
    """Single/multicast: P [B,n,n] -> per-edge rate R and interference I ([B,n,n]).
    Mirrors calc_sum_rate's interference model."""
    ch = links.abs() ** 2
    p2 = (P.real ** 2)
    desired = ch * p2
    txp = p2.sum(dim=2)                                  # [B,n]  each node's total tx power
    nbr = (ch > eps).float()
    rx_tot = torch.einsum("bkj,bk->bj", ch * nbr, txp)   # [B,n]
    I = (rx_tot[:, None, :] - desired).clamp_min(0.0)     # [B,n,n]
    sinr = desired / (sigma ** 2 + I + eps)
    R = torch.log2(1.0 + sinr)
    return R.real, I


def per_edge_rate_and_interf_multi(P, links, adj, sigma, eps=1e-12):
    """Multi family: P [B,K,n,n] -> R,I [B,K,n,n]. Mirrors edge_rates_multicommodity
    (same-band, neighbor-only, interference summed over ALL commodities => captures
    inter-commodity interference). z is taken =1 (p^2 already zeroes unpowered edges)."""
    adj_mask = (adj != 0).to(device=P.device, dtype=torch.float32)     # [n,n]
    ch = (links.abs() ** 2).to(torch.float32) * adj_mask               # [B,n,n]
    p2 = (P.real ** 2).to(torch.float32) * adj_mask[None, None]        # [B,K,n,n]
    sig = ch[:, None, :, :] * p2                                       # [B,K,n,n]
    tx_pwr = p2.sum(dim=(1, 3))                                        # [B,n] over commodities+dests
    rx_tot = torch.einsum("bij,bi->bj", ch, tx_pwr)                    # [B,n]
    I = (rx_tot[:, None, None, :] - sig).clamp_min(0.0)                # [B,K,n,n]
    sinr = sig / (sigma ** 2 + I + eps)
    R = torch.log2(1.0 + sinr)
    return R.real, I


# ---------------------------------------------------------------------------
# Route metrics
# ---------------------------------------------------------------------------
def _generic_pwr_stats(P):
    p2 = (P.real ** 2)
    total_power = float(p2.sum().item())
    lead = tuple(range(P.dim() - 2))                     # collapse all but the last two axes
    active_edges = int((p2 > PWR_THR).any(dim=lead).sum().item())
    return total_power, active_edges


def route_metrics_single(P, links, sigma, paths):
    R, I = per_edge_rate_and_interf(P, links, sigma)
    p2 = (P.real ** 2)
    total_power, active_edges = _generic_pwr_stats(P)

    off_fracs, inrs, wts, rate = [], [], [], 0.0
    for b in range(B):
        best_val, best_edges = -1.0, None
        for path in paths:
            edges = list(zip(path[:-1], path[1:]))
            mn = min(float(R[b, u, v]) for u, v in edges)
            if mn > best_val:
                best_val, best_edges = mn, edges
        rate += best_val if best_edges is not None else 0.0
        if best_edges is None:
            continue
        tot_b = float(p2[b].sum().item())
        if tot_b < 1e-12:
            continue
        on = sum(float(p2[b, u, v]) for u, v in best_edges)
        off_fracs.append(1.0 - on / tot_b)
        inrs.append(np.mean([float(I[b, u, v]) / (sigma ** 2) for u, v in best_edges]))
        wts.append(tot_b)

    w = np.asarray(wts) if wts else np.asarray([1.0])
    off = float(np.average(off_fracs, weights=w)) if off_fracs else 0.0
    inr = float(np.average(inrs, weights=w)) if inrs else 0.0
    return dict(off=off, inr=inr, active_edges=active_edges,
                total_power=total_power, rate=rate)


def route_metrics_multi(P, links, adj, sigma, paths_k):
    """P [B,K,n,n]. Per-commodity per-band best path; rate = min_k sum_b (eq 10)."""
    R, I = per_edge_rate_and_interf_multi(P, links, adj, sigma)
    p2 = (P.real ** 2)
    total_power, active_edges = _generic_pwr_stats(P)

    off_fracs, inrs, wts = [], [], []
    per_k_rate = []
    for k, paths in enumerate(paths_k):
        rk = 0.0
        if paths:
            for b in range(B):
                best_val, best_edges = -1.0, None
                for path in paths:
                    edges = list(zip(path[:-1], path[1:]))
                    mn = min(float(R[b, k, u, v]) for u, v in edges)
                    if mn > best_val:
                        best_val, best_edges = mn, edges
                rk += best_val if best_edges is not None else 0.0
                if best_edges is None:
                    continue
                tot_bk = float(p2[b, k].sum().item())
                if tot_bk < 1e-12:
                    continue
                on = sum(float(p2[b, k, u, v]) for u, v in best_edges)
                off_fracs.append(1.0 - on / tot_bk)
                inrs.append(np.mean([float(I[b, k, u, v]) / (sigma ** 2) for u, v in best_edges]))
                wts.append(tot_bk)
        per_k_rate.append(rk)

    w = np.asarray(wts) if wts else np.asarray([1.0])
    off = float(np.average(off_fracs, weights=w)) if off_fracs else 0.0
    inr = float(np.average(inrs, weights=w)) if inrs else 0.0
    rate = min(per_k_rate) if per_k_rate else 0.0     # eq (10): min over commodities
    return dict(off=off, inr=inr, active_edges=active_edges,
                total_power=total_power, rate=rate, per_k=per_k_rate)


def route_metrics_multicast(P, links, sigma, subgraphs):
    """P [B,n,n]. Per-band best subgraph (max over subgraphs of min-edge R); rate = sum_b."""
    R, I = per_edge_rate_and_interf(P, links, sigma)
    p2 = (P.real ** 2)
    total_power, active_edges = _generic_pwr_stats(P)

    off_fracs, inrs, wts, rate = [], [], [], 0.0
    for b in range(B):
        best_val, best_edges = -1.0, None
        for sg in subgraphs:
            edges = [(int(u), int(v)) for (u, v) in sg]
            if not edges:
                continue
            mn = min(float(R[b, u, v]) for u, v in edges)
            if mn > best_val:
                best_val, best_edges = mn, edges
        rate += best_val if best_edges is not None else 0.0
        if best_edges is None:
            continue
        tot_b = float(p2[b].sum().item())
        if tot_b < 1e-12:
            continue
        on = sum(float(p2[b, u, v]) for u, v in best_edges)
        off_fracs.append(1.0 - on / tot_b)
        inrs.append(np.mean([float(I[b, u, v]) / (sigma ** 2) for u, v in best_edges]))
        wts.append(tot_b)

    w = np.asarray(wts) if wts else np.asarray([1.0])
    off = float(np.average(off_fracs, weights=w)) if off_fracs else 0.0
    inr = float(np.average(inrs, weights=w)) if inrs else 0.0
    return dict(off=off, inr=inr, active_edges=active_edges,
                total_power=total_power, rate=rate)


# ---------------------------------------------------------------------------
# Route construction (topology-only) + GNN allocation
# ---------------------------------------------------------------------------
def build_routes(adj, tx_, rx_):
    """Return the routing structures for the current problem, or None if no valid route."""
    adjc = adj.cpu()
    if problem == "single":
        t = int(tx_[0]) if isinstance(tx_, (list, tuple)) else int(tx_)
        r = int(rx_[0]) if isinstance(rx_, (list, tuple)) else int(rx_)
        raw = find_all_paths(adjc, t, r)
        if not raw:
            return None
        return dict(paths_raw=raw, paths_t=paths_to_tensor(raw, device))

    if problem == "multicast":
        t = int(tx_[0]) if isinstance(tx_, (list, tuple)) else int(tx_)
        rxl = [int(x) for x in as_list(rx_)]
        subgraphs = find_multicast_subgraphs(adj, t, rxl)
        if (subgraphs is None) or (len(subgraphs) == 0):
            return None
        return dict(subgraphs=subgraphs, subgraphs_per_band=[subgraphs for _ in range(B)])

    # multi family
    pairs = commodity_pairs(tx_, rx_)
    paths_k_raw = [find_all_paths(adjc, t, r) for (t, r) in pairs]
    if not any(paths_k_raw):
        return None
    paths_k_t = [
        paths_to_tensor(raw, device) if raw
        else torch.empty((0, 0), dtype=torch.long, device=device)
        for raw in paths_k_raw
    ]
    labels = [f"{t}->{r}" for (t, r) in pairs]
    return dict(paths_k_raw=paths_k_raw, paths_k_t=paths_k_t, labels=labels)


@torch.no_grad()
def gnn_alloc(d, sigma, routes):
    d.sigma = torch.tensor(sigma, device=device)
    kw = dict(problem=problem, tau_min=0.0, tau_max=0.0)
    if problem == "single":
        kw["paths"] = routes["paths_t"]
    elif problem == "multicast":
        kw["subgraphs_per_band"] = routes["subgraphs_per_band"]
    else:
        kw["paths_k"] = routes["paths_k_t"]
    rates_layer, p_list, _ = _compute_rates_per_layer(model, d.to(device), **kw)
    best = int(torch.stack(rates_layer).argmax())
    return p_list[best]


def unwrap_P(entry):
    """Greedy returns (P, Z) for the multi family; optimizer returns P. Normalize to P."""
    return entry[0] if isinstance(entry, (list, tuple)) else entry


def metrics_for(P, d, routes):
    links = d.links_matrix.to(device)
    adj = d.adj_matrix.to(device)
    P = P.to(device)
    if problem == "single":
        return route_metrics_single(P, links, sigma, routes["paths_raw"])
    if problem == "multicast":
        return route_metrics_multicast(P, links, sigma, routes["subgraphs"])
    return route_metrics_multi(P, links, adj, sigma, routes["paths_k_raw"])


def min_hops(paths):
    """Shortest Tx->Rx hop count over the candidate paths (edges = nodes-1)."""
    return min(len(p) - 1 for p in paths)


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------
print(f"device={device}  subset={SUBSET}  opt_iters={OPT_ITERS}  mean_channel_var={mean_var:.3f}\n")
HOP_BUCKETS = ["1", "2", "3+"]
METHODS = ("optimizer", "greedy", "GNN")

for snr in SNR_LIST:
    sigma = float((mean_var / (10.0 ** (snr / 10.0))) ** 0.5)
    acc = {m: {k: [] for k in ("off", "inr", "active_edges", "total_power", "rate")}
           for m in METHODS}
    hop_acc = {m: {b: [] for b in HOP_BUCKETS} for m in METHODS}   # single only
    perk_acc = {m: [] for m in METHODS}                             # multi family only
    hopk_acc = {m: {b: [] for b in HOP_BUCKETS} for m in METHODS}   # multi: per-commodity by pair hops

    for d in dataset:
        routes = build_routes(d.adj_matrix, d.tx, d.rx)
        if routes is None:
            continue

        P_gnn = gnn_alloc(d, sigma, routes)
        _, popt = evaluate_centralized_adam([d], B=B, noise_std=sigma,
                                            num_iterations=OPT_ITERS, problem=problem)
        P_opt = unwrap_P(popt[0])
        _, pgd = compute_centralized_greedy_power_rate([d], sigma_noise=sigma, problem=problem)
        P_gd = unwrap_P(pgd[0])

        h = min_hops(routes["paths_raw"]) if problem == "single" else None
        bucket = None
        if problem == "single":
            bucket = "1" if h <= 1 else ("2" if h == 2 else "3+")
        # per-commodity shortest-hop count (topology-only, same for all methods)
        per_k_hops = None
        if problem in MULTI_FAMILY:
            per_k_hops = [(min(len(pp) - 1 for pp in paths) if paths else None)
                          for paths in routes["paths_k_raw"]]

        for name, P in (("GNN", P_gnn), ("optimizer", P_opt), ("greedy", P_gd)):
            m = metrics_for(P, d, routes)
            for k in acc[name]:
                acc[name][k].append(m[k])
            if problem == "single":
                hop_acc[name][bucket].append(m["rate"])
            elif problem in MULTI_FAMILY:
                perk_acc[name].append(m["per_k"])
                for k, rk in enumerate(m["per_k"]):
                    hk = per_k_hops[k]
                    if hk is None:
                        continue
                    hkb = "1" if hk <= 1 else ("2" if hk == 2 else "3+")
                    hopk_acc[name][hkb].append(rk)

    print(f"===== SNR {snr} dB  (sigma={sigma:.3e}, sigma^2={sigma**2:.3e}) =====")
    print(f"{'method':<10}{'rate':>9}{'off_route':>11}{'INR_route':>11}{'act_edges':>11}{'tot_pwr':>9}")
    for name in METHODS:
        a = acc[name]
        if not a["rate"]:
            continue
        print(f"{name:<10}{np.mean(a['rate']):>9.3f}{np.mean(a['off']):>11.3f}"
              f"{np.mean(a['inr']):>11.2f}{np.mean(a['active_edges']):>11.2f}{np.mean(a['total_power']):>9.3f}")

    # --- second table ---
    if problem == "single":
        counts = {b: len(hop_acc["optimizer"][b]) for b in HOP_BUCKETS}
        print(f"  rate by Tx->Rx hops  (#samples  1:{counts['1']}  2:{counts['2']}  3+:{counts['3+']})")
        print(f"  {'method':<10}" + "".join(f"{b + '-hop':>10}" for b in HOP_BUCKETS) + f"{'gnn/opt':>10}")
        for name in METHODS:
            cells = []
            for b in HOP_BUCKETS:
                vals = hop_acc[name][b]
                cells.append(f"{np.mean(vals):>10.2f}" if vals else f"{'-':>10}")
            ratio = ""
            if name == "GNN":
                per = []
                for b in HOP_BUCKETS:
                    g, o = hop_acc["GNN"][b], hop_acc["optimizer"][b]
                    per.append(np.mean(g) / np.mean(o) if g and o else float("nan"))
                ratio = f"{np.nanmean(per):>10.2f}"
            print(f"  {name:<10}" + "".join(cells) + ratio)

    elif problem in MULTI_FAMILY:
        # per-commodity mean rate: exposes which message the max-min is starving.
        labels = [f"{t}->{r}" for (t, r) in commodity_pairs(tx, rx)]
        print(f"  per-commodity rate (K={K}; 'min' is the eq-10 objective)")
        print(f"  {'method':<10}" + "".join(f"{lab:>10}" for lab in labels) + f"{'min':>10}{'gnn/opt':>10}")
        opt_mean = np.mean(perk_acc["optimizer"], axis=0) if perk_acc["optimizer"] else None
        for name in METHODS:
            if not perk_acc[name]:
                continue
            arr = np.mean(perk_acc[name], axis=0)     # [K]
            cells = "".join(f"{v:>10.2f}" for v in arr)
            mn = f"{arr.min():>10.2f}"
            ratio = ""
            if name == "GNN" and opt_mean is not None:
                ratio = f"{(arr.min() / opt_mean.min()):>10.2f}" if opt_mean.min() > 0 else f"{'-':>10}"
            print(f"  {name:<10}" + cells + mn + ratio)

        # Per-commodity rate bucketed by THAT pair's shortest-hop count. Isolates the L-depth
        # routing limit: if GNN's 3+-hop pairs collapse to ~0 while opt/greedy serve them, the
        # decentralized L-hop receptive field is the barrier (not power shaping / concentration).
        counts = {b: len(hopk_acc["optimizer"][b]) for b in HOP_BUCKETS}
        print(f"  per-commodity rate by pair hops  "
              f"(#pairs  1:{counts['1']}  2:{counts['2']}  3+:{counts['3+']})")
        print(f"  {'method':<10}" + "".join(f"{b + '-hop':>10}" for b in HOP_BUCKETS) + f"{'gnn/opt':>10}")
        for name in METHODS:
            cells = []
            for b in HOP_BUCKETS:
                vals = hopk_acc[name][b]
                cells.append(f"{np.mean(vals):>10.2f}" if vals else f"{'-':>10}")
            ratio = ""
            if name == "GNN":
                per = []
                for b in HOP_BUCKETS:
                    g, o = hopk_acc["GNN"][b], hopk_acc["optimizer"][b]
                    per.append(np.mean(g) / np.mean(o) if g and o else float("nan"))
                ratio = f"{np.nanmean(per):>10.2f}"
            print(f"  {name:<10}" + "".join(cells) + ratio)

    print()
