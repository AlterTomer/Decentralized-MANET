"""
Mean rate versus number of communication rounds.

Compares the decentralized methods at an equal signaling budget, on the R2 test sets:
  * MANET-GNN: the allocation decoded after gated layer l = 1, ..., L/2 (each layer = 2 neighbor
    message exchanges), i.e., at 2, 4, ..., L rounds, with decentralized candidate routing whose hop
    horizon grows with the layer, exactly as in deployment.
  * Decentralized Greedy-Split and Decentralized Widest-Path: Bellman-Ford route discovery stopped
    after t = 1, ..., |V|-1 rounds (t = |V|-1 is full convergence).
  * Equal-Split: 0 rounds (no exchanges).
  * Centralized optimizer: horizontal reference read from the R2 benchmark pickle (it needs the CSI
    of all links at a central unit, so it has no per-round counterpart).

Every allocation is decided from d.links_matrix (H_hat under estimated CSI) and scored on the true
channel with the original objective, exactly as in evaluate_across_snr. The SNR -> noise mapping uses
the true-channel power. Outputs per config: <fig data stem>_rate_vs_comm.csv / .pkl / .png.
"""
import os
import csv
import math
import pickle
from configparser import ConfigParser
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from scripts.RoutingSoftVsHard import load_setup
from utils.CentralizedUtils import (compute_decentralized_best_single_channel_rate,
                                    compute_decentralized_greedy_power_rate, compute_equal_power_bound)
from utils.ComparisonUtils import (_build_eval_structures, _call_compute_rates_for_eval, _score_allocations,
                                   reference_channel_var)

CFG_DIR = Path(os.path.join(os.environ.get("MANET_DATA_ROOT", r"C:\Users\alter\OneDrive\Desktop\PhD\Decentralized MANET"), r"Config Files"))
FRAMEWORKS = (("Single Tx-Rx", "unicast"), ("Multicommodity", "multicommodity"),
              ("Multiunicast", "multiunicast"), ("Convergencecast", "convergecast"))
CSI_LIST = ("FullCSI",)          # add "EstimatedCSI" to also produce the estimated-CSI version
SNR_DB_LIST = (10, 20, 30)


def ci95(x):
    x = np.asarray(x, dtype=float)
    return 1.96 * x.std(ddof=1) / math.sqrt(len(x)) if len(x) > 1 else 0.0


def gnn_per_layer(model, dataset, device, meta, sigma, cache):
    """Rates of the allocation decoded after each gated layer, scored on the true channel."""
    model.eval()
    n_layers = model.num_layers
    stores = [[] for _ in range(n_layers)]
    with torch.no_grad():
        for i, d in enumerate(dataset):
            d.sigma = torch.tensor(sigma, device=device)
            paths, sub, paths_k, skip = _build_eval_structures(i, d, meta["B"], meta["mode"], device, cache)
            if skip:
                for s in stores:
                    s.append(None)
                continue
            setattr(d, "problem", meta["mode"])
            _, p_list, z_list = _call_compute_rates_for_eval(
                model, d.to(device), paths, sub, paths_k, meta["mode"], meta["reduce"], meta["max_hops"],
                decentralized_gnn_inference=True, hard_candidate_gnn_inference=True)
            for l in range(n_layers):
                stores[l].append((p_list[l].detach(), z_list[l].detach() if z_list else None))
    return [_score_allocations(dataset, stores[l], None, sigma, meta["mode"], meta["reduce"], meta["B"],
                               device, cache, use_true_csi=True) for l in range(n_layers)]


def rescore(dataset, out, meta, sigma, device, cache):
    rates, store = out[0], out[1]
    aux = out[2] if len(out) > 2 else None
    return _score_allocations(dataset, store, aux, sigma, meta["mode"], meta["reduce"], meta["B"], device,
                              cache, use_true_csi=True)


def centralized_reference(meta):
    """Centralized-optimizer mean rate per SNR from the R2 benchmark pickle, if it exists."""
    pkl = Path(meta["out"])
    if not pkl.exists():
        return {}
    with open(pkl, "rb") as f:
        res = pickle.load(f)
    return {int(s): float(v) for s, v in res.get("centralized", {}).items()}


def run(cfg_path):
    dataset, model, device, meta = load_setup(cfg_path)
    ds = [dataset[i] for i in range(len(dataset))]
    n = int(ds[0].adj_matrix.shape[0])
    ref = reference_channel_var(ds)
    cache, paths_cache = {}, {}
    rows = []
    cent = centralized_reference(meta)
    for snr in SNR_DB_LIST:
        sigma = (ref / 10 ** (snr / 10)) ** 0.5
        for d in ds:
            d.sigma = torch.tensor(sigma, device=device)
        for l, r in enumerate(gnn_per_layer(model, ds, device, meta, sigma, cache), start=1):
            rows.append(dict(snr_db=snr, method="MANET-GNN", rounds=2 * l, mean=r.mean(), ci95=ci95(r)))
        for t in range(1, n):
            g = rescore(ds, compute_decentralized_greedy_power_rate(
                ds, sigma_noise=sigma, problem=meta["mode"], max_iters=t, paths_cache=paths_cache,
                reduce=meta["reduce"]), meta, sigma, device, cache)
            rows.append(dict(snr_db=snr, method="Decentralized Greedy-Split", rounds=t, mean=g.mean(), ci95=ci95(g)))
            w = rescore(ds, compute_decentralized_best_single_channel_rate(
                ds, problem=meta["mode"], sigma_noise=sigma, max_iters=t, paths_cache=paths_cache,
                reduce=meta["reduce"]), meta, sigma, device, cache)
            rows.append(dict(snr_db=snr, method="Decentralized Widest-Path", rounds=t, mean=w.mean(), ci95=ci95(w)))
        e = rescore(ds, compute_equal_power_bound(ds, sigma_noise=sigma, problem=meta["mode"],
                                                  paths_cache=paths_cache, reduce=meta["reduce"]),
                    meta, sigma, device, cache)
        rows.append(dict(snr_db=snr, method="Equal-Split", rounds=0, mean=e.mean(), ci95=ci95(e)))
        if snr in cent:
            rows.append(dict(snr_db=snr, method="Centralized Optimizer", rounds=float("nan"),
                             mean=cent[snr], ci95=float("nan")))
        print(f"  SNR {snr} dB: " + " | ".join(
            f"{r['method']}@{r['rounds']}={r['mean']:.3f}" for r in rows
            if r["snr_db"] == snr and (r["method"] != "Decentralized Widest-Path" or r["rounds"] in (1, 3, n - 1))
            and (r["method"] != "Decentralized Greedy-Split" or r["rounds"] in (1, 3, n - 1))))
    return rows, n, meta


def plot(rows, n, meta, title, png):
    fig, axes = plt.subplots(1, len(SNR_DB_LIST), figsize=(5.2 * len(SNR_DB_LIST), 4.2), squeeze=False)
    style = {"MANET-GNN": ("tab:orange", "s", "--"),
             "Decentralized Greedy-Split": ("tab:pink", "o", ":"),
             "Decentralized Widest-Path": ("tab:purple", "v", "-")}
    for ax, snr in zip(axes[0], SNR_DB_LIST):
        sub = [r for r in rows if r["snr_db"] == snr]
        for m, (c, mk, ls) in style.items():
            pts = sorted((r["rounds"], r["mean"], r["ci95"]) for r in sub if r["method"] == m)
            if pts:
                x, y, e = map(np.array, zip(*pts))
                ax.errorbar(x, y, yerr=e, color=c, marker=mk, linestyle=ls, capsize=3, label=m)
        for m, c, ls in (("Equal-Split", "gray", "--"), ("Centralized Optimizer", "tab:blue", "-")):
            v = [r["mean"] for r in sub if r["method"] == m]
            if v:
                ax.axhline(v[0], color=c, linestyle=ls, linewidth=1.2,
                           label=m + (" (0 rounds)" if m == "Equal-Split" else " (global CSI)"))
        ax.set_xticks(range(0, n))
        ax.set_xlabel("Communication rounds")
        ax.set_title(f"SNR = {snr} dB")
        ax.grid(True, alpha=0.4)
    axes[0][0].set_ylabel("Mean rate")
    axes[0][0].legend(fontsize=8, loc="best")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(png, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    for fw, tag in FRAMEWORKS:
        for csi in CSI_LIST:
            cfg = CFG_DIR / fw / "R2" / f"comp_{tag}_{csi}_R2.ini"
            p = ConfigParser()
            p.read_file(open(cfg))
            missing = [k for k in ("channel path", "model path") if not Path(p["Files"][k].strip()).exists()]
            if missing:
                print(f"[skip] {cfg.name}: missing {', '.join(missing)}")
                continue
            print(f"\n=== {cfg.name} ===")
            rows, n, meta = run(cfg)
            stem = Path(meta["out"])
            out = stem.with_name(stem.stem + "_rate_vs_comm")
            with open(out.with_suffix(".csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            with open(out.with_suffix(".pkl"), "wb") as f:
                pickle.dump(rows, f)
            fig_png = Path(p["Files"]["fig path"].strip())
            png = fig_png.with_name(fig_png.stem + "_rate_vs_comm.png")
            plot(rows, n, meta, f"{tag} ({csi})", png)
            print(f"saved: {out.with_suffix('.csv')} | {png}")
