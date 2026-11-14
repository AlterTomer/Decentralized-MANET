"""
Soft (training) vs hard (inference) candidate-route selection gap.

Route selection is trained with a per-band softmax
pi_{b,k,m} = softmax(a_{b,k,m} / tau_r) but deployed with a hard per-band argmax. For each
multi-message benchmark config (F3/F5, and F4 if listed), this script evaluates the SAME trained
checkpoint on the SAME test set, under the TRUE (hard min/max) objective, with:

  * hard : per-band argmax route selection (deployed MANET-GNN; identical to the benchmark curve
           when FINAL_LAYER_ONLY=False)
  * soft@tau_end : per-band softmax at the final annealed training temperature
  * soft@1.0     : per-band softmax at the initial training temperature (reference)

and records the mean max_m pi_{b,k,m} at tau_end (how close the trained distribution is to one-hot).
Only the routing head is switched to train mode for the soft runs; the rest of the model stays in
eval mode, so dropout is off and the only difference between the runs is the route selection.

Output: a CSV next to each config's `fig data path` (suffix `_soft_vs_hard.csv`) and a printed table.
"""
import os
import csv
import math
from configparser import ConfigParser
from pathlib import Path

import numpy as np
import torch

from models.models import ChainedGNN
from utils.ComparisonUtils import (_build_eval_structures, _call_compute_rates_for_eval, _score_allocations,
                                   reference_channel_var)
from utils.DataUtils import generate_graph_data
from utils.EstimationUtils import masked_band_variance_from_dataset, precompute_csi_estimates
from utils.ParseUtils import parse_tx_rx_data

CFG_DIR = Path(os.path.join(os.environ.get("MANET_DATA_ROOT", r"C:\Users\alter\OneDrive\Desktop\PhD\Decentralized MANET"), r"Config Files"))
CFG_ARR = [
    CFG_DIR / "Multicommodity" / "R2" / "comp_multicommodity_EstimatedCSI_R2.ini",
    CFG_DIR / "Multiunicast" / "R2" / "comp_multiunicast_EstimatedCSI_R2.ini",
    CFG_DIR / "Convergencecast" / "R2" / "comp_convergecast_EstimatedCSI_R2.ini",
]
SNR_DB_LIST = list(range(0, 51, 5))
TAU_SOFT_LIST = [0.05, 1.0]   # first entry = final annealed training temperature ('routing tau end')
FINAL_LAYER_ONLY = False      # False = best layer per sample, exactly as in evaluate_across_snr


def load_setup(cfg_path):
    parser = ConfigParser()
    parser.read_file(open(cfg_path))
    tp, fp = parser["Train Parameters"], parser["Files"]
    mode, B, L, n = tp["mode"], int(tp["B"]), int(tp["L"]), int(tp["n"])
    num_samples, seed = int(tp["num samples"]), int(tp["SEED"])
    tx, rx = parse_tx_rx_data(tp["tx"].strip(), tp["rx"].strip())
    K = 1 if mode == "single" else (len(rx) if mode in {"multicast", "multi"} else len(tx))

    dataset = generate_graph_data(
        n_list=[n] * num_samples, tx_list=[tx] * num_samples, rx_list=[rx] * num_samples,
        sigma_list=[float(tp["sigma"])] * num_samples, B=B, K=K, problem=mode, seed=seed,
        channel_path=fp.get("channel path", None), device="cpu",
    )
    if int(tp["LMMSE estimation"]) == 1:
        dataset = precompute_csi_estimates(
            dataset, pilots_M=4, pilot_power=1,
            prior_var=masked_band_variance_from_dataset(dataset),
            est_noise_std=None, seed=seed, device=torch.device("cpu"),
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChainedGNN(
        num_layers=L, B=B, K=K, problem=mode, dropout=float(tp["dropout"]), use_jk=True,
        jk_mode="concat", noise_conditioning=int(tp.get("noise conditioning", 0)) == 1,
        z_mode=tp.get("z mode", "edge").strip().lower(),
    ).to(device)
    sd = torch.load(fp["model path"], map_location=device, weights_only=False)["model_state_dict"]
    sd = {(k.replace("head.", "p_head.", 1) if k.startswith("head.") else k): v for k, v in sd.items()}
    msd = model.state_dict()
    bad = [k for k, v in sd.items() if k in msd and v.shape != msd[k].shape]
    if bad:
        raise RuntimeError(f"Checkpoint/model shape mismatch: {bad[:4]}")
    model.load_state_dict(sd, strict=False)

    meta = dict(mode=mode, B=B, max_hops=int(tp.get("max route hops", L)),
                reduce=tp.get("reduce", "fair").strip().lower(), out=fp["fig data path"])
    return dataset, model, device, meta


def set_selection(model, soft, tau):
    """Whole model in eval (no dropout); routing head in train mode only for soft selection."""
    model.eval()
    model.route_head.train(soft)
    model.routing_tau = tau


def run_pass(model, dataset, device, meta, sigma, soft, tau, cache, conf_acc=None):
    set_selection(model, soft, tau)
    hook = None
    sample_conf = []
    if conf_acc is not None:
        B = meta["B"]
        def _hook(_m, _inp, out):  # scorer output [B*M, 1] for one commodity at one layer
            s = out.detach().reshape(B, -1)
            if s.shape[1] > 0:
                sample_conf.append(torch.softmax(s / tau, dim=1).max(dim=1).values.mean().item())
        hook = model.route_head.scorer.register_forward_hook(_hook)
    store = []
    with torch.no_grad():
        for i, d in enumerate(dataset):
            d.sigma = torch.tensor(sigma, device=device)
            paths, sub, paths_k, skip = _build_eval_structures(i, d, meta["B"], meta["mode"], device, cache)
            if skip:
                store.append(None)
                continue
            setattr(d, "problem", meta["mode"])
            d = d.to(device)
            sample_conf.clear()
            r_layers, p_list, z_list = _call_compute_rates_for_eval(
                model, d, paths, sub, paths_k, meta["mode"], meta["reduce"], meta["max_hops"],
                decentralized_gnn_inference=True, hard_candidate_gnn_inference=not soft,
            )
            if conf_acc is not None and sample_conf:
                # The scorer runs once per (layer, commodity); keep the final layer's calls only.
                per_layer = len(sample_conf) // max(len(r_layers), 1)
                conf_acc.extend(sample_conf[-per_layer:] if per_layer > 0 else sample_conf)
            r = torch.stack(r_layers)
            b = len(r_layers) - 1 if FINAL_LAYER_ONLY else int(r.argmax())  # same layer choice as the benchmark
            store.append((p_list[b].detach(), z_list[b].detach() if z_list else None))
    if hook is not None:
        hook.remove()
    # Decisions were made on d.links_matrix (H_hat under estimated CSI); the achieved rate is
    # scored on the true channel (d.links_true), exactly as in evaluate_across_snr.
    return _score_allocations(dataset, store, None, sigma, meta["mode"], meta["reduce"], meta["B"], device,
                              cache, use_true_csi=True)


def ci95(x):
    return 1.96 * x.std(ddof=1) / math.sqrt(len(x)) if len(x) > 1 else 0.0


if __name__ == "__main__":
    for cfg_path in CFG_ARR:
        _p = ConfigParser()
        _p.read_file(open(cfg_path))
        if not Path(_p["Files"]["model path"].strip()).exists():
            print(f"[skip] {cfg_path.name}: model not trained yet")
            continue
        dataset, model, device, meta = load_setup(cfg_path)
        if model.z_mode != "candidate":
            raise RuntimeError(f"{cfg_path.name}: z_mode='{model.z_mode}', soft/hard gap applies to 'candidate' only.")
        noise_ref = reference_channel_var(dataset)  # true-channel power (same noise as the benchmark)
        cache = {}
        rows = []
        print(f"\n=== {cfg_path.name} | mode={meta['mode']} | N={len(dataset)} ===")
        print(f"{'SNR':>4} | {'hard':>14} | " + " | ".join(f"soft@{t:<5}" + " " * 7 for t in TAU_SOFT_LIST)
              + f" | {'gap@tau_end':>11} | {'max pi':>6}")
        for snr_db in SNR_DB_LIST:
            sigma = (noise_ref / 10.0 ** (snr_db / 10.0)) ** 0.5
            hard = run_pass(model, dataset, device, meta, sigma, soft=False, tau=TAU_SOFT_LIST[0], cache=cache)
            row = dict(snr_db=snr_db, hard_mean=hard.mean(), hard_ci95=ci95(hard))
            conf = []
            for j, t in enumerate(TAU_SOFT_LIST):
                soft = run_pass(model, dataset, device, meta, sigma, soft=True, tau=t, cache=cache,
                                conf_acc=conf if j == 0 else None)
                row[f"soft{t}_mean"], row[f"soft{t}_ci95"] = soft.mean(), ci95(soft)
                if j == 0:
                    # Relative gap of the deployed hard policy w.r.t. the trained soft policy (paired).
                    row["rel_gap_tau_end_pct"] = 100.0 * (hard.mean() - soft.mean()) / max(soft.mean(), 1e-12)
                    row["frac_hard_ge_soft"] = float((hard >= soft - 1e-9).mean())
            row["mean_max_pi_tau_end"] = float(np.mean(conf)) if conf else float("nan")
            rows.append(row)
            print(f"{snr_db:>4} | {row['hard_mean']:7.3f}±{row['hard_ci95']:.3f} | "
                  + " | ".join(f"{row[f'soft{t}_mean']:7.3f}±{row[f'soft{t}_ci95']:.3f}" for t in TAU_SOFT_LIST)
                  + f" | {row['rel_gap_tau_end_pct']:+10.2f}% | {row['mean_max_pi_tau_end']:.3f}")

        out = Path(meta["out"]).with_name(Path(meta["out"]).stem + "_soft_vs_hard.csv")
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"saved: {out}")
