"""
Shared engine for the MANET-GNN-only studies: depth ablation (scripts/DepthAblation.py) and network-size
generalization (scripts/SizeGeneralization.py).

A study config (.ini) lists one or more trained models and one or more test sets; every model is evaluated
on every test set across an SNR grid, exactly as MANET-GNN is evaluated in the benchmark
(utils/ComparisonUtils.evaluate_across_snr): decentralized candidate routing with hard per-band argmax for
F3-F5, the same per-network layer choice, the true-channel SNR reference, and scoring with the original
(hard min/max) objective on the true channel.

Config layout (see config/paper/Generalization/ for complete examples):

    [Study]
    mode: multiunicast            # single / multicast / multi / converge / multiunicast
    B: 6
    sigma: 1
    dropout: 0.2
    noise conditioning: 1         # must match the training configs
    z mode: candidate
    reduce: fair
    LMMSE estimation: 0
    seed: 15337
    snr list: 0, 2, 4, ..., 50
    layer: best                   # best (as in the benchmark) or final
    out dir: <folder for the .csv/.pkl/.png outputs>
    out prefix: DepthAblation_n_30_multiunicast

    [Model L10]                   # one section per trained model ("Model <name>")
    label: $L=10$                 # legend label (optional; default: L = 2 x layers)
    layers: 5                     # gated layers = the training config's L (paper: L = 2 x layers rounds)
    max route hops: 5             # candidate-route hop limit (optional; default: layers)
    path: <checkpoint .pth>

    [Test n30]                    # one section per test set ("Test <name>")
    label: $|\\mathcal{V}|=30$
    n: 30
    tx: 0, 2, 4, 6
    rx: 3, 5, 7, 9
    num samples: 200
    channel path: <.mat or .pt channel file>
"""
import math
import pickle
from configparser import ConfigParser
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from models.models import ChainedGNN
from utils.ComparisonUtils import (_build_eval_structures, _call_compute_rates_for_eval, _score_allocations,
                                   reference_channel_var)
from utils.DataUtils import generate_graph_data
from utils.EstimationUtils import masked_band_variance_from_dataset, precompute_csi_estimates
from utils.ParseUtils import parse_tx_rx_data

MULTI = {"multi", "converge", "multiunicast"}


def int_list(raw):
    return [int(float(x)) for x in str(raw).replace(" ", "").split(",") if x]


def read_study(cfg_path):
    """Parse a study config into (study dict, list of model dicts, list of test dicts)."""
    p = ConfigParser()
    p.read_file(open(cfg_path))
    s = p["Study"]
    study = dict(
        mode=s["mode"].strip(), B=int(s["B"]), sigma=float(s.get("sigma", 1)),
        dropout=float(s.get("dropout", 0.2)), noise_conditioning=int(s.get("noise conditioning", 0)) == 1,
        z_mode=s.get("z mode", "edge").strip().lower(), reduce=s.get("reduce", "fair").strip().lower(),
        lmmse=int(s.get("LMMSE estimation", 0)) == 1, seed=int(s.get("seed", 15337)),
        snr_list=int_list(s.get("snr list", "0,5,10,15,20,25,30,35,40,45,50")),
        layer=s.get("layer", "best").strip().lower(),
        out_dir=Path(s["out dir"].strip()), out_prefix=s.get("out prefix", Path(cfg_path).stem).strip(),
    )
    if study["layer"] not in {"best", "final"}:
        raise ValueError("[Study] layer must be 'best' or 'final'")
    models, tests = [], []
    for name in p.sections():
        sec = p[name]
        if name.lower().startswith("model"):
            layers = int(sec["layers"])
            models.append(dict(name=name.split(None, 1)[-1], layers=layers,
                               label=sec.get("label", f"$L={2 * layers}$"),
                               max_hops=int(sec.get("max route hops", layers)), path=Path(sec["path"].strip())))
        elif name.lower().startswith("test"):
            tests.append(dict(name=name.split(None, 1)[-1], n=int(sec["n"]), tx=sec["tx"].strip(),
                              rx=sec["rx"].strip(), num_samples=int(sec["num samples"]),
                              label=sec.get("label", f"$|\\mathcal{{V}}|={sec['n']}$"),
                              channel_path=sec["channel path"].strip()))
    if not models or not tests:
        raise ValueError(f"{cfg_path}: needs at least one [Model ...] and one [Test ...] section")
    return study, models, tests


def num_messages(mode, tx, rx):
    if mode == "single":
        return 1
    return len(rx) if mode in {"multicast", "multi"} else len(tx)


def build_test_set(study, test):
    """Test networks from a channel file, with (optionally) precomputed LMMSE estimates."""
    tx, rx = parse_tx_rx_data(test["tx"], test["rx"])
    S = test["num_samples"]
    dataset = generate_graph_data(
        n_list=[test["n"]] * S, tx_list=[tx] * S, rx_list=[rx] * S, sigma_list=[study["sigma"]] * S,
        B=study["B"], K=num_messages(study["mode"], tx, rx), problem=study["mode"], seed=study["seed"],
        channel_path=test["channel_path"], device="cpu",
    )
    if study["lmmse"]:
        dataset = precompute_csi_estimates(
            dataset, pilots_M=4, pilot_power=1, prior_var=masked_band_variance_from_dataset(dataset),
            est_noise_std=None, seed=study["seed"], device=torch.device("cpu"),
        )
    return dataset, num_messages(study["mode"], tx, rx)


def load_model(study, model, K, device):
    """Build ChainedGNN with the model's depth and load its checkpoint (shape-checked)."""
    net = ChainedGNN(
        num_layers=model["layers"], B=study["B"], K=K, problem=study["mode"], dropout=study["dropout"],
        use_jk=True, jk_mode="concat", noise_conditioning=study["noise_conditioning"], z_mode=study["z_mode"],
    ).to(device)
    sd = torch.load(model["path"], map_location=device, weights_only=False)["model_state_dict"]
    sd = {(k.replace("head.", "p_head.", 1) if k.startswith("head.") else k): v for k, v in sd.items()}
    msd = net.state_dict()
    bad = [k for k, v in sd.items() if k in msd and v.shape != msd[k].shape]
    # A depth mismatch shows up as parameters missing from the checkpoint (model deeper than trained) or as
    # checkpoint entries of the model's own modules that the model lacks (model shallower than trained).
    missing = [k for k, _ in net.named_parameters() if k not in sd]
    top = {k.split(".", 1)[0] for k in msd}
    extra = [k for k in sd if k not in msd and k.split(".", 1)[0] in top]
    if bad or missing or extra:
        raise RuntimeError(f"{model['path'].name}: checkpoint does not match the model "
                           f"(layers={model['layers']}, noise conditioning={study['noise_conditioning']}); "
                           f"shape mismatch {bad[:3]}, missing {missing[:3]}, unexpected {extra[:3]}")
    net.load_state_dict(sd, strict=False)
    net.eval()
    return net


def evaluate_model(study, net, model, dataset, device, cache):
    """Per-network MANET-GNN rates at every SNR of the study: {snr_db: np.ndarray [N]}."""
    mode, B = study["mode"], study["B"]
    decentralized = study["z_mode"] == "candidate" and mode in MULTI
    noise_ref = reference_channel_var(dataset)   # true-channel power (same noise as the benchmark)
    out = {}
    for snr_db in study["snr_list"]:
        sigma = (noise_ref / 10.0 ** (snr_db / 10.0)) ** 0.5
        store = []
        with torch.no_grad():
            for i, d in enumerate(dataset):
                d.sigma = torch.tensor(sigma, device=device)
                paths, sub, paths_k, skip = _build_eval_structures(i, d, B, mode, device, cache)
                if skip:
                    store.append(None)
                    continue
                setattr(d, "problem", mode)
                d = d.to(device)
                r_layers, p_list, z_list = _call_compute_rates_for_eval(
                    net, d, paths, sub, paths_k, mode, study["reduce"], model["max_hops"],
                    decentralized_gnn_inference=decentralized, hard_candidate_gnn_inference=True,
                )
                r = torch.stack([torch.as_tensor(x, dtype=torch.float32).reshape(()) for x in r_layers])
                b = len(r_layers) - 1 if study["layer"] == "final" else int(r.argmax())
                store.append((p_list[b].detach(), z_list[b].detach() if z_list else None))
        # Decisions are made on d.links_matrix (the estimate under estimated CSI); the achieved rate is
        # scored on the true channel, exactly as in evaluate_across_snr.
        out[snr_db] = _score_allocations(dataset, store, None, sigma, mode, study["reduce"], B, device, cache,
                                         use_true_csi=True)
        print(f"    {snr_db:3d} dB: mean rate {out[snr_db].mean():.4f}")
    return out


def run_study(cfg_path):
    """Evaluate every model on every test set. Returns the results dict (also saved as .pkl)."""
    study, models, tests = read_study(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = dict(config=str(cfg_path), study={k: (str(v) if isinstance(v, Path) else v) for k, v in study.items()},
                   models=[{k: (str(v) if isinstance(v, Path) else v) for k, v in m.items()} for m in models],
                   tests=tests, per_sample={}, mean={}, ci95={})
    for test in tests:
        missing = ([] if Path(test["channel_path"]).exists() else [test["channel_path"]]) \
            + [str(m["path"]) for m in models if not m["path"].exists()]
        if missing:
            print(f"[skip] test '{test['name']}': missing\n    " + "\n    ".join(missing))
            continue
        dataset, K = build_test_set(study, test)
        cache = {}   # path/subgraph enumeration is topology-only: shared by all models on this test set
        for model in models:
            print(f"[{test['name']}] model '{model['name']}' ({model['layers']} gated layers, "
                  f"{2 * model['layers']} rounds) on {len(dataset)} networks")
            net = load_model(study, model, K, device)
            rates = evaluate_model(study, net, model, dataset, device, cache)
            key = (model["name"], test["name"])
            results["per_sample"][key] = {s: r.tolist() for s, r in rates.items()}
            results["mean"][key] = {s: float(r.mean()) for s, r in rates.items()}
            results["ci95"][key] = {s: (1.96 * r.std(ddof=1) / math.sqrt(len(r)) if len(r) > 1 else 0.0)
                                    for s, r in rates.items()}
    study["out_dir"].mkdir(parents=True, exist_ok=True)
    pkl = study["out_dir"] / f"{study['out_prefix']}.pkl"
    with open(pkl, "wb") as f:
        pickle.dump(results, f)
    write_csv(results, study["out_dir"] / f"{study['out_prefix']}.csv")
    print(f"saved: {pkl}")
    return results, study, models, tests


def write_csv(results, path):
    rows = ["model,test,snr_db,mean_rate,ci95"]
    for (m, t), means in results["mean"].items():
        for s, v in means.items():
            rows.append(f"{m},{t},{s},{v:.6f},{results['ci95'][(m, t)][s]:.6f}")
    Path(path).write_text("\n".join(rows) + "\n", encoding="utf-8")


def plot_rate_vs_snr(results, curves, path, title=None):
    """curves: list of (key, label). Same style as the paper's MANET-GNN figures (log y axis)."""
    plt.figure(figsize=(16, 12))
    for key, label in curves:
        if key not in results["mean"]:
            continue
        snr = sorted(results["mean"][key])
        y = np.array([results["mean"][key][s] for s in snr])
        e = np.array([results["ci95"][key][s] for s in snr])
        lower = np.minimum(e, y * (1 - 1e-6))          # keep whiskers positive on the log axis
        plt.errorbar(snr, y, yerr=[lower, e], marker="s", linestyle="--", label=label,
                     markersize=12, capsize=5, capthick=1.5, elinewidth=1.5)
    plt.yscale("log")
    plt.xlabel("SNR (dB)", fontsize=30)
    plt.ylabel("Mean Rate", fontsize=30)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.grid(True, which="both")
    plt.legend(fontsize=30)
    if title:
        plt.title(title, fontsize=24)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"saved: {path}")
