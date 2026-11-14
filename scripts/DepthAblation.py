r"""
Depth ablation of MANET-GNN: models that differ only in the number of gated layers (the paper's L = 2 x
layers communication rounds), evaluated on the same test networks.

1) Make the training configs (one per depth) from a training config:

    python scripts/DepthAblation.py --make-train-configs <training config.ini> --layers 1 2 3 5 7 [--out-dir <dir>]

   Each copy sets `L` (gated layers) and `max route hops` (candidate routes of at most L/2 hops) to the
   depth; everything else (data, hyperparameters, checkpoint prefix) is kept, and the checkpoint name
   carries the depth (`<prefix>_L<layers>_val..._<timestamp>.pth`). Train them with
   scripts/Decentralized_MANET_Simulation.py.

2) Evaluate them (one [Model ...] section per depth, one or more [Test ...] sections; format in
   utils/GNNStudyUtils.py, example in config/paper/Generalization/depth_ablation.ini):

    python scripts/DepthAblation.py <study config.ini> [<study config 2.ini> ...]

   Outputs in the study's `out dir`: <out prefix>.pkl / .csv, <out prefix>_<test>.png (mean rate vs SNR
   per depth, as in the paper's depth-ablation figure), and <out prefix>_<test>_vs_depth.png (mean rate vs
   L at the `summary snr` values).

Without arguments, runs $MANET_DATA_ROOT/Config Files/Generalization/depth_ablation.ini.
"""
import argparse
import os
import re
import sys
from configparser import ConfigParser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.GNNStudyUtils import int_list, plot_rate_vs_snr, run_study

DEFAULT_CFG = Path(os.environ.get("MANET_DATA_ROOT", r"C:\Users\alter\OneDrive\Desktop\PhD\Decentralized MANET"),
                   "Config Files", "Generalization", "depth_ablation.ini")


def setkey(text, key, value):
    new, n = re.subn(rf"(?m)^{re.escape(key)}\s*:.*$", lambda _: f"{key}: {value}", text)
    if n == 0:
        new = text.replace("[Train Parameters]\n", f"[Train Parameters]\n{key}: {value}\n", 1)
    elif n > 1:
        raise ValueError(f"'{key}' appears {n} times")
    return new


def make_train_configs(template, layers, out_dir):
    template = Path(template)
    out_dir = Path(out_dir) if out_dir else template.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    text = template.read_text(encoding="utf-8")
    for l in layers:
        t = setkey(text, "L", l)
        if re.search(r"(?m)^max route hops\s*:", t) or re.search(r"(?m)^z mode\s*:\s*candidate", t):
            t = setkey(t, "max route hops", l)
        dst = out_dir / f"{template.stem}_L{l}{template.suffix}"
        if dst.exists():
            print(f"exists, kept: {dst}")
            continue
        dst.write_text(t, encoding="utf-8")
        print(f"wrote {dst}  (L = {l} gated layers = {2 * l} rounds)")


def summarize(results, study, models, tests, summary_snr):
    for test in tests:
        order = sorted((m for m in models if (m["name"], test["name"]) in results["mean"]), key=lambda m: m["layers"])
        if not order:
            continue
        out = study["out_dir"] / f"{study['out_prefix']}_{test['name']}.png"
        plot_rate_vs_snr(results, [((m["name"], test["name"]), m["label"]) for m in order], out)

        print(f"\n[{test['name']}] mean rate vs depth")
        print("  rounds L | " + " | ".join(f"{s:>3} dB" for s in summary_snr))
        for m in order:
            means = results["mean"][(m["name"], test["name"])]
            print(f"  {2 * m['layers']:8d} | " + " | ".join(f"{means.get(s, float('nan')):6.3f}" for s in summary_snr))

        plt.figure(figsize=(12, 9))
        xs = [2 * m["layers"] for m in order]
        for s in summary_snr:
            ys = [results["mean"][(m["name"], test["name"])].get(s, float("nan")) for m in order]
            plt.plot(xs, ys, marker="s", linestyle="--", markersize=10, label=f"{s} dB")
        plt.xlabel("Communication rounds $L$", fontsize=24)
        plt.ylabel("Mean Rate", fontsize=24)
        plt.xticks(xs, fontsize=18)
        plt.yticks(fontsize=18)
        plt.grid(True, which="both")
        plt.legend(fontsize=20)
        plt.tight_layout()
        out = study["out_dir"] / f"{study['out_prefix']}_{test['name']}_vs_depth.png"
        plt.savefig(out, dpi=300)
        plt.close()
        print(f"saved: {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("configs", nargs="*", type=Path, help="study configs (default: %(default)s)")
    ap.add_argument("--make-train-configs", metavar="TEMPLATE", help="write one training config per depth")
    ap.add_argument("--layers", nargs="+", type=int, default=[1, 2, 3, 5, 7],
                    help="gated layers per depth (default: 1 2 3 5 7, i.e., L = 2, 4, 6, 10, 14 rounds)")
    ap.add_argument("--out-dir", help="folder for the generated training configs (default: next to the template)")
    args = ap.parse_args()

    if args.make_train_configs:
        make_train_configs(args.make_train_configs, args.layers, args.out_dir)
        return
    for cfg in args.configs or [DEFAULT_CFG]:
        p = ConfigParser()
        p.read_file(open(cfg))
        summary_snr = int_list(p["Study"].get("summary snr", "10, 20, 30"))
        results, study, models, tests = run_study(cfg)
        summarize(results, study, models, tests, summary_snr)


if __name__ == "__main__":
    main()
