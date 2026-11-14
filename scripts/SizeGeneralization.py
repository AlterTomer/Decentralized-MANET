r"""
Network-size generalization of MANET-GNN: trained models (e.g., one trained on |V| = 10 networks and one
trained on |V| = 30 networks) evaluated, without retraining, on test networks of one or more sizes.

    python scripts/SizeGeneralization.py <study config.ini> [<study config 2.ini> ...]

The study config lists the models ([Model ...] sections, each with its own depth) and the test sets
([Test ...] sections, each with its own size n, Tx/Rx nodes, and channel file); every model is evaluated on
every test set (format in utils/GNNStudyUtils.py, example in config/paper/Generalization/size_generalization.ini).
A model applies to any network size as long as the number of messages K matches its training (the
Tx/Rx node indices may differ).

Outputs in the study's `out dir`: <out prefix>.pkl / .csv, <out prefix>_<test>.png (mean rate vs SNR per
model on that test set, as in the paper's size-generalization figure), and, with two or more test sets,
<out prefix>_vs_size.png (mean rate vs network size per model at the `summary snr` values).

Without arguments, runs $MANET_DATA_ROOT/Config Files/Generalization/size_generalization.ini.
"""
import argparse
import os
import sys
from configparser import ConfigParser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.GNNStudyUtils import int_list, plot_rate_vs_snr, run_study

DEFAULT_CFG = Path(os.environ.get("MANET_DATA_ROOT", r"C:\Users\alter\OneDrive\Desktop\PhD\Decentralized MANET"),
                   "Config Files", "Generalization", "size_generalization.ini")


def summarize(results, study, models, tests, summary_snr):
    done_tests = [t for t in tests if any((m["name"], t["name"]) in results["mean"] for m in models)]
    for test in done_tests:
        curves = [((m["name"], test["name"]), m["label"]) for m in models if (m["name"], test["name"]) in results["mean"]]
        plot_rate_vs_snr(results, curves, study["out_dir"] / f"{study['out_prefix']}_{test['name']}.png")

    print("\nmean rate (model x test set)")
    print(f"  {'model':>20} | {'test':>10} | " + " | ".join(f"{s:>3} dB" for s in summary_snr))
    for m in models:
        for t in done_tests:
            key = (m["name"], t["name"])
            if key in results["mean"]:
                means = results["mean"][key]
                print(f"  {m['name']:>20} | {t['name']:>10} | "
                      + " | ".join(f"{means.get(s, float('nan')):6.3f}" for s in summary_snr))

    if len(done_tests) < 2:
        return
    order = sorted(done_tests, key=lambda t: t["n"])
    fig, axes = plt.subplots(1, len(summary_snr), figsize=(7 * len(summary_snr), 6), squeeze=False)
    for ax, s in zip(axes[0], summary_snr):
        for m in models:
            pts = [(t["n"], results["mean"][(m["name"], t["name"])].get(s, float("nan")))
                   for t in order if (m["name"], t["name"]) in results["mean"]]
            if pts:
                ax.plot(*zip(*pts), marker="s", linestyle="--", markersize=9, label=m["label"])
        ax.set_title(f"{s} dB", fontsize=18)
        ax.set_xlabel(r"Network size $|\mathcal{V}|$", fontsize=16)
        ax.set_ylabel("Mean Rate", fontsize=16)
        ax.grid(True, which="both")
        ax.legend(fontsize=14)
    fig.tight_layout()
    out = study["out_dir"] / f"{study['out_prefix']}_vs_size.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"saved: {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("configs", nargs="*", type=Path)
    args = ap.parse_args()
    for cfg in args.configs or [DEFAULT_CFG]:
        p = ConfigParser()
        p.read_file(open(cfg))
        summary_snr = int_list(p["Study"].get("summary snr", "10, 20, 30"))
        results, study, models, tests = run_study(cfg)
        summarize(results, study, models, tests, summary_snr)


if __name__ == "__main__":
    main()
