"""
Same-model CSI ratio.

For MANET-GNN and the FFN, compares the ESTIMATED-CSI model deciding on the estimated channels
(EstimatedCSI_..._R2.pkl) with the same model deciding on the true channels (EstModelTrueCSI_..._R2.pkl,
from scripts/make_same_model_csi_configs.py); both are scored on the true channels, over the same networks.
Prints, per framework, the ratio at every SNR with a paired 95% confidence interval of the difference,
and the min--max range over 0-50 dB as LaTeX cells for Table tab:csi.
"""
import os
import pickle

import numpy as np

FD = os.path.join(os.environ.get("MANET_DATA_ROOT", r"C:\Users\alter\OneDrive\Desktop\PhD\Decentralized MANET"), r"Figures Data")
FW = [("Single Tx-Rx", "n_10_B_6_unicast"), ("Multicast", "n_6_Q_2_B_6_multicast"),
      ("Multicommodity", "n_10_B_6_multicommodity"), ("Convergencecast", "n_10_B_6_converge"),
      ("Multiunicast", "n_10_B_6_multiunicast")]

cells = {"gnn": [], "ffn": []}
for folder, tag in FW:
    pe = os.path.join(FD, folder, f"EstimatedCSI_Quadriga_benchmark_{tag}_R2.pkl")
    pt = os.path.join(FD, folder, f"EstModelTrueCSI_Quadriga_benchmark_{tag}_R2.pkl")
    if not (os.path.exists(pe) and os.path.exists(pt)):
        print(f"{tag}: missing {'' if os.path.exists(pe) else pe} {'' if os.path.exists(pt) else pt}")
        for k in cells:
            cells[k].append(r"\textcolor{red}{TODO}")
        continue
    e, t = pickle.load(open(pe, "rb")), pickle.load(open(pt, "rb"))
    for k in ("gnn", "ffn"):
        ratios, line = [], []
        for s in sorted(t[k]):
            if s not in e[k]:
                continue
            a = np.asarray(t["per_sample"][k][s], float)      # true CSI as input
            b = np.asarray(e["per_sample"][k][s], float)      # estimated CSI as input
            m = np.isfinite(a) & np.isfinite(b)
            a, b = a[m], b[m]
            d = b - a
            ci = 1.96 * d.std(ddof=1) / np.sqrt(len(d))
            r = 100 * b.mean() / a.mean()
            ratios.append(r)
            line.append(f"{s}dB {r:5.1f}% ({d.mean():+.3f}+-{ci:.3f}{'*' if abs(d.mean()) > ci else ''})")
        print(f"{k:3s} {tag:26s} " + " | ".join(line))
        lo, hi = round(min(ratios)), round(max(ratios))
        cells[k].append(f"${lo}$--${hi}$" if lo != hi else f"${lo}$")

print("\nLaTeX cells (unicast, multicast, multicommodity, convergecast, many-to-many):")
print("MANET-GNN & " + " & ".join(cells["gnn"]) + r" \\")
print(r"\ac{ffn} \ref{itm:ffn} & " + " & ".join(cells["ffn"]) + r" \\")
