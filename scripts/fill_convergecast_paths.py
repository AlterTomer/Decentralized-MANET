"""
Point the R2 convergecast benchmark configs at the newly trained checkpoints.

After scripts/Decentralized_MANET_Simulation.py and MANET_FFN/main.py have trained the convergecast
models, this finds the newest checkpoint of each run (by the training-config prefix) and writes it
into the 'model path' / 'ffn path' lines of every comp_convergecast_*_R2.ini:
  * comp_convergecast_EstimatedCSI_R2.ini -> estimated-CSI GNN and FFN
  * every other comp_convergecast_*_R2.ini (FullCSI, density, rgg, hub, hetradio) -> full-CSI models
Exits with code 1 (and changes nothing) if any of the four trained models is missing.
"""
import os
import re
import sys
from pathlib import Path

ROOT = Path(os.environ.get("MANET_DATA_ROOT", r"C:\Users\alter\OneDrive\Desktop\PhD\Decentralized MANET"))
CFG = ROOT / "Config Files" / "Convergencecast" / "R2"
GNN_DIR = ROOT / "Trained Models" / "Convergencecast"
FFN_DIR = ROOT / "Trained Models" / "FFN" / "Convergencecast"

GNN_PREFIX = {"full": "Quadriga_Convergecast_n_10_B_6_K_4_z_candidate_noisecond",
              "est": "Estimated_Quadriga_Convergecast_n_10_B_6_K_4_z_candidate_noisecond"}
FFN_PREFIX = {"full": "FFN_Quadriga_Convergecast_n_10_B_6_K_4",
              "est": "Estimated_FFN_Quadriga_Convergecast_n_10_B_6_K_4"}


def newest_gnn(prefix):
    # Checkpoints are named <prefix>_L<L>_val<v>_<timestamp>.pth; "_L" keeps the full-CSI prefix
    # from matching the estimated-CSI files (which only contain it after "Estimated_").
    c = [p for p in GNN_DIR.glob("*.pth") if p.name.startswith(prefix + "_L")] if GNN_DIR.exists() else []
    return max(c, key=lambda p: p.stat().st_mtime) if c else None


def ffn_dir(prefix):
    d = FFN_DIR / f"{prefix}_ffn"
    return d if d.is_dir() and any(d.iterdir()) else None


paths = {k: (newest_gnn(GNN_PREFIX[k]), ffn_dir(FFN_PREFIX[k])) for k in ("full", "est")}
missing = [f"{k} {w}" for k, pair in paths.items() for w, p in zip(("GNN", "FFN"), pair) if p is None]
if missing:
    print("missing trained models: " + ", ".join(missing) + " -- configs left unchanged")
    sys.exit(1)

for cfg in sorted(CFG.glob("comp_convergecast_*_R2.ini")):
    key = "est" if "EstimatedCSI" in cfg.name else "full"
    gnn, ffn = paths[key]
    text = cfg.read_text(encoding="utf-8")
    new, n1 = re.subn(r"(?m)^model path:.*$", lambda _: f"model path: {gnn}", text)
    new, n2 = re.subn(r"(?m)^ffn path:.*$", lambda _: f"ffn path: {ffn}", new)
    if n1 != 1 or n2 != 1:
        print(f"{cfg.name}: expected one 'model path' and one 'ffn path' line, found {n1}/{n2}")
        sys.exit(1)
    cfg.write_text(new, encoding="utf-8")
    print(f"{cfg.name}: model -> {gnn.name} | ffn -> {ffn.name}")
