"""
Point the comp_<tag>_FullCSI_hetradio_retrained_R2.ini benchmark configs at the GNN and FFN checkpoints
trained on heterogeneous radios. For each framework, takes the newest
GNN checkpoint whose name starts with its HetRadio prefix + "_L", and the FFN folder <prefix>_ffn.
A framework whose models are missing is left unchanged (and reported); exits with code 1 if any is.
"""
import os
import re
import sys
from pathlib import Path

ROOT = Path(os.environ.get("MANET_DATA_ROOT", r"C:\Users\alter\OneDrive\Desktop\PhD\Decentralized MANET"))
CFG = ROOT / "Config Files"
TM = ROOT / "Trained Models"
FW = [  # config folder, tag, GNN checkpoint dir, GNN prefix, FFN prefix
    ("Multicommodity", "multicommodity", TM / "Multicommodity",
     "HetRadio_Quadriga_Multicommodity_n_10_B_6_K_4_z_candidate_noisecond", "HetRadio_FFN_Quadriga_Multicommodity_n_10_B_6_K_4"),
    ("Convergencecast", "convergecast", TM / "Convergencecast",
     "HetRadio_Quadriga_Convergecast_n_10_B_6_K_4_z_candidate_noisecond", "HetRadio_FFN_Quadriga_Convergecast_n_10_B_6_K_4"),
    ("Multiunicast", "multiunicast", TM / "multiunicast",
     "HetRadio_Quadriga_Multiunicast_n_10_B_6_K_4_z_candidate_noisecond", "HetRadio_FFN_Quadriga_Multiunicast_n_10_B_6_K_4"),
]

# Pinned GNN checkpoints (trained on heterogeneous radios without the dead-route mask; the mask is applied at
# inference, CandidateRoutingHead.mask_dead_routes). The later mask-trained runs (2026-11-12) were worse on
# the heterogeneous-radio test set, so the newest checkpoint is NOT used when a pin matches.
PIN = {"multicommodity": "val0.162_2026-11-11_08-59-31",
       "convergecast": "val0.259_2026-11-11_14-35-05",
       "multiunicast": "val0.286_2026-11-11_16-24-25"}

only = set(sys.argv[1:])          # optional framework tags, e.g. "multiunicast"; default: all three
failed = False
for folder, tag, gnn_dir, gnn_prefix, ffn_prefix in FW:
    if only and tag not in only:
        continue
    ckpts = [p for p in gnn_dir.glob("*.pth") if p.name.startswith(gnn_prefix + "_L")] if gnn_dir.exists() else []
    pinned = [p for p in ckpts if PIN.get(tag, "\0") in p.name]
    gnn = pinned[0] if pinned else (max(ckpts, key=lambda p: p.stat().st_mtime) if ckpts else None)
    ffn = TM / "FFN" / folder / f"{ffn_prefix}_ffn"
    ffn = ffn if ffn.is_dir() and any(ffn.iterdir()) else None
    cfg = CFG / folder / "R2" / f"comp_{tag}_FullCSI_hetradio_retrained_R2.ini"
    if gnn is None or ffn is None:
        print(f"{cfg.name}: missing {'GNN' if gnn is None else ''} {'FFN' if ffn is None else ''} -- left unchanged")
        failed = True
        continue
    text = cfg.read_text(encoding="utf-8")
    text, n1 = re.subn(r"(?m)^model path:.*$", lambda _: f"model path: {gnn}", text)
    text, n2 = re.subn(r"(?m)^ffn path:.*$", lambda _: f"ffn path: {ffn}", text)
    if n1 != 1 or n2 != 1:
        print(f"{cfg.name}: expected one 'model path' and one 'ffn path' line, found {n1}/{n2}")
        failed = True
        continue
    cfg.write_text(text, encoding="utf-8")
    print(f"{cfg.name}: model -> {gnn.name} | ffn -> {ffn.name}")
sys.exit(1 if failed else 0)
