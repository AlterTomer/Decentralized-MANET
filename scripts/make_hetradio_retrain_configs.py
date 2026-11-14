"""
Configs for retraining MANET-GNN and the FFN on heterogeneous radios.

From the configs of the full-CSI models used in the R2 benchmarks, writes:
  * Config Files/<fw>/R2/ChainedGNN_<tag>_hetradio_R2.ini     (GNN training, F3/F4/F5)
  * Config Files/FFN/<fw>/FFN_<tag>_hetradio_R2.ini           (FFN training, F3/F4/F5)
  * Config Files/<fw>/R2/comp_<tag>_FullCSI_hetradio_retrained_R2.ini (benchmark on the hetero test set)
Only the channel file, the full-CSI setting, the checkpoint prefix, and the output paths change; all
hyperparameters are copied. The benchmark configs get placeholder model paths, which
scripts/fill_hetradio_paths.py fills after training (run_benchmarks_parallel.py skips them until then).
Refuses to overwrite existing files.
"""
import os
import re
from pathlib import Path

ROOT = Path(os.environ.get("MANET_DATA_ROOT", r"C:\Users\alter\OneDrive\Desktop\PhD\Decentralized MANET"))
CFG = ROOT / "Config Files"
TRAIN = ROOT / "Quadriga channels" / "Train_H_HETRADIO_seed_48312_n_10_B_6_samples_2000.pt"
FW = [  # folder, tag, GNN training config, FFN training config, GNN prefix, FFN prefix
    ("Multicommodity", "multicommodity", CFG / "Multicommodity" / "ChainedGNN_multicommodity.ini",
     CFG / "FFN" / "Multicommodity" / "FFN_multicommodity.ini",
     "HetRadio_Quadriga_Multicommodity_n_10_B_6_K_4_z_candidate_noisecond", "HetRadio_FFN_Quadriga_Multicommodity_n_10_B_6_K_4"),
    ("Convergencecast", "convergecast", CFG / "Convergencecast" / "R2" / "ChainedGNN_convergecast_FullCSI_R2.ini",
     CFG / "FFN" / "Convergencecast" / "FFN_convergecast_FullCSI_R2.ini",
     "HetRadio_Quadriga_Convergecast_n_10_B_6_K_4_z_candidate_noisecond", "HetRadio_FFN_Quadriga_Convergecast_n_10_B_6_K_4"),
    ("Multiunicast", "multiunicast", CFG / "Multiunicast" / "ChainedGNN_multiunicast.ini",
     CFG / "FFN" / "Multiunicast" / "FFN_multiunicast.ini",
     "HetRadio_Quadriga_Multiunicast_n_10_B_6_K_4_z_candidate_noisecond", "HetRadio_FFN_Quadriga_Multiunicast_n_10_B_6_K_4"),
]


def setkey(text, key, value):
    new, n = re.subn(rf"(?m)^{re.escape(key)}\s*:.*$", lambda _: f"{key}: {value}", text)
    if n != 1:
        raise ValueError(f"expected one '{key}' line, found {n}")
    return new


def write(path, text):
    if path.exists():
        raise FileExistsError(path)
    path.write_text(text, encoding="utf-8")
    print("wrote", path)


for folder, tag, gnn_src, ffn_src, gnn_prefix, ffn_prefix in FW:
    g = gnn_src.read_text(encoding="utf-8")
    g = setkey(g, "LMMSE estimation", "0")
    g = setkey(g, "channel path", str(TRAIN))
    g = setkey(g, "prefix", gnn_prefix)
    write(CFG / folder / "R2" / f"ChainedGNN_{tag}_hetradio_R2.ini", g)

    f = ffn_src.read_text(encoding="utf-8")
    f = setkey(f, "LMMSE estimation", "0")
    f = setkey(f, "channel_path", str(TRAIN))
    f = setkey(f, "prefix", ffn_prefix)
    write(CFG / "FFN" / folder / f"FFN_{tag}_hetradio_R2.ini", f)

    c = (CFG / folder / "R2" / f"comp_{tag}_FullCSI_hetradio_R2.ini").read_text(encoding="utf-8")
    c = setkey(c, "model path", "TO_BE_FILLED_AFTER_TRAINING")
    c = setkey(c, "ffn path", "TO_BE_FILLED_AFTER_TRAINING")
    for key in ("fig path", "fig data path"):
        old = re.search(rf"(?m)^{key}\s*:\s*(.*)$", c).group(1).strip()
        c = setkey(c, key, old.replace("_hetradio_R2", "_hetradio_retrained_R2"))
    write(CFG / folder / "R2" / f"comp_{tag}_FullCSI_hetradio_retrained_R2.ini", c)
