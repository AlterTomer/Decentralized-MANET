"""
Same-model CSI comparison.

The full-CSI and estimated-CSI curves of MANET-GNN and the FFN come from separately trained models, so
their ratio mixes the effect of the estimation errors with the difference between two trainings. To
isolate the former, the ESTIMATED-CSI models are also run with the TRUE channels as input. For each
comp_<tag>_EstimatedCSI_R2.ini this writes comp_<tag>_EstModelTrueCSI_R2.ini, identical except:
  * LMMSE estimation: 0      -> every method decides on the true channels (scored on them as well)
  * optimizer snr: 999       -> the centralized optimizer is skipped (not needed, and by far the most costly)
  * collect diagnostics: 0
  * fig / fig data paths: EstimatedCSI_... -> EstModelTrueCSI_...
The model and FFN paths stay those of the estimated-CSI models. scripts/same_model_csi_ratio.py then
compares each pickle with the corresponding EstimatedCSI_..._R2.pkl. Refuses to overwrite.

Run all five with (from the repository root, PYTHONPATH=.):
    python scripts/run_benchmarks_parallel.py --workers 5 --log-dir logs_samemodel --only EstModelTrueCSI
"""
import os
import re
from pathlib import Path

CFG = Path(os.path.join(os.environ.get("MANET_DATA_ROOT", r"C:\Users\alter\OneDrive\Desktop\PhD\Decentralized MANET"), r"Config Files"))
FW = [("Single Tx-Rx", "unicast"), ("Multicast", "multicast"), ("Multicommodity", "multicommodity"),
      ("Convergencecast", "convergecast"), ("Multiunicast", "multiunicast")]


def setkey(text, key, value):
    new, n = re.subn(rf"(?m)^{re.escape(key)}\s*:.*$", lambda _: f"{key}: {value}", text)
    if n == 0:   # key absent: add it to [Train Parameters]
        new = text.replace("[Train Parameters]\n", f"[Train Parameters]\n{key}: {value}\n", 1)
    elif n > 1:
        raise ValueError(f"'{key}' appears {n} times")
    return new


for folder, tag in FW:
    src = CFG / folder / "R2" / f"comp_{tag}_EstimatedCSI_R2.ini"
    dst = CFG / folder / "R2" / f"comp_{tag}_EstModelTrueCSI_R2.ini"
    if dst.exists():
        raise FileExistsError(dst)
    t = src.read_text(encoding="utf-8")
    t = setkey(t, "LMMSE estimation", "0")
    t = setkey(t, "optimizer snr", "999")
    t = setkey(t, "collect diagnostics", "0")
    for key in ("fig path", "fig data path"):
        old = re.search(rf"(?m)^{key}\s*:\s*(.*)$", t).group(1).strip()
        assert "EstimatedCSI_" in old, old
        t = setkey(t, key, old.replace("EstimatedCSI_", "EstModelTrueCSI_"))
    dst.write_text(t, encoding="utf-8")
    print("wrote", dst.name)
