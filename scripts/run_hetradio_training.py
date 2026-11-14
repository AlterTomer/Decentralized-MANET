"""
Train MANET-GNN and the FFN on heterogeneous radios.

Runs the GNN trainings (multicommodity, convergecast, many-to-many) and then the FFN trainings, one after
the other, with the configs written by scripts/make_hetradio_retrain_configs.py.

    python scripts/run_hetradio_training.py           # GNN, then FFN
    python scripts/run_hetradio_training.py gnn       # GNN only
    python scripts/run_hetradio_training.py ffn       # FFN only
    python scripts/run_hetradio_training.py gnn convergecast multiunicast   # only these frameworks

Afterwards: python scripts/fill_hetradio_paths.py
"""
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CFG = Path(os.path.join(os.environ.get("MANET_DATA_ROOT", r"C:\Users\alter\OneDrive\Desktop\PhD\Decentralized MANET"), r"Config Files"))
JOBS = {
    "gnn": (REPO / "scripts" / "Decentralized_MANET_Simulation.py",
            [CFG / "Multicommodity" / "R2" / "ChainedGNN_multicommodity_hetradio_R2.ini",
             CFG / "Convergencecast" / "R2" / "ChainedGNN_convergecast_hetradio_R2.ini",
             CFG / "Multiunicast" / "R2" / "ChainedGNN_multiunicast_hetradio_R2.ini"]),
    "ffn": (REPO / "MANET_FFN" / "main.py",
            [CFG / "FFN" / "Multicommodity" / "FFN_multicommodity_hetradio_R2.ini",
             CFG / "FFN" / "Convergencecast" / "FFN_convergecast_hetradio_R2.ini",
             CFG / "FFN" / "Multiunicast" / "FFN_multiunicast_hetradio_R2.ini"]),
}

if __name__ == "__main__":
    args = sys.argv[1:]
    which = [a for a in args if a in JOBS] or ["gnn", "ffn"]
    only = [a for a in args if a not in JOBS]        # optional framework filter, e.g. convergecast multiunicast
    env = dict(os.environ, PYTHONPATH=str(REPO))
    for name in which:
        script, cfgs = JOBS[name]
        if only:
            cfgs = [c for c in cfgs if any(o.lower() in c.name.lower() for o in only)]
        missing = [c for c in cfgs if not c.exists()]
        if missing:
            sys.exit(f"missing configs: {missing}")
        print(f"\n===== {name.upper()} training: {len(cfgs)} configs =====", flush=True)
        subprocess.run([sys.executable, str(script), *map(str, cfgs)], cwd=REPO, env=env, check=True)
    print("\ndone. Next: python scripts/fill_hetradio_paths.py")
