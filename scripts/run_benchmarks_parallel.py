"""
Run the R2 benchmark configs in parallel worker processes.

Each worker runs `scripts/Optimizer_vs_GNN.py --configs <its configs>` sequentially and logs to
logs/benchmark_worker_<k>.log. Configs whose inputs (channel file, model) do not exist yet are left
out. Configs are assigned to workers by estimated cost (longest-processing-time first), where the
cost is dominated by the centralized optimizer:
    networks x optimizer SNRs x optimizer runs per network x measured seconds per run.

Usage (from the repository root):
    python scripts/run_benchmarks_parallel.py --workers 4            # all R2 configs
    python scripts/run_benchmarks_parallel.py --workers 4 --only density
    python scripts/run_benchmarks_parallel.py --workers 4 --exclude density,multicast
    python scripts/run_benchmarks_parallel.py --dry-run               # print the assignment only
"""
import argparse
import os
import subprocess
import sys
import time
from configparser import ConfigParser
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CFG_DIR = Path(os.path.join(os.environ.get("MANET_DATA_ROOT", r"C:\Users\alter\OneDrive\Desktop\PhD\Decentralized MANET"), r"Config Files"))
# Measured wall-clock of one optimizer run (1000 iterations) per network on this machine [s].
SEC_PER_RUN = {"single": 2.7, "multicast": 3.0, "multi": 7.4, "converge": 7.4, "multiunicast": 8.1}
FULL_SEARCH_RUNS = 4 * 5          # inits x step sizes
REDUCED_RUNS = 3                  # greedy, widest path, MANET-GNN


def int_list(raw):
    raw = (raw or "").strip()
    return [int(float(x)) for x in raw.replace(" ", "").split(",") if x] if raw else None


def config_cost(path):
    p = ConfigParser()
    p.read_file(open(path))
    tp, fp = p["Train Parameters"], p["Files"]
    missing = [k for k in ("channel path", "model path") if not Path(fp.get(k, "").strip()).exists()]
    if missing:
        return None, missing
    snrs = int_list(tp.get("optimizer snr", "")) or int_list(tp.get("snr list", "")) or list(range(0, 51, 2))
    runs = FULL_SEARCH_RUNS if int(tp.get("optimizer full search", 0)) == 1 else REDUCED_RUNS
    return int(tp["num samples"]) * len(snrs) * runs * SEC_PER_RUN.get(tp["mode"], 8.0), []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", default="",
                    help="comma-separated substrings; keep only configs whose file names contain one of them")
    ap.add_argument("--exclude", default="",
                    help="comma-separated substrings; configs whose file names contain any of them are left out")
    ap.add_argument("--log-dir", default="logs", help="worker log folder, relative to the repository root")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfgs = sorted(CFG_DIR.glob("*/R2/comp_*_R2.ini"))
    only = [o.strip() for o in args.only.split(",") if o.strip()]
    if only:
        cfgs = [c for c in cfgs if any(o in c.name for o in only)]
    excl = [e.strip() for e in args.exclude.split(",") if e.strip()]
    if excl:
        cfgs = [c for c in cfgs if not any(e in c.name for e in excl)]
    jobs = []
    for c in cfgs:
        cost, missing = config_cost(c)
        if cost is None:
            print(f"[skip] {c.name}: missing {', '.join(missing)}")
        else:
            jobs.append((cost, c))
    if not jobs:
        print("nothing to run")
        return

    # Longest-processing-time-first assignment to the least-loaded worker.
    loads = [0.0] * args.workers
    groups = [[] for _ in range(args.workers)]
    for cost, c in sorted(jobs, key=lambda t: -t[0]):
        k = loads.index(min(loads))
        loads[k] += cost
        groups[k].append(c)
    for k, (g, ld) in enumerate(zip(groups, loads)):
        if g:
            print(f"worker {k}: ~{ld / 86400:.1f} days sequential estimate | " + ", ".join(c.name for c in g))
    print(f"total ~{sum(loads) / 86400:.1f} days sequential; wall-clock ~ max worker load "
          f"(~{max(loads) / 86400:.1f} days) if the GPU scales across {args.workers} processes")
    if args.dry_run:
        return

    log_dir = REPO / args.log_dir
    log_dir.mkdir(exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(REPO))
    procs = []
    for k, g in enumerate(groups):
        if not g:
            continue
        log = open(log_dir / f"benchmark_worker_{k}.log", "w", buffering=1)
        cmd = [sys.executable, str(REPO / "scripts" / "Optimizer_vs_GNN.py"), "--configs", *map(str, g)]
        procs.append((k, subprocess.Popen(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT), log))
        print(f"started worker {k} (pid {procs[-1][1].pid}) -> {log_dir / f'benchmark_worker_{k}.log'}")
    while procs:
        time.sleep(30)
        for item in list(procs):
            k, pr, log = item
            if pr.poll() is not None:
                log.close()
                print(f"worker {k} finished with exit code {pr.returncode}")
                procs.remove(item)


if __name__ == "__main__":
    main()
