r"""
Install the paper's configs (config/paper/) into a data root and create the output folders.

Every config in config/paper/ refers to its inputs and outputs as <DATA_ROOT>\<subfolder>\...; this
copies the tree to <data root>/Config Files/, replacing <DATA_ROOT> by the data root (and, on Linux or
macOS, backslashes in those paths by '/'), and creates every checkpoint/figure/results folder the
configs write to. The scripts find the configs through the MANET_DATA_ROOT environment variable.

Usage (from the repository root):
    python scripts/install_paper_configs.py <data root>            # refuses to overwrite existing configs
    python scripts/install_paper_configs.py <data root> --force    # overwrite
Then:
    set MANET_DATA_ROOT=<data root>          (Windows cmd)
    $env:MANET_DATA_ROOT = "<data root>"     (PowerShell)
    export MANET_DATA_ROOT=<data root>       (bash)
"""
import argparse
import os
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "config" / "paper"
PLACEHOLDER = "<DATA_ROOT>"
DIR_KEYS = ("ckpt dir", "figs dir", "out dir")         # folders written by training / the GNN studies
FILE_KEYS = ("fig path", "fig data path")              # files written by the benchmark


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_root", type=Path)
    ap.add_argument("--force", action="store_true", help="overwrite existing configs")
    args = ap.parse_args()
    root = args.data_root.resolve()
    dst_root = root / "Config Files"

    written, skipped, folders = 0, 0, set()
    for src in sorted(SRC.rglob("*.ini")):
        dst = dst_root / src.relative_to(SRC)
        if dst.exists() and not args.force:
            skipped += 1
            continue
        lines = []
        for line in src.read_text(encoding="utf-8").split("\n"):
            if PLACEHOLDER in line:
                key, _, value = line.partition(":")
                value = value.strip().replace(PLACEHOLDER, str(root))
                if os.sep == "/":
                    value = value.replace("\\", "/")
                line = f"{key}: {value}"
                k = key.strip().lower()
                if k in DIR_KEYS:
                    folders.add(Path(value))
                elif k in FILE_KEYS:
                    folders.add(Path(value).parent)
            lines.append(line)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text("\n".join(lines), encoding="utf-8")
        written += 1

    folders.add(root / "Quadriga channels")
    for f in sorted(folders):
        f.mkdir(parents=True, exist_ok=True)
    print(f"configs: {written} written, {skipped} kept (use --force to overwrite) -> {dst_root}")
    print(f"created/verified {len(folders)} folders under {root}")
    print(f'next: set MANET_DATA_ROOT="{root}"')


if __name__ == "__main__":
    main()
