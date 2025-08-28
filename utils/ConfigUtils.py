import argparse
from pathlib import Path
from configparser import ConfigParser


def parse_args():
    p = argparse.ArgumentParser(
        description="Train model from an .ini config"
    )
    p.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to .ini file (e.g., config/train.ini)"
    )
    return p.parse_args()

def load_ini_config(cfg_path: Path) -> ConfigParser:
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    parser = ConfigParser()
    # .read() handles BOM/encoding and closes file; no need to open()
    read_ok = parser.read(cfg_path, encoding="utf-8")
    if not read_ok:
        raise RuntimeError(f"Failed to read config: {cfg_path}")
    return parser