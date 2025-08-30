import os, tempfile
from datetime import datetime
import torch

def atomic_save(obj, final_path):
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=os.path.dirname(final_path)) as tmp:
        torch.save(obj, tmp.name)
        tmp_path = tmp.name
    os.replace(tmp_path, final_path)

def format_best_fname(prefix, L, best_val):
    ts = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    return f"{prefix}_L{L}_val{best_val:.3f}_{ts}.pth"

def save_best_ckpt(*, model, epoch, best_val, cfg_path, ckpt_dir, prefix,
                   include_training_state, optimizer=None, scheduler=None,
                   scaler=None):
    fname = format_best_fname(prefix, model.num_layers, best_val)
    fpath = os.path.join(ckpt_dir, fname)
    payload = {
        "epoch": int(epoch + 1),
        "val_best": float(best_val),
        "model_state_dict": model.state_dict(),
        "config_text": cfg_path,
        "model_tag": {"name": prefix, "L": model.num_layers, "B": model.B},
    }
    if include_training_state:
        if optimizer is not None: payload["optimizer_state"] = optimizer.state_dict()
        if scheduler is not None: payload["scheduler_state"] = scheduler.state_dict()
        if scaler is not None:    payload["scaler_state"]    = scaler.state_dict()
    atomic_save(payload, fpath)
    return fpath
