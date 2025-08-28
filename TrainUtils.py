import datetime
import os
import math
import tempfile
import torch

def cosine_warm_restart_lambda(warmup_epochs, total_epochs, eta_min, base_lr):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # start at 20% of base_lr instead of 0
            return 0.2 + 0.8 * (epoch / warmup_epochs)
        else:
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return eta_min / base_lr + 0.5 * (1 - eta_min / base_lr) * (1 + math.cos(math.pi * progress))
    return lr_lambda

def save_best_ckpt(model, epoch, val_best, cfg, ckpt_dir, prefix="ChainedGNN"):
    # cfg: dict with static run config (B, L, seed, data gen args, etc.)
    ts = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    fname = f"{prefix}_L{cfg['L']}_best_{val_best:.6f}_{ts}.pth"
    fpath = os.path.join(ckpt_dir, fname)

    # atomic write
    with tempfile.NamedTemporaryFile(delete=False, dir=ckpt_dir) as tmp:
        torch.save({
            "epoch": epoch + 1,
            "val_best": float(val_best),
            "model_state_dict": model.state_dict(),
            "cfg": cfg,   # small dict; not the dataset object
        }, tmp.name)
        tmp_path = tmp.name

    os.replace(tmp_path, fpath)
    return fpath