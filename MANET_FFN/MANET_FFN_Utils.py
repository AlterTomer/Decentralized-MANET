import os
import torch
import matplotlib.pyplot as plt


def move_batch_to_device(batch, device):
    """
    Move tensor-valued batch fields to a target device.

    Parameters
    ----------
    batch : dict
        Batch returned by DataLoader over FFNDataset.
    device : str or torch.device
        Target device, e.g. "cuda" or "cpu".

    Returns
    -------
    dict
        Same batch dictionary, with tensor fields moved to device.
    """
    batch["links_matrix"] = batch["links_matrix"].to(device)
    batch["adj_matrix"] = batch["adj_matrix"].to(device)
    batch["sigma"] = batch["sigma"].to(device)
    return batch


def unpack_batch(batch):
    """
    Unpack a DataLoader batch.

    This helper assumes batch_size=1, which is recommended unless the objective
    function and normalize_power are explicitly batch-aware.

    Parameters
    ----------
    batch : dict
        Batch from DataLoader.

    Returns
    -------
    tuple
        h : torch.Tensor
            CSI tensor [B, n, n].
        adj : torch.Tensor
            Adjacency matrix [n, n].
        sigma : torch.Tensor
            Noise standard deviation.
        tx : int or list[int]
            Transmitter definition.
        rx : int or list[int]
            Receiver definition.
    """
    h = batch["links_matrix"].squeeze(0)
    adj = batch["adj_matrix"].squeeze(0)
    sigma = batch["sigma"].squeeze(0)
    tx = batch["tx"]
    rx = batch["rx"]
    return h, adj, sigma, tx, rx


def save_learning_curves(history, save_dir):
    """
    Save learning-curve data and plots.

    Parameters
    ----------
    history : dict
        Dictionary containing train_loss, val_loss, train_rate, and val_rate.
    save_dir : str
        Directory where the history file and PNG plots are saved.

    Outputs
    -------
    Files written to save_dir:
        ffn_history.pt
        ffn_loss_curve.png
        ffn_rate_curve.png
    """
    os.makedirs(save_dir, exist_ok=True)
    torch.save(history, os.path.join(save_dir, "ffn_history.pt"))

    plt.figure()
    plt.plot(history["train_loss"], label="Train loss")
    plt.plot(history["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "ffn_loss_curve.png"), dpi=300)
    plt.close()

    plt.figure()
    plt.plot(history["train_rate"], label="Train rate")
    plt.plot(history["val_rate"], label="Validation rate")
    plt.xlabel("Epoch")
    plt.ylabel("Rate")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "ffn_rate_curve.png"), dpi=300)
    plt.close()
