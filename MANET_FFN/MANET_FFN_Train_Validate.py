import copy
from torch.utils.data import DataLoader, random_split
from MANET_FFN.MANET_FFN_Dataset import FFNDataset
from MANET_FFN.MANET_FFN_Utils import *
from MANET_FFN.model import FFNPowerAllocator



def run_epoch(
    model,
    loader,
    objective_fn,
    normalize_power,
    optimizer=None,
    device="cuda",
):
    """
    Run one training or validation epoch.

    This function mirrors the MANET-GNN validation policy more closely:
        - disconnected / invalid samples are skipped;
        - averages are computed over valid samples only;
        - training is unsupervised with loss = -rate.

    Returns
    -------
    avg_loss : float
        Mean loss over valid samples.

    avg_rate : float
        Mean achieved rate over valid samples.

    num_valid : int
        Number of samples that contributed to the epoch average.
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_rate = 0.0
    valid_count = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        h, adj, sigma, tx, rx = unpack_batch(batch)

        with torch.set_grad_enabled(is_train):
            p_raw = model(h).squeeze(0)
            p = normalize_power(p_raw, adj)

            rate = objective_fn(h, p, sigma, adj, tx, rx)

            # Allow objective_fn to signal invalid/disconnected samples.
            if rate is None:
                continue

            # Skip NaN/Inf samples.
            if not torch.isfinite(rate):
                continue

            loss = -rate

            # If the rate is detached, this usually means an invalid route returned
            # a constant zero. Skip it, matching validate_chained's continue behavior.
            if is_train:
                if not loss.requires_grad:
                    continue

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += float(loss.item())
        total_rate += float(rate.item())
        valid_count += 1

    if valid_count == 0:
        return 0.0, 0.0, 0

    avg_loss = total_loss / valid_count
    avg_rate = total_rate / valid_count

    return avg_loss, avg_rate, valid_count


def train_ffn(
    adj_list,
    links_list,
    tx_list,
    rx_list,
    sigma_list,
    B,
    objective_fn,
    normalize_power,
    problem="single",
    K=1,
    val_ratio=0.2,
    hidden_dim=512,
    num_layers=4,
    dropout=0.1,
    lr=1e-3,
    weight_decay=0.0,
    num_epochs=200,
    batch_size=1,
    seed=0,
    device=None,
    save_dir="ffn_results",
):
    """
    Train the FFN benchmark and save checkpoint/learning curves.

    Parameters
    ----------
    adj_list : list
        List of adjacency matrices [n, n].
    links_list : list
        List of complex CSI tensors [B, n, n].
    tx_list : list
        List of transmitter definitions.
    rx_list : list
        List of receiver definitions.
    sigma_list : list
        List of noise standard deviations.
    B : int
        Number of frequency bands.
    objective_fn : callable
        Objective/rate function used for unsupervised training.
        Expected signature: objective_fn(h, p, sigma, adj, tx, rx).
    normalize_power : callable
        External power projection/normalization function.
    problem : str
        Problem type: {"single", "multicast", "multi", "converge", "multiunicast"}.
    K : int
        Number of commodities/messages for multi-message cases.
    val_ratio : float
        Fraction of the dataset used for validation.
    hidden_dim : int
        Hidden width of the FFN.
    num_layers : int
        Number of linear layers in the FFN.
    dropout : float
        Dropout probability.
    lr : float
        Adam learning rate.
    weight_decay : float
        Adam weight decay.
    num_epochs : int
        Number of training epochs.
    batch_size : int
        Recommended value is 1 unless objective utilities are batch-aware.
    seed : int
        Random seed for reproducible train/validation split.
    device : str or torch.device, optional
        Computation device. If None, uses CUDA when available.
    save_dir : str
        Output directory for checkpoint and plots.

    Returns
    -------
    model : FFNPowerAllocator
        Trained model loaded with the best validation-rate checkpoint.
    history : dict
        Learning curves: train_loss, val_loss, train_rate, val_rate.

    Outputs
    -------
    Files written to save_dir:
        ffn_checkpoint.pt
        ffn_history.pt
        ffn_loss_curve.png
        ffn_rate_curve.png
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(seed)

    full_dataset = FFNDataset(
        adj_list=adj_list,
        links_list=links_list,
        tx_list=tx_list,
        rx_list=rx_list,
        sigma_list=sigma_list,
        B=B,
        problem=problem,
        K=K,
    )

    if batch_size != 1:
        print(
            "Warning: batch_size != 1. This is safe only if objective_fn and "
            "normalize_power support batched inputs."
        )

    n_total = len(full_dataset)
    n_val = int(val_ratio * n_total)
    n_train = n_total - n_val

    train_dataset, val_dataset = random_split(
        full_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )

    # FFN input size depends on n, so infer n from the first sample.
    sample0 = full_dataset[0]
    n_nodes = sample0["adj_matrix"].shape[0]

    model = FFNPowerAllocator(
        n_nodes=n_nodes,
        n_bands=B,
        K=K,
        problem=problem,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_rate": [],
        "val_rate": [],
    }

    best_val_rate = -float("inf")
    best_state = None

    for epoch in range(num_epochs):
        train_loss, train_rate, train_count = run_epoch(
            model=model,
            loader=train_loader,
            objective_fn=objective_fn,
            normalize_power=normalize_power,
            optimizer=optimizer,
            device=device,
        )

        val_loss, val_rate, val_count = run_epoch(
            model=model,
            loader=val_loader,
            objective_fn=objective_fn,
            normalize_power=normalize_power,
            optimizer=None,
            device=device,
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_rate"].append(train_rate)
        history["val_rate"].append(val_rate)

        if val_rate > best_val_rate:
            best_val_rate = val_rate
            best_state = copy.deepcopy(model.state_dict())

        print(
            f"Epoch {epoch + 1:03d}/{num_epochs} | "
            f"Train loss: {train_loss:.4f} | "
            f"Val loss: {val_loss:.4f} | "
            f"Train rate: {train_rate:.4f} | "
            f"Val rate: {val_rate:.4f} | "
            f"Valid train/val: {train_count}/{val_count}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    os.makedirs(save_dir, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "history": history,
            "config": {
                "B": B,
                "problem": problem,
                "K": K,
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "dropout": dropout,
                "lr": lr,
                "weight_decay": weight_decay,
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "seed": seed,
                "best_val_rate": best_val_rate,
            },
        },
        os.path.join(save_dir, "ffn_checkpoint.pt"),
    )

    save_learning_curves(history, save_dir)
    return model, history