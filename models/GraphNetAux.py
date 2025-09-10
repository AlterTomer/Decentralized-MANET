import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.nn.utils import clip_grad_norm_
from utils.TensorUtils import normalize_power
from utils.MetricUtils import calc_sum_rate, hybrid_power_loss
from utils.PathUtils import find_all_paths, paths_to_tensor
from utils.EstimationUtils import lmmse_from_truth_masked, build_estimate_lookup
#=======================================================================================================================
# ChainedNet
#=======================================================================================================================
def _compute_rates_per_layer(model, data, paths, tau=0):
    """
    Helper: forward pass + normalization + rate per layer.

    Returns:
        rates_per_layer (List[Tensor]): each shape [], scalar rate (positive, higher is better).
        p_list (List[Tensor]): normalized power tensors per layer, each [B, n, n].
    """
    outputs = model(data)  # list of [B, n, n]
    rates_per_layer, p_list = [], []
    for p_arr in outputs:
        p_arr = normalize_power(p_arr, adj=data.adj_matrix.to(p_arr.device), eps=1e-12)
        rate = calc_sum_rate(data.links_matrix.to(p_arr.device),
                             p_arr, data.sigma.to(p_arr.device),
                             paths, model.B, tau=tau)
        if not torch.isfinite(rate):
            # mark as invalid; caller will decide whether to skip this batch
            rates_per_layer.append(torch.tensor(float("-inf"), device=p_arr.device))
        else:
            rates_per_layer.append(rate)  # scalar
        p_list.append(p_arr)
    return rates_per_layer, p_list


def train_chained(
    model,
    loader,
    optimizer,
    epoch,
    *,
    warmup=0,
    device=None,
    mono_weight=0.0,
    use_amp=False,
    scaler=None,
    grad_clip=None,
    grad_accum_steps=1,
    tau=0.0,
    est_dataset=None,  # Optional EstimatedCSIDataset or dict(sample_id -> H_hat)
):
    """
    Train ChainedGNN for one epoch.

    If `est_dataset` is provided:
        - Forward uses LMMSE-estimated CSI (H_hat) to produce P.
        - Loss and penalties are computed under TRUE CSI (H_true).

    If `est_dataset` is None:
        - Forward and loss both use TRUE CSI (full-CSI training).

    Returns:
        dict with: {"loss": float, "lr": float}
    """
    device = next(model.parameters()).device if device is None else device
    model.train()
    optimizer.zero_grad(set_to_none=True)

    # If the user passed an EstimatedCSIDataset, also create a fast lookup map.
    est_lookup = None
    if est_dataset is not None:
        if isinstance(est_dataset, dict):
            est_lookup = est_dataset
        else:
            est_lookup = build_estimate_lookup(est_dataset)

    total_loss_val = 0.0
    num_batches = 0

    for step, data in enumerate(loader):
        data = data.to(device)

        # Build per-graph paths
        paths_list = find_all_paths(data.adj_matrix, data.tx, data.rx)
        if len(paths_list) == 0:
            continue
        paths = paths_to_tensor(paths_list, device)

        # Keep TRUE CSI safe
        H_true = data.links_matrix
        using_estimate = False

        try:
            # If we have precomputed estimates, fetch and swap CSI for forward
            if est_lookup is not None:
                sid = int(getattr(data, "sample_id", step))
                H_hat = est_lookup.get(sid, None)
                if H_hat is not None:
                    data.links_matrix = H_hat
                    using_estimate = True

            with autocast(device_type=device.type, enabled=use_amp):
                # Forward pass
                _, p_list = _compute_rates_per_layer(model, data, paths, tau=tau)

                # Restore TRUE CSI before computing the loss
                if using_estimate:
                    data.links_matrix = H_true

                # Compute per-layer rates under TRUE CSI
                rates_true_list = []
                for P in p_list:
                    r = calc_sum_rate(
                        h_arr=H_true,
                        p_arr=P,
                        sigma=data.sigma,
                        paths_tensor=paths,
                        B=model.B,
                        tau=tau
                    )
                    rates_true_list.append(r)

                rates_true = torch.stack(rates_true_list)  # [L]
                rate_last_true = rates_true[-1]

                # Unsupervised objective: maximize last-layer TRUE-CSI rate
                loss_unsup = -rate_last_true

                # Monotonicity penalty
                if mono_weight > 0.0 and rates_true.numel() >= 2:
                    margin = 0.05
                    deltas = rates_true[1:] - rates_true[:-1]
                    shortfall = F.relu(margin - deltas)
                    penalty = mono_weight * shortfall.mean()
                else:
                    penalty = torch.tensor(0.0, device=device)

                # Supervised warmup (if available)
                if epoch <= warmup and hasattr(data, "p_opt") and data.p_opt is not None:
                    Pc = data.p_opt.to(device)
                    loss_sup = hybrid_power_loss(
                        pred_P=p_list[-1],
                        target_P=Pc,
                        adj=data.adj_matrix,
                        alpha=1.0,
                        normalize_mse=False
                    ).to(device)
                    loss = loss_sup + penalty
                else:
                    loss = loss_unsup + penalty

                # Scale loss for grad accumulation
                loss = loss / grad_accum_steps

        finally:
            # Ensure TRUE CSI is always restored
            data.links_matrix = H_true

        # Backward
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Optimizer step on accumulation boundary
        if (step + 1) % grad_accum_steps == 0:
            if grad_clip is not None:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), grad_clip)

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

        total_loss_val += float(loss.detach().cpu()) * grad_accum_steps
        num_batches += 1

    avg_loss = (total_loss_val / num_batches) if num_batches else 0.0
    stats = {
        "loss": avg_loss,
        "lr": optimizer.param_groups[0]["lr"],
    }

    print(f"[E{epoch:02d}] loss={stats['loss']:.6f} | lr={stats['lr']:.2e}")
    return stats


@torch.no_grad()
def validate_chained(
    model,
    loader,
    *,
    device=None,
    est_dataset=None,
    verbose=False,
):
    """
    Validation with optional estimated CSI for the forward pass.

    If `est_dataset` is provided:
      - Forward uses estimated CSI (H_hat) to predict powers.
      - Evaluation metric (rate) is computed under TRUE CSI (H_true).

    If `est_dataset` is None:
      - Forward and evaluation both use TRUE CSI.

    Returns:
      {"rate": <average best-path rate under TRUE CSI>}
    """
    device = next(model.parameters()).device if device is None else device
    tau = 0
    model.eval()

    # Build a fast lookup dict if a dataset wrapper was provided
    est_lookup = None
    if est_dataset is not None:
        if isinstance(est_dataset, dict):
            est_lookup = est_dataset
        else:
            est_lookup = build_estimate_lookup(est_dataset)

    total_best = 0.0
    count = 0

    for step, data in enumerate(loader):
        data = data.to(device)

        # Build per-graph paths
        paths_list = find_all_paths(data.adj_matrix, data.tx, data.rx)
        if len(paths_list) == 0:
            continue
        paths = paths_to_tensor(paths_list, device)

        # Keep TRUE CSI safe
        H_true = data.links_matrix
        using_estimate = False

        try:
            # If we have precomputed estimates, fetch and swap CSI for forward
            if est_lookup is not None:
                sid = int(getattr(data, "sample_id", step))
                H_hat = est_lookup.get(sid, None)
                if H_hat is not None:
                    data.links_matrix = H_hat
                    using_estimate = True

            # Forward to get predicted powers (rates returned here are under the current CSI;
            # we'll recompute under TRUE CSI below).
            with autocast(device_type=device.type, enabled=False):  # AMP usually not needed for val, keep off unless desired
                _, p_list = _compute_rates_per_layer(model, data, paths, tau=tau)

            # Restore TRUE CSI and compute rates under it
            if using_estimate:
                data.links_matrix = H_true

            rates_true_list = []
            for P in p_list:
                r = calc_sum_rate(
                    h_arr=H_true,
                    p_arr=P,
                    sigma=data.sigma,
                    paths_tensor=paths,
                    B=model.B,
                    tau=tau
                )
                rates_true_list.append(r)

            # Skip entirely invalid samples
            finite_vals = [float(r.item()) if torch.isfinite(r) else float("-inf") for r in rates_true_list]
            if all(v == float("-inf") for v in finite_vals):
                continue

            best_rate_true = max(v for v in finite_vals if v != float("-inf"))
            total_best += best_rate_true
            count += 1

        finally:
            # Ensure TRUE CSI remains on the Data object
            data.links_matrix = H_true

    avg_best = (total_best / count) if count > 0 else 0.0
    if verbose:
        print(f"[VAL] rate={avg_best:.6f} (avg best-path, TRUE CSI)")

    return {"best_rate": avg_best}


def tau_linear(epoch: int, max_epochs: int, start: float = 2.0, end: float = 32.0) -> float:
    """Linearly increase τ from start → end."""
    t = min(max(epoch / max(1, max_epochs - 1), 0.0), 1.0)
    return start + t * (end - start)

