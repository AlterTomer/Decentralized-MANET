import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.nn.utils import clip_grad_norm_
from utils.TensorUtils import normalize_power
from utils.MetricUtils import calc_sum_rate
from utils.PathUtils import find_all_paths, paths_to_tensor
from utils.EstimationUtils import lmmse_from_truth_masked, build_estimate_lookup
from Multicast.Objective import objective_multicast
from Multicast.SubGraphs import find_multicast_subgraphs
from Multicommodity.Objective import objective_multicommodity
#=======================================================================================================================
# Helpers
#=======================================================================================================================
def tau_linear(epoch: int, max_epochs: int, start: float = 2.0, end: float = 32.0) -> float:
    """Linearly increase τ from start → end."""
    t = min(max(epoch / max(1, max_epochs - 1), 0.0), 1.0)
    return start + t * (end - start)

def _compute_rates_per_layer(
    model,
    data,
    paths=None,
    subgraphs_per_band=None,
    paths_k=None,                 # NEW: list of length K of padded path tensors
    problem: str = "single",
    tau_min: float = 0.0,
    tau_max: float = 0.0,
):
    """
    Forward pass -> per-layer power normalization -> objective value per layer.

    Args:
        model:  GNN model; model.B is number of bands; model.K may be >1 for multi-commodity.
        data:   batch with at least: adj_matrix [n,n], links_matrix [B,n,n] (complex), sigma (float or [B]).
        paths:  (single) LongTensor [num_paths, max_len] (−1 padded).
        subgraphs_per_band: (multicast) list length B; each element is a list of subgraphs (edge lists or masks).
        paths_k: (multi) list length K; each item is LongTensor [num_paths_k, max_len] (−1 padded).
        problem: "single" | "multicast" | "multi".
        tau_min: soft-min temperature (edges).
        tau_max: soft-max temperature (paths/subgraphs). (Kept for completeness.)

    Returns:
        rates_per_layer: list[Tensor], scalars.
        p_list:          list[Tensor], each [B,n,n] (single/multicast) or [B,K,n,n] (multi) after normalization.
        z_list:          list[Tensor|None], only for problem="multi" (each [B,K,n,n]); else [].
    """
    if problem == "single":
        if paths is None:
            raise ValueError("paths must be provided for problem='single'.")
    elif problem == "multicast":
        if subgraphs_per_band is None:
            raise ValueError("subgraphs_per_band must be provided for problem='multicast'.")
    elif problem == "multi":
        if paths_k is None:
            raise ValueError("paths_k must be provided for problem='multi'.")
    else:
        raise ValueError("problem must be 'single', 'multicast', or 'multi'.")

    outputs = model(data)  # K==1 → list[[B,n,n]]; K>1 multi → list[(P[ B,K,n,n ], Z[ B,K,n,n ])]
    rates_per_layer, p_list, z_list = [], [], []

    adj = data.adj_matrix
    h   = data.links_matrix
    sig = data.sigma

    for out in outputs:
        if problem == "multi":
            # out = (P, Z) where P,Z are [B,K,n,n]
            P_raw, Z = out
            # normalize P per-commodity: for each k, normalize [B,n,n] with your existing routine
            P_norm = torch.empty_like(P_raw)
            for k in range(P_raw.shape[1]):
                P_norm[:, k] = normalize_power(P_raw[:, k], adj=adj.to(P_raw.device), eps=1e-12)
            rate = objective_multicommodity(
                h=h.to(P_norm.device),
                p=P_norm,
                z=Z,
                sigma=sig.to(P_norm.device),
                adj=adj.to(P_norm.device),
                paths_k=paths_k,
                tau_min=tau_min,
                tau_max=tau_max,
                reduce="mean",
                per_band=False,
            )
            p_list.append(P_norm)
            z_list.append(Z)
        else:
            # out = P [B,n,n]
            P_raw = out
            P_norm = normalize_power(P_raw, adj=adj.to(P_raw.device), eps=1e-12)

            if problem == "single":
                rate = calc_sum_rate(
                    h_arr=h.to(P_norm.device),
                    p_arr=P_norm,
                    sigma=sig.to(P_norm.device),
                    paths_tensor=paths,
                    B=model.B,
                    tau=0.0,
                    eps=1e-12,
                    per_band=False,
                )
            else:  # multicast
                rate = objective_multicast(
                    h=h.to(P_norm.device),
                    p=P_norm,
                    sigma=sig.to(P_norm.device),
                    adj=adj.to(P_norm.device),
                    subgraphs_per_band=subgraphs_per_band,
                    eps=1e-12,
                    tau_min=tau_min,
                    tau_max=tau_max,
                    per_band=False,
                )
            p_list.append(P_norm)

        rates_per_layer.append(rate if torch.isfinite(rate) else torch.tensor(float("-inf"), device=h.device))

    return rates_per_layer, p_list, z_list

#=======================================================================================================================
# ChainedNet
#=======================================================================================================================

def train_chained(
    model,
    loader,
    optimizer,
    epoch,
    *,
    batch_size: int = 1,
    mode: str = "single",          # "single" | "multicast" | "multi"
    device=None,
    mono_weight: float = 0.0,
    use_amp: bool = False,
    scaler=None,
    grad_clip=None,
    grad_accum_steps: int = 1,
    tau: float = 0.0,              # used as tau_min; hard max by default
    est_dataset=None,
):
    """
    Unified trainer for Problem 1 (single Tx→Rx path selection) and Problem 2 (multicast).
    Args:
    model: torch.nn.Module
         ChainedGNN model (produces a list of per-layer power tensors).
    loader: torch.utils.data.DataLoader
        Yields graph batches with fields:
          - adj_matrix: [n, n] (bool/0-1)
          - links_matrix: [B, n, n] (complex64/complex128)
          - sigma: float or [B] (noise std per band)
          - tx: int (source node index)
          - rx: int or List[int] (dest in "single"; receivers in "multicast")
          - optional sample_id: int (for estimated-CSI lookup)
    optimizer: torch.optim.Optimizer
        Optimizer over model parameters.
    epoch: int
        Current epoch index (for logging only).

    Keyword Args:
        mode: str = "single"
            Which objective to train:
              - "single":         - Uses paths via find_all_paths + paths_to_tensor
                                  - Objective: calc_sum_rate(...)
              - "multicast":      - Uses minimal subgraphs via find_multicast_subgraphs (band-agnostic)
                                - Objective: objective_multicast(...), i.e., max_S min_edge rate.

              - "multi":        K distinct messages (sum over commodities; each max-path then min-edge).

        device: torch.device | None
            If None, inferred from model parameters.
        mono_weight: float = 0.0
            Weight of the monotonicity penalty across layers.
            Encourages rates[layer+1] ≥ rates[layer] + margin.
        use_amp: bool = False
            Enable PyTorch AMP autocast for the forward/backward pass.
        scaler: torch.cuda.amp.GradScaler | None
            Required if use_amp=True; ignored otherwise.
        grad_clip: float | None
            If set, clip gradient norm to this value before optimizer.step().
        grad_accum_steps: int = 1
            Gradient accumulation steps (loss is divided internally).
        tau: float = 0.0
        Temperature parameter:
          - "single": forwarded to calc_sum_rate (soft-min over edges if >0).
          - "multicast": used as tau_min (soft-min over edges in a subgraph).
            Max over subgraphs is hard when tau_max=0 in this trainer.
    est_dataset: Optional[dict | EstimatedCSIDataset] = None
        If provided, forward pass uses estimated CSI (H_hat) looked up by sample_id,
        while the loss is computed under TRUE CSI (H_true).
        - If dict: maps sample_id -> H_hat (tensor shaped [B, n, n], complex).
        - If EstimatedCSIDataset: will be converted to a fast lookup via build_estimate_lookup().


    """
    assert mode in {"single", "multicast", "multi"}
    device = next(model.parameters()).device if device is None else device
    model.train()
    optimizer.zero_grad(set_to_none=True)

    est_lookup = None
    if est_dataset is not None:
        est_lookup = est_dataset if isinstance(est_dataset, dict) else build_estimate_lookup(est_dataset)

    total_loss_val, num_batches = 0.0, 0

    for step, data in enumerate(loader):
        data = data.to(device)
        if batch_size == 1:
            if isinstance(data.rx, (list, tuple)) and len(data.rx) == 1 and isinstance(data.rx[0], (list, tuple)):
                data.rx = data.rx[0]

        # ----- Build graph structures -----
        if mode == "single":
            paths_list = find_all_paths(data.adj_matrix, data.tx, data.rx)
            if len(paths_list) == 0:
                continue
            paths = paths_to_tensor(paths_list, device)
            subgraphs_per_band, paths_k = None, None

        elif mode == "multicast":
            graph_list = find_multicast_subgraphs(data.adj_matrix, data.tx, data.rx)
            if len(graph_list) == 0:
                continue
            subgraphs_per_band = [list(graph_list) for _ in range(model.B)]
            paths, paths_k = None, None

        else:  # mode == "multi"
            # data.rx is a list/1D tensor of K receivers; build per-commodity paths
            rx = getattr(data, "rx", [])
            if isinstance(rx, torch.Tensor):
                rx = rx.view(-1).tolist()
            paths_k = []
            for r in rx:
                plist = find_all_paths(data.adj_matrix, data.tx, int(r))
                if len(plist) == 0:
                    # If any commodity has no path, skip this sample
                    paths_k = None
                    break
                paths_k.append(paths_to_tensor(plist, device))
            if paths_k is None:
                continue
            subgraphs_per_band, paths = None, None

        # Save TRUE CSI
        H_true = data.links_matrix
        using_estimate = False

        try:
            # Optional: swap in estimated CSI for forward
            if est_lookup is not None:
                sid = int(getattr(data, "sample_id", step))
                H_hat = est_lookup.get(sid, None)
                if H_hat is not None:
                    data.links_matrix = H_hat.to(device)
                    using_estimate = True

            with autocast(device_type=device.type, enabled=use_amp):
                # Forward to get per-layer P (and Z for multi) under current CSI
                if mode == "single":
                    _, p_list, _ = _compute_rates_per_layer(
                        model, data,
                        paths=paths,
                        problem="single",
                        tau_min=0.0, tau_max=0.0
                    )
                elif mode == "multicast":
                    _, p_list, _ = _compute_rates_per_layer(
                        model, data,
                        subgraphs_per_band=subgraphs_per_band,
                        problem="multicast",
                        tau_min=tau, tau_max=0.0
                    )
                else:  # multi
                    _, p_list, z_list = _compute_rates_per_layer(
                        model, data,
                        paths_k=paths_k,
                        problem="multi",
                        tau_min=tau, tau_max=0.0
                    )

                # Restore TRUE CSI for loss
                if using_estimate:
                    data.links_matrix = H_true

                # Compute per-layer TRUE-CSI objectives
                rates_true_list = []
                for idx, P in enumerate(p_list):
                    if mode == "single":
                        r = calc_sum_rate(
                            h_arr=H_true, p_arr=P, sigma=data.sigma,
                            paths_tensor=paths, B=model.B, tau=tau
                        )
                    elif mode == "multicast":
                        r = objective_multicast(
                            h=H_true, p=P, sigma=data.sigma, adj=data.adj_matrix,
                            subgraphs_per_band=subgraphs_per_band,
                            eps=1e-12, tau_min=tau, tau_max=0.0, per_band=False
                        )
                    else:  # multi
                        Z = z_list[idx]
                        r = objective_multicommodity(
                            h=H_true, p=P, z=Z, sigma=data.sigma, adj=data.adj_matrix,
                            paths_k=paths_k, tau_min=tau, tau_max=0.0,
                            reduce="sum", per_band=False
                        )
                    rates_true_list.append(r)

                rates_true = torch.stack(rates_true_list)  # [L]
                rate_last_true = rates_true[-1]
                loss_unsup = -rate_last_true

                if mono_weight > 0.0 and rates_true.numel() >= 2:
                    margin = 0.05
                    deltas = rates_true[1:] - rates_true[:-1]
                    shortfall = F.relu(margin - deltas)
                    penalty = mono_weight * shortfall.mean()
                else:
                    penalty = torch.tensor(0.0, device=device)

                loss = (loss_unsup + penalty) / grad_accum_steps

        finally:
            data.links_matrix = H_true

        # Backward/step
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

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

    avg_loss = total_loss_val / max(num_batches, 1)
    stats = {"loss": avg_loss, "lr": optimizer.param_groups[0]["lr"]}
    print(f"[E{epoch:02d}][{mode}] loss={stats['loss']:.6f} | lr={stats['lr']:.2e}")
    return stats


@torch.no_grad()
def validate_chained(
    model,
    loader,
    *,
    batch_size: int = 1,
    mode: str = "single",          # "single" (Tx→Rx paths) or "multicast" (shared message over subgraphs)
    device=None,
    est_dataset=None,              # Optional EstimatedCSIDataset or dict(sample_id -> H_hat)
    tau: float = 0.0,              # soft-min temperature (used in both modes the same way you did before)
    verbose: bool = False,
):
    """
    Validate a ChainedGNN on either:
      - single:     best-path single Tx→Rx.
      - multicast:  shared message, max over subgraphs then min edge.
      - multi:      K distinct messages; sum over commodities of max-path min-edge rates.

    Evaluation policy:
      - If `est_dataset` is provided:
          * Forward pass uses estimated CSI (H_hat) to predict powers.
          * Metric is computed under TRUE CSI (H_true).
      - If `est_dataset` is None:
          * Forward and metric both use TRUE CSI.
      - Metric per sample is the **best over layers** (to match your original validation choice).
      - Final report is the average of per-sample best-layer rates.

    Args:
        model (torch.nn.Module):
            ChainedGNN; model.B is number of bands. Forward returns a list of per-layer power tensors [B, n, n].
        loader (torch.utils.data.DataLoader):
            Yields graph batches with fields:
              - adj_matrix: [n, n] (bool/0-1)
              - links_matrix: [B, n, n] (complex)
              - sigma: float or [B] tensor
              - tx: int (source node)
              - rx: int (dest in "single") OR list[int] (receivers in "multicast")
              - optional sample_id: int (for estimated-CSI lookup)
        mode (str, optional):
            "single" → enumerate all paths and use calc_sum_rate(...).
            "multicast" → enumerate minimal subgraphs and use objective_multicast(...) - K receivers, one shared message.
            "multi" → enumerate all paths and use objective_multicommodity(...) - K receivers, K different messages.
        device (torch.device | None, optional):
            If None, inferred from model parameters.
        est_dataset (dict | EstimatedCSIDataset | None, optional):
            If provided, maps sample_id → H_hat (or a dataset wrapper converted via build_estimate_lookup()).
        tau (float, optional):
            Soft-min temperature. For "single", forwarded to calc_sum_rate (as before).
            For "multicast", used as tau_min over edges (outer max over subgraphs kept hard here).
        verbose (bool, optional):
            If True, prints the averaged metric.

    Returns:
        dict:
          {
            "best_rate": float,   # average over samples of the best per-layer TRUE-CSI rate
          }
    """
    assert mode in {"single", "multicast", "multi"}
    device = next(model.parameters()).device if device is None else device
    model.eval()

    est_lookup = None
    if est_dataset is not None:
        est_lookup = est_dataset if isinstance(est_dataset, dict) else build_estimate_lookup(est_dataset)

    total_best, count = 0.0, 0

    for step, data in enumerate(loader):
        data = data.to(device)
        if batch_size == 1:
            if isinstance(data.rx, (list, tuple)) and len(data.rx) == 1 and isinstance(data.rx[0], (list, tuple)):
                data.rx = data.rx[0]

        # ----- Build graph structures -----
        if mode == "single":
            paths_list = find_all_paths(data.adj_matrix, data.tx, data.rx)
            if len(paths_list) == 0:
                continue
            paths = paths_to_tensor(paths_list, device)
            subgraphs_per_band, paths_k = None, None

        elif mode == "multicast":
            graph_list = find_multicast_subgraphs(data.adj_matrix, data.tx, data.rx)
            if len(graph_list) == 0:
                continue
            subgraphs_per_band = [list(graph_list) for _ in range(model.B)]
            paths, paths_k = None, None

        else:  # mode == "multi"
            rx = getattr(data, "rx", [])
            if isinstance(rx, torch.Tensor):
                rx = rx.view(-1).tolist()
            paths_k = []
            for r in rx:
                plist = find_all_paths(data.adj_matrix, data.tx, int(r))
                if len(plist) == 0:
                    paths_k = None
                    break
                paths_k.append(paths_to_tensor(plist, device))
            if paths_k is None:
                continue
            subgraphs_per_band, paths = None, None

        # TRUE CSI
        H_true = data.links_matrix
        using_estimate = False

        try:
            if est_lookup is not None:
                sid = int(getattr(data, "sample_id", step))
                H_hat = est_lookup.get(sid, None)
                if H_hat is not None:
                    data.links_matrix = H_hat.to(device)
                    using_estimate = True

            with autocast(device_type=device.type, enabled=False):
                if mode == "single":
                    _, p_list, _ = _compute_rates_per_layer(
                        model, data,
                        paths=paths, problem="single",
                        tau_min=0.0, tau_max=0.0
                    )
                elif mode == "multicast":
                    _, p_list, _ = _compute_rates_per_layer(
                        model, data,
                        subgraphs_per_band=subgraphs_per_band, problem="multicast",
                        tau_min=0.0, tau_max=0.0
                    )
                else:  # multi
                    _, p_list, z_list = _compute_rates_per_layer(
                        model, data,
                        paths_k=paths_k, problem="multi",
                        tau_min=0.0, tau_max=0.0
                    )

            if using_estimate:
                data.links_matrix = H_true

            rates_true_list = []
            for idx, P in enumerate(p_list):
                if mode == "single":
                    r = calc_sum_rate(
                        h_arr=H_true, p_arr=P, sigma=data.sigma,
                        paths_tensor=paths, B=model.B, tau=tau
                    )
                elif mode == "multicast":
                    r = objective_multicast(
                        h=H_true, p=P, sigma=data.sigma, adj=data.adj_matrix,
                        subgraphs_per_band=subgraphs_per_band,
                        eps=1e-12, tau_min=0.0, tau_max=0.0, per_band=False
                    )
                else:  # multi
                    Z = z_list[idx]
                    r = objective_multicommodity(
                        h=H_true, p=P, z=Z, sigma=data.sigma, adj=data.adj_matrix,
                        paths_k=paths_k, tau_min=0.0, tau_max=0.0,
                        reduce="mean", per_band=False
                    )
                rates_true_list.append(r)

            finite_vals = [float(r.item()) if torch.isfinite(r) else float("-inf") for r in rates_true_list]
            if all(v == float("-inf") for v in finite_vals):
                continue
            best_rate_true = max(v for v in finite_vals if v != float("-inf"))
            total_best += best_rate_true
            count += 1

        finally:
            data.links_matrix = H_true

    avg_best = (total_best / count) if count > 0 else 0.0
    if verbose:
        print(f"[VAL][{mode}] best_rate={avg_best:.6f} (TRUE CSI)")

    return {"best_rate": avg_best}
