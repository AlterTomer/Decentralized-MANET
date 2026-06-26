import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from utils.PathUtils import find_all_paths, paths_to_tensor
from utils.TensorUtils import normalize_power, create_normalized_tensor, init_equal_power
from utils.MetricUtils import calc_sum_rate, objective_single_wrapper
from Multicast.SubGraphs import find_multicast_subgraphs
from Multicast.Objective import objective_multicast_wrapper, objective_multicast
from Multicommodity.Objective import  objective_multicommodity_wrapper, objective_multicommodity
from MANET_FFN.MANET_FFN_Utils import *
from MANET_FFN.main import select_ffn_objective

#========================================== Centralized Optimizer ======================================================
def classic_opt(
    num_iterations,
    optimizer,
    adj_mat,
    links_mat,
    p_arr,
    sigma_noise,
    objective_fn,
    objective_kwargs,
):
    """
    Runs centralized ADAM optimization for power allocation, with a generic objective.

    Args:
        num_iterations (int): Number of optimization steps.
        optimizer (torch.optim.Optimizer): ADAM/ADAMW optimizer.
        adj_mat (torch.Tensor): [n, n] adjacency matrix.
        links_mat (torch.Tensor): [B, n, n] channel matrices.
        p_arr (torch.nn.Parameter): Power allocation variable (B, n, n).
        sigma_noise (float): Noise power (σ²) or equivalent.
        objective_fn (callable): Function that computes the scalar objective given
                                 (links_mat, p_arr, sigma_noise, **objective_kwargs).
        objective_kwargs (dict): Extra arguments specific to the problem (paths,
                                 subgraphs, tx/rx lists, etc.).

    Returns:
        torch.Tensor: Optimized power allocation tensor.
    """
    a = 0
    with torch.no_grad():
        p_arr.copy_(normalize_power(p_arr, adj_mat))

    for i in range(num_iterations):
        optimizer.zero_grad()

        # Maximize objective => minimize negative objective
        loss = -objective_fn(
            links_mat=links_mat,
            P=p_arr,
            sigma_noise=sigma_noise,
            **objective_kwargs
        )

        try:
            loss.backward()
            optimizer.step()
        except RuntimeError:
            a = 1
            p0 = p_arr.detach().clone()
            p0 = normalize_power(p0, adj_mat)
            break

        with torch.no_grad():
            p_arr.copy_(normalize_power(p_arr, adj_mat))

    return p_arr.detach() if a == 0 else p0


def evaluate_centralized_adam(
    loader,
    B: int,
    noise_std=None,
    lr: float = 0.1,
    num_iterations: int = 30,
    problem: str = "single",  # "single", "multicast", "multi", "converge", or "multiunicast"
):
    """
    Evaluate centralized ADAM-based power allocation over a dataset for different
    problem types, using objective wrappers.

    This function constructs a centralized optimization baseline that:
      1. Initializes a learnable power tensor P (and Z for multi-message problems).
      2. Runs `classic_opt`, which repeatedly:
           - normalizes P (per-node power constraint),
           - evaluates the appropriate objective via a wrapper,
           - performs ADAM/AdamW updates.
      3. Returns the achieved objective values and the optimized P tensors.

    Args:
        loader:
            Iterable or DataLoader of graph data objects. Each `data` must have:
              - data.adj_matrix:    [n, n] adjacency (0/1 or bool).
              - data.links_matrix:  [B, n, n] complex channel matrix h.
              - data.sigma:         noise std (float or tensor).
              - data.tx:            transmitter node index or indices.
              - data.rx:            receiver node index or indices.
              - (optionally) data.B: number of bands; if absent, B argument is used.

        B (int):
            Number of frequency bands.

        noise_std (float, list, or None):
            - If float: same sigma for all graphs.
            - If list: per-graph sigma, aligned with loader order.
            - If None: use `data.sigma` for each graph.

        lr (float):
            Learning rate for the AdamW optimizer.

        num_iterations (int):
            Number of gradient-descent iterations in `classic_opt`.

        problem (str):
            One of:
              - "single"       : single Tx→Rx unicast.
              - "multicast"    : single Tx, K receivers, same message.
              - "multi"        : single Tx, K receivers, distinct messages.
              - "converge"     : K transmitters, single Rx, distinct messages.
              - "Multiunicast" : K Tx-Rx pairs, distinct messages.

    Returns:
        rate_results (list[float]):
            For each graph, the final objective value obtained by centralized optimization.

        P_opt_results (list[torch.Tensor]):
            For each graph, the optimized power tensor:
              - "single" / "multicast": [B, n, n]
              - "multi" / "converge" / "multiunicast": [B, K, n, n]
    """
    rate_results = []
    p_opt_results = []

    sigma_arr = None
    if isinstance(noise_std, list):
        sigma_arr = noise_std

    for k_batch, data in enumerate(loader):
        # Move data to appropriate device
        data = data.to(data.links_matrix.device)
        adj = data.adj_matrix            # [n, n]
        links = data.links_matrix        # [B, n, n]
        device = links.device
        n = adj.shape[0]

        # ----- noise -----
        if sigma_arr is not None:
            sigma = sigma_arr[k_batch]
        elif isinstance(noise_std, float):
            sigma = noise_std
        else:
            sigma = data.sigma

        # normalize tx / rx containers a bit
        tx_raw = data.tx
        rx_raw = data.rx

        if isinstance(tx_raw, torch.Tensor):
            tx_raw = tx_raw.view(-1).tolist() if tx_raw.numel() > 1 else int(tx_raw.item())
        if isinstance(rx_raw, torch.Tensor):
            rx_raw = rx_raw.view(-1).tolist() if rx_raw.numel() > 1 else int(rx_raw.item())

        # Fix DataLoader nesting for batch_size=1
        if isinstance(tx_raw, (list, tuple)) and len(tx_raw) == 1 and isinstance(tx_raw[0], (list, tuple)):
            tx_raw = tx_raw[0]
        if isinstance(rx_raw, (list, tuple)) and len(rx_raw) == 1 and isinstance(rx_raw[0], (list, tuple)):
            rx_raw = rx_raw[0]

        # ----- problem-specific structures + p_arr shape + objective wrapper -----
        if problem == "single":
            # Single Tx→Rx
            paths = find_all_paths(adj.cpu(), tx_raw, rx_raw)
            paths = paths_to_tensor(paths, device)

            # Initialize P: [B, n, n]
            p0 = torch.stack(
                [
                    create_normalized_tensor(
                        n, n, mask=adj, device=device
                    )
                    for _ in range(B)
                ],
                dim=0,
            )  # [B, n, n]

            objective_fn = objective_single_wrapper
            objective_kwargs = dict(
                paths=paths,
                B=B,
                tau_min=0.0,
                tau_max=0.0,
                per_band=False,
                eps=1e-12,
            )

        elif problem == "multicast":
            # Single Tx, multiple Rx, same message
            rx_list = list(rx_raw) if isinstance(rx_raw, (list, tuple)) else [rx_raw]

            # Multicast subgraphs from Tx to all rx_list
            subgraphs = find_multicast_subgraphs(adj.cpu(), tx_raw, rx_list)
            subgraphs_per_band = [subgraphs for _ in range(B)]

            # Initialize P: [B, n, n]
            p0 = torch.stack(
                [
                    create_normalized_tensor(
                        n, n, mask=adj, device=device
                    )
                    for _ in range(B)
                ],
                dim=0,
            )  # [B, n, n]

            objective_fn = objective_multicast_wrapper
            objective_kwargs = dict(
                subgraphs_per_band=subgraphs_per_band,
                B=B,
                adj_mat=adj,
                tau_min=0.0,
                tau_max=0.0,
                per_band=False,
                eps=1e-12,
            )

        elif problem == "multi":
            # One Tx, K receivers, distinct messages
            rx_list = list(rx_raw) if isinstance(rx_raw, (list, tuple)) else [rx_raw]
            K = len(rx_list)

            paths_k = []
            for rx_k in rx_list:
                paths_k_k = find_all_paths(adj.cpu(), tx_raw, int(rx_k))
                paths_k_k = paths_to_tensor(paths_k_k, device)
                paths_k.append(paths_k_k)

            # Initialize P: [B, K, n, n]
            p0 = torch.stack(
                [
                    torch.stack(
                        [
                            create_normalized_tensor(
                                n, n, mask=adj, device=device
                            )
                            for _ in range(K)
                        ],
                        dim=0,
                    )
                    for _ in range(B)
                ],
                dim=0,
            )  # [B, K, n, n]

            # Initialize Z: [B, K, n, n]
            Z0 = (
                adj.bool()
                .unsqueeze(0)
                .unsqueeze(0)
                .expand(B, K, n, n)
                .to(device)
                .float()
            )
            Z = nn.Parameter(Z0, requires_grad=True)

            objective_fn = objective_multicommodity_wrapper
            objective_kwargs = dict(
                paths_k=paths_k,
                B=B,
                adj_mat=adj,
                Z=Z,
                tau_min=0.0,
                tau_max=0.0,
                per_band=False,
                eps=1e-12,
            )

        elif problem == "converge":
            # K transmitters, one Rx, distinct messages
            tx_list = list(tx_raw) if isinstance(tx_raw, (list, tuple)) else [tx_raw]
            if isinstance(rx_raw, (list, tuple)):
                if len(rx_raw) != 1:
                    raise ValueError("For problem='converge', rx must contain exactly one receiver.")
                rx_scalar = int(rx_raw[0])
            else:
                rx_scalar = int(rx_raw)

            K = len(tx_list)

            paths_k = []
            for tx_k in tx_list:
                paths_k_k = find_all_paths(adj.cpu(), int(tx_k), rx_scalar)
                paths_k_k = paths_to_tensor(paths_k_k, device)
                paths_k.append(paths_k_k)

            # Initialize P: [B, K, n, n]
            p0 = torch.stack(
                [
                    torch.stack(
                        [
                            create_normalized_tensor(
                                n, n, mask=adj, device=device
                            )
                            for _ in range(K)
                        ],
                        dim=0,
                    )
                    for _ in range(B)
                ],
                dim=0,
            )  # [B, K, n, n]

            # Initialize Z: [B, K, n, n]
            Z0 = (
                adj.bool()
                .unsqueeze(0)
                .unsqueeze(0)
                .expand(B, K, n, n)
                .to(device)
                .float()
            )
            Z = nn.Parameter(Z0, requires_grad=True)

            objective_fn = objective_multicommodity_wrapper
            objective_kwargs = dict(
                paths_k=paths_k,
                B=B,
                adj_mat=adj,
                Z=Z,
                tau_min=0.0,
                tau_max=0.0,
                per_band=False,
                eps=1e-12,
            )

        elif problem == "multiunicast":
            # K Tx-Rx pairs, distinct messages
            tx_list = list(tx_raw) if isinstance(tx_raw, (list, tuple)) else [tx_raw]
            rx_list = list(rx_raw) if isinstance(rx_raw, (list, tuple)) else [rx_raw]

            if len(tx_list) != len(rx_list):
                raise ValueError(
                    f"For problem='multiunicast', len(tx) must equal len(rx), "
                    f"got {len(tx_list)} and {len(rx_list)}."
                )

            K = len(tx_list)

            paths_k = []
            for tx_k, rx_k in zip(tx_list, rx_list):
                paths_k_k = find_all_paths(adj.cpu(), int(tx_k), int(rx_k))
                paths_k_k = paths_to_tensor(paths_k_k, device)
                paths_k.append(paths_k_k)

            # Initialize P: [B, K, n, n]
            p0 = torch.stack(
                [
                    torch.stack(
                        [
                            create_normalized_tensor(
                                n, n, mask=adj, device=device
                            )
                            for _ in range(K)
                        ],
                        dim=0,
                    )
                    for _ in range(B)
                ],
                dim=0,
            )  # [B, K, n, n]

            # Initialize Z: [B, K, n, n]
            Z0 = (
                adj.bool()
                .unsqueeze(0)
                .unsqueeze(0)
                .expand(B, K, n, n)
                .to(device)
                .float()
            )
            Z = nn.Parameter(Z0, requires_grad=True)

            objective_fn = objective_multicommodity_wrapper
            objective_kwargs = dict(
                paths_k=paths_k,
                B=B,
                adj_mat=adj,
                Z=Z,
                tau_min=0.0,
                tau_max=0.0,
                per_band=False,
                eps=1e-12,
            )

        else:
            raise ValueError(f"Unknown problem type: {problem}")

        # ----- create optimizer (for multi-message problems optimize both P and Z) -----
        p_arr = nn.Parameter(p0, requires_grad=True)
        if problem in {"multi", "converge", "multiunicast"}:
            optimizer = optim.AdamW([p_arr, Z], lr=lr)
        else:
            optimizer = optim.AdamW([p_arr], lr=lr)

        # ----- run centralized optimization loop -----
        p_opt = classic_opt(
            num_iterations=num_iterations,
            optimizer=optimizer,
            adj_mat=adj,
            links_mat=links,
            p_arr=p_arr,
            sigma_noise=sigma,
            objective_fn=objective_fn,
            objective_kwargs=objective_kwargs,
        )
        p_opt_results.append(p_opt.detach())

        # ----- evaluate final objective (positive, not negated) -----
        rate = objective_fn(
            links_mat=links,
            P=p_opt,
            sigma_noise=sigma,
            **objective_kwargs,
        ).item()
        rate_results.append(rate)

    return rate_results, p_opt_results


# =============================================== Centralized FFN ======================================================
@torch.no_grad()
def evaluate_ffn(
    model,
    loader,
    problem="single",
):
    """
    Evaluate a trained FFN model.

    Args
    ----
    model : FFNPowerAllocator
        Trained FFN model.

    loader : DataLoader
        Evaluation dataset.

    problem : str
        Communication framework.

    Returns
    -------
    rate_results : list[float]
        Achieved rate for each graph.

    p_results : list[torch.Tensor]
        Predicted feasible allocations.
    """

    model.eval()
    objective_fn = select_ffn_objective(problem)
    rate_results = []
    p_results = []

    device = next(model.parameters()).device

    for batch in loader:

        batch = move_batch_to_device(batch, device)
        h, adj, sigma, tx, rx = unpack_batch(batch)

        # FFN prediction
        p_raw = model(h).squeeze(0)

        # Projection
        p = normalize_power(p_raw, adj)

        # Objective evaluation
        rate = objective_fn(
            h=h,
            p=p,
            sigma=sigma,
            adj=adj,
            tx=tx,
            rx=rx,
        )

        rate_results.append(float(rate.item()))
        p_results.append(p.detach().cpu())

    return rate_results, p_results

 # ============================= Centralized Bottleneck Rate ===========================================================
def compute_centralized_best_single_channel_rate(
    dataset,
    problem: str = "single",          # "single", "multicast", "multi", "converge", "multiunicast"
    sigma_noise=None,
):
    """
    Compute a bottleneck-based lower bound for each graph in the dataset.

    Convention:
      - links_matrix: complex channel links [B, n, n]
      - We work with |h|^2 at the chosen bottleneck edge(s).

    Problems:

    1) "single":
         Use the calc_sum_rate objective (mean over bands of best path),
         evaluated at a one-hot power tensor p_min (one active edge).

    2) "multicast":
         Use the multicast objective (mean over bands of max-subgraph
         bottleneck), evaluated at a one-hot p_min.

    3) problem in {"multi", "converge", "multiunicast"}:
         Multi-message case. For each message k, choose a bottleneck link
         as in the single case, assign amplitude 1/K on that link for that
         message, and evaluate the multi-commodity objective
         (reduced with 'mean' over messages).

         - "Multi": one Tx, K receivers
         - "converge": K transmitters, one Rx
         - "multiunicast": K Tx-Rx pairs
    """
    lower_bounds = []
    p_min = []

    for data in dataset:
        adj   = data.adj_matrix           # [n, n]
        links = data.links_matrix         # [B, n, n] (complex)
        B     = data.B if hasattr(data, "B") else links.shape[0]
        tx    = data.tx
        rx    = data.rx
        n     = adj.shape[0]

        sigma = sigma_noise if sigma_noise is not None else data.sigma
        if isinstance(sigma, torch.Tensor):
            sigma = float(sigma.item())

        # normalize tx/rx containers
        if isinstance(tx, torch.Tensor):
            tx = tx.view(-1).tolist() if tx.numel() > 1 else int(tx.item())
        if isinstance(rx, torch.Tensor):
            rx = rx.view(-1).tolist() if rx.numel() > 1 else int(rx.item())

        # ------------------------------------------------------------------
        # 1) SINGLE-UNICAST: bottleneck baseline
        # ------------------------------------------------------------------
        if problem == "single":
            paths = find_all_paths(adj, tx, rx)
            if not paths:
                lower_bounds.append(0.0)
                p_min.append(torch.zeros_like(links))
                continue

            h_band_vals   = []
            idx_band_best = []

            for b in range(B):
                band_path_mins = []
                band_path_idx  = []

                for path in paths:
                    if len(path) < 2:
                        continue

                    edge_list = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
                    rows, cols = zip(*edge_list)
                    rows_t = torch.tensor(rows, dtype=torch.long)
                    cols_t = torch.tensor(cols, dtype=torch.long)

                    gains = torch.abs(links[b][rows_t, cols_t]) ** 2
                    if gains.numel() == 0:
                        continue

                    min_val, argmin = gains.min(dim=0)
                    min_val = float(min_val.item())
                    i_min, j_min = edge_list[argmin.item()]

                    band_path_mins.append(min_val)
                    band_path_idx.append((b, i_min, j_min))

                if band_path_mins:
                    h_b = max(band_path_mins)
                    best_idx = band_path_mins.index(h_b)
                    h_band_vals.append(h_b)
                    idx_band_best.append(band_path_idx[best_idx])
                else:
                    h_band_vals.append(float("-inf"))
                    idx_band_best.append(None)

            if all(v == float("-inf") for v in h_band_vals):
                lower_bounds.append(0.0)
                p_min.append(torch.zeros_like(links))
                continue

            h_best = max(h_band_vals)

            p = torch.zeros_like(links)
            best_band_idx = h_band_vals.index(h_best)
            b_best, i_best, j_best = idx_band_best[best_band_idx]
            p[b_best, i_best, j_best] = 1.0
            p_min.append(p)

            paths_tensor = paths_to_tensor(paths, adj.device)
            rate = calc_sum_rate(
                h_arr=links,
                p_arr=p,
                sigma=sigma,
                paths_tensor=paths_tensor,
                B=B,
                tau=0.0,
                per_band=False,
                ignore_zero_edges=True,
                power_threshold=1e-8,
            )
            lower_bounds.append(float(rate.item()))
            continue

        # ------------------------------------------------------------------
        # 2) MULTICAST: bottleneck baseline
        # ------------------------------------------------------------------
        if problem == "multicast":
            if isinstance(rx, (list, tuple)):
                rx_list = list(rx)
            else:
                rx_list = [rx]

            subgraphs = find_multicast_subgraphs(adj, tx, rx_list)
            if (subgraphs is None) or (len(subgraphs) == 0):
                lower_bounds.append(0.0)
                p_min.append(torch.zeros_like(links))
                continue

            h_band_vals   = []
            idx_band_best = []

            for b in range(B):
                band_sub_min_vals = []
                band_sub_idx      = []

                for S in subgraphs:
                    if S is None or not torch.any(S):
                        continue
                    edge_idx = S.nonzero(as_tuple=False)
                    if edge_idx.numel() == 0:
                        continue

                    rows = edge_idx[:, 0]
                    cols = edge_idx[:, 1]

                    gains = torch.abs(links[b][rows, cols]) ** 2
                    min_val, argmin = gains.min(dim=0)
                    min_val = float(min_val.item())
                    i_min = int(rows[argmin].item())
                    j_min = int(cols[argmin].item())

                    band_sub_min_vals.append(min_val)
                    band_sub_idx.append((b, i_min, j_min))

                if band_sub_min_vals:
                    h_b = max(band_sub_min_vals)
                    best_idx = band_sub_min_vals.index(h_b)
                    h_band_vals.append(h_b)
                    idx_band_best.append(band_sub_idx[best_idx])
                else:
                    h_band_vals.append(float("-inf"))
                    idx_band_best.append(None)

            if all(v == float("-inf") for v in h_band_vals):
                lower_bounds.append(0.0)
                p_min.append(torch.zeros_like(links))
                continue

            h_best = max(h_band_vals)

            p = torch.zeros_like(links)
            best_band_idx = h_band_vals.index(h_best)
            b_best, i_best, j_best = idx_band_best[best_band_idx]
            p[b_best, i_best, j_best] = 1.0
            p_min.append(p)

            subgraphs_per_band = [subgraphs for _ in range(B)]

            rate = objective_multicast(
                h=links,
                p=p,
                sigma=sigma,
                adj=adj,
                subgraphs_per_band=subgraphs_per_band,
                eps=1e-12,
                tau_min=0.0,
                tau_max=0.0,
                per_band=False,
                ignore_zero_edges=True,
                power_threshold=1e-8,
            )
            lower_bounds.append(float(rate.item()))
            continue

        # ------------------------------------------------------------------
        # 3) MULTI-MESSAGE: "multi" / "converge" / "multiunicast"
        # ------------------------------------------------------------------
        if problem in {"multi", "converge", "multiunicast"}:
            # Build (tx_k, rx_k) pairs for each message k
            if problem == "multi":
                rx_list = list(rx) if isinstance(rx, (list, tuple)) else [rx]
                tx_list = [tx] * len(rx_list) if not isinstance(tx, (list, tuple)) else list(tx)
                if len(tx_list) == 1 and len(rx_list) > 1:
                    tx_list = tx_list * len(rx_list)

            elif problem == "converge":
                tx_list = list(tx) if isinstance(tx, (list, tuple)) else [tx]
                if isinstance(rx, (list, tuple)):
                    if len(rx) != 1:
                        raise ValueError("For problem='converge', rx must contain exactly one receiver.")
                    rx_scalar = int(rx[0])
                else:
                    rx_scalar = int(rx)
                rx_list = [rx_scalar] * len(tx_list)

            else:  # multiunicast
                tx_list = list(tx) if isinstance(tx, (list, tuple)) else [tx]
                rx_list = list(rx) if isinstance(rx, (list, tuple)) else [rx]
                if len(tx_list) != len(rx_list):
                    raise ValueError(
                        f"For problem='multiunicast', len(tx) must equal len(rx), "
                        f"got {len(tx_list)} and {len(rx_list)}."
                    )

            K = len(tx_list)

            if K == 0:
                lower_bounds.append(0.0)
                p_min.append(torch.zeros_like(links))
                continue

            # Precompute all paths for each message
            paths_k_list = []
            for tx_k, rx_k in zip(tx_list, rx_list):
                paths_k = find_all_paths(adj, int(tx_k), int(rx_k))
                if not paths_k:
                    paths_k_list.append(None)
                else:
                    paths_k_list.append(paths_to_tensor(paths_k, adj.device))

            chosen_edges = [None] * K

            for k, paths_k_tensor in enumerate(paths_k_list):
                if paths_k_tensor is None:
                    continue

                edge_start = paths_k_tensor[:, :-1]
                edge_end   = paths_k_tensor[:,  1:]
                if edge_start.numel() == 0:
                    continue

                h_band_vals_k   = []
                idx_band_best_k = []

                for b in range(B):
                    band_path_mins = []
                    band_path_idx  = []

                    for row in range(edge_start.shape[0]):
                        es = edge_start[row]
                        ee = edge_end[row]
                        valid_mask = (es >= 0) & (ee >= 0)
                        if not torch.any(valid_mask):
                            continue

                        rows = es[valid_mask]
                        cols = ee[valid_mask]

                        gains = torch.abs(links[b][rows, cols]) ** 2
                        if gains.numel() == 0:
                            continue

                        min_val, argmin = gains.min(dim=0)
                        min_val = float(min_val.item())
                        i_min = int(rows[argmin].item())
                        j_min = int(cols[argmin].item())

                        band_path_mins.append(min_val)
                        band_path_idx.append((b, i_min, j_min))

                    if band_path_mins:
                        h_b = max(band_path_mins)
                        best_idx = band_path_mins.index(h_b)
                        h_band_vals_k.append(h_b)
                        idx_band_best_k.append(band_path_idx[best_idx])
                    else:
                        h_band_vals_k.append(float("-inf"))
                        idx_band_best_k.append(None)

                if all(v == float("-inf") for v in h_band_vals_k):
                    continue

                h_k = max(h_band_vals_k)
                best_band_idx = h_band_vals_k.index(h_k)
                chosen_edges[k] = idx_band_best_k[best_band_idx]

            p_multi = torch.zeros((B, K, n, n), dtype=links.dtype, device=links.device)
            z_multi = torch.zeros_like(p_multi, dtype=links.dtype)

            for k, edge in enumerate(chosen_edges):
                if edge is None:
                    continue
                b_best, i_best, j_best = edge
                p_multi[b_best, k, i_best, j_best] = 1.0 / K
                z_multi[b_best, k, i_best, j_best] = 1.0

            if not any(edge is not None for edge in chosen_edges):
                lower_bounds.append(0.0)
                p_min.append(torch.zeros_like(links))
                continue

            rate = objective_multicommodity(
                h=links,
                p=p_multi,
                z=z_multi,
                sigma=sigma,
                adj=adj,
                paths_k=paths_k_list,
                tau_min=0.0,
                tau_max=0.0,
                reduce="mean",
                per_band=False,
                outage_as_neg_inf=False,
                ignore_zero_edges=True,
                power_threshold=1e-8,
            )

            lower_bounds.append(float(rate.item()))
            p_min.append(p_multi.sum(dim=1))   # [B, n, n]
            continue

        # ------------------------------------------------------------------
        # Fallback if problem is unknown
        # ------------------------------------------------------------------
        raise ValueError(f"Unknown problem type: {problem}")

    return np.array(lower_bounds, dtype=float), p_min


# ============================== Decentralized Bottleneck Rate =========================================================

def distributed_widest_path_single_band(
    adj,
    channel_power_b,
    tx,
    rx,
    max_iters=None,
    eps=1e-12,
):
    """
    Distributed widest-path / max-min bottleneck routing on one band.

    Each node maintains a local label q_i, interpreted as the best bottleneck
    value from node i to the destination rx.

    Update rule:
        q_i <- max_{j in N(i)} min(|h_ij|^2, q_j)

    Parameters
    ----------
    adj : torch.Tensor
        Directed adjacency matrix [n, n].

    channel_power_b : torch.Tensor
        Channel powers on one band [n, n], i.e. |h_b|^2.

    tx : int
        Source node.

    rx : int
        Destination node.

    max_iters : int or None
        Number of local update rounds. If None, uses n - 1.

    eps : float
        Numerical tolerance.

    Returns
    -------
    path : list[int] or None
        Selected path from tx to rx.

    bottleneck : float
        Bottleneck value along the selected path.

    next_hop : torch.Tensor
        Next-hop table [n].
    """
    device = adj.device
    n = adj.shape[0]

    tx = int(tx)
    rx = int(rx)

    if max_iters is None:
        max_iters = n - 1

    q = torch.zeros(n, device=device)
    q[rx] = float("inf")

    next_hop = -torch.ones(n, dtype=torch.long, device=device)

    for _ in range(max_iters):
        q_old = q.clone()

        for i in range(n):
            if i == rx:
                continue

            neigh = torch.where(adj[i] > 0)[0]
            if neigh.numel() == 0:
                continue

            candidate_vals = torch.minimum(
                channel_power_b[i, neigh],
                q_old[neigh],
            )

            val, idx = candidate_vals.max(dim=0)

            if val > q[i] + eps:
                q[i] = val
                next_hop[i] = neigh[idx]

        if torch.allclose(q, q_old, atol=eps, rtol=0.0):
            break

    if q[tx] <= eps:
        return None, 0.0, next_hop

    path = [tx]
    cur = tx
    visited = {cur}

    while cur != rx:
        nh = int(next_hop[cur].item())

        if nh < 0 or nh in visited:
            return None, 0.0, next_hop

        path.append(nh)
        visited.add(nh)
        cur = nh

    return path, float(q[tx].item()), next_hop


def distributed_best_band_widest_path(
    adj,
    links,
    tx,
    rx,
    B=None,
    max_iters=None,
    eps=1e-12,
):
    """
    Run decentralized widest-path routing on every band and select the best band.

    Parameters
    ----------
    adj : torch.Tensor
        Directed adjacency matrix [n, n].

    links : torch.Tensor
        Complex CSI tensor [B, n, n].

    tx : int
        Source node.

    rx : int
        Destination node.

    B : int or None
        Number of bands. If None, inferred from links.

    Returns
    -------
    best_path : list[int] or None
        Best selected path.

    best_band : int or None
        Selected band index.

    best_bottleneck : float
        Bottleneck value of selected path/band.
    """
    if B is None:
        B = links.shape[0]

    channel_power = links.abs() ** 2

    best_path = None
    best_band = None
    best_bottleneck = -float("inf")

    for b in range(B):
        path_b, bottleneck_b, _ = distributed_widest_path_single_band(
            adj=adj,
            channel_power_b=channel_power[b],
            tx=tx,
            rx=rx,
            max_iters=max_iters,
            eps=eps,
        )

        if path_b is not None and bottleneck_b > best_bottleneck:
            best_path = path_b
            best_band = b
            best_bottleneck = bottleneck_b

    if best_path is None:
        return None, None, 0.0

    return best_path, best_band, best_bottleneck


def compute_decentralized_best_single_channel_rate(
    dataset,
    problem="single",
    sigma_noise=None,
    max_iters=None,
    eps=1e-12,
):
    """
    Decentralized Best Single Channel / widest-path heuristic for all frameworks.

    The algorithm uses local widest-path updates to select paths and bands.
    The resulting power allocation is then evaluated using the same
    interference-aware objectives as the other benchmarks.

    Parameters
    ----------
    dataset : iterable
        Dataset yielding samples with:
            adj_matrix, links_matrix, sigma, tx, rx, B.

    problem : str
        One of:
            "single", "multicast", "multi", "converge", "multiunicast".

    sigma_noise : float or None
        Optional fixed noise value. If None, uses data.sigma.

    max_iters : int or None
        Number of distributed message-passing rounds.
        If None, uses n - 1, which is enough for exact widest path in a
        connected n-node graph under Bellman-Ford-style updates.

    eps : float
        Numerical tolerance.

    Returns
    -------
    rates : np.ndarray
        Achieved rate per sample.

    p_store : list
        Allocations used by the heuristic.
            - single/multicast: [B, n, n]
            - multi-like: [B, K, n, n]

    aux_store : list[dict]
        Metadata per sample: selected paths, bands, bottlenecks.
    """
    rates = []
    p_store = []
    aux_store = []

    for data in dataset:
        adj = data.adj_matrix
        links = data.links_matrix
        B = data.B if hasattr(data, "B") else links.shape[0]
        n = adj.shape[0]
        device = links.device

        sigma = data.sigma if sigma_noise is None else sigma_noise

        tx_list, rx_list = _normalize_tx_rx_for_problem(
            data.tx,
            data.rx,
            problem,
        )

        K = len(tx_list)

        selected_paths = []
        selected_bands = []
        selected_bottlenecks = []

        for tx_k, rx_k in zip(tx_list, rx_list):
            path_k, band_k, bottleneck_k = distributed_best_band_widest_path(
                adj=adj,
                links=links,
                tx=tx_k,
                rx=rx_k,
                B=B,
                max_iters=max_iters,
                eps=eps,
            )

            selected_paths.append(path_k)
            selected_bands.append(band_k)
            selected_bottlenecks.append(bottleneck_k)

        # ------------------------------------------------------------
        # SINGLE
        # ------------------------------------------------------------
        if problem == "single":
            p = torch.zeros_like(links)

            if selected_paths[0] is None:
                rates.append(0.0)
                p_store.append(p)
                aux_store.append(
                    {
                        "paths": selected_paths,
                        "bands": selected_bands,
                        "bottlenecks": selected_bottlenecks,
                    }
                )
                continue

            b = selected_bands[0]
            path = selected_paths[0]

            for u, v in zip(path[:-1], path[1:]):
                p[b, u, v] = 1.0

            paths_tensor = paths_to_tensor([path], device)

            rate = calc_sum_rate(
                h_arr=links,
                p_arr=p,
                sigma=sigma,
                paths_tensor=paths_tensor,
                B=B,
                tau=0.0,
                per_band=False,
                ignore_zero_edges=True,
                power_threshold=1e-8,
            )

            rates.append(float(rate.item()))
            p_store.append(p.detach())
            aux_store.append(
                {
                    "paths": selected_paths,
                    "bands": selected_bands,
                    "bottlenecks": selected_bottlenecks,
                }
            )
            continue

        # ------------------------------------------------------------
        # MULTICAST
        # ------------------------------------------------------------
        if problem == "multicast":
            p = torch.zeros_like(links)

            valid_paths = [path for path in selected_paths if path is not None]

            if len(valid_paths) == 0:
                rates.append(0.0)
                p_store.append(p)
                aux_store.append(
                    {
                        "paths": selected_paths,
                        "bands": selected_bands,
                        "bottlenecks": selected_bottlenecks,
                    }
                )
                continue

            # For multicast, each receiver may select a different best band.
            # We build one union subgraph per band.
            subgraphs_per_band = []
            for b in range(B):
                paths_b = [
                    path for path, band in zip(selected_paths, selected_bands)
                    if path is not None and band == b
                ]

                if len(paths_b) == 0:
                    subgraphs_per_band.append([])
                    continue

                S_b = _paths_to_multicast_subgraph(paths_b, n, device)
                subgraphs_per_band.append([S_b])

                for path in paths_b:
                    for u, v in zip(path[:-1], path[1:]):
                        p[b, u, v] = 1.0

            rate = objective_multicast(
                h=links,
                p=p,
                sigma=sigma,
                adj=adj,
                subgraphs_per_band=subgraphs_per_band,
                tau_min=0.0,
                tau_max=0.0,
                per_band=False,
                eps=1e-12,
                ignore_zero_edges=True,
                power_threshold=1e-8,
            )

            rates.append(float(rate.item()))
            p_store.append(p.detach())
            aux_store.append(
                {
                    "paths": selected_paths,
                    "bands": selected_bands,
                    "bottlenecks": selected_bottlenecks,
                    "subgraphs_per_band": subgraphs_per_band,
                }
            )
            continue

        # ------------------------------------------------------------
        # MULTI / CONVERGE / MULTIUNICAST
        # ------------------------------------------------------------
        if problem in {"multi", "converge", "multiunicast"}:
            p = torch.zeros(B, K, n, n, device=device)
            z = torch.zeros(B, K, n, n, device=device)

            paths_k = []
            has_any_path = False

            for k, path in enumerate(selected_paths):
                if path is None:
                    paths_k.append(torch.empty((0, 0), device=device, dtype=torch.long))
                    continue

                has_any_path = True
                b = selected_bands[k]

                for u, v in zip(path[:-1], path[1:]):
                    p[b, k, u, v] = 1.0 / max(K ** 0.5, 1)
                    z[b, k, u, v] = 1.0

                paths_k.append(paths_to_tensor([path], device))

            if not has_any_path:
                rates.append(0.0)
                p_store.append(p)
                aux_store.append(
                    {
                        "paths": selected_paths,
                        "bands": selected_bands,
                        "bottlenecks": selected_bottlenecks,
                    }
                )
                continue

            rate = objective_multicommodity(
                h=links,
                p=p,
                z=z,
                sigma=sigma,
                adj=adj,
                paths_k=paths_k,
                tau_min=0.0,
                tau_max=0.0,
                reduce="mean",
                per_band=False,
                outage_as_neg_inf=False,
                ignore_zero_edges=True,
                power_threshold=1e-8,
            )

            rates.append(float(rate.item()))
            p_store.append(p.detach())
            aux_store.append(
                {
                    "paths": selected_paths,
                    "bands": selected_bands,
                    "bottlenecks": selected_bottlenecks,
                    "Z": z.detach(),
                }
            )
            continue

        raise ValueError(f"Unknown problem={problem}")

    return np.array(rates, dtype=float), p_store, aux_store

# ============================== Decentralized Equal Power =============================================================

def compute_equal_power_bound(dataset, sigma_noise=False, problem="single"):
    """
    Compute the equal-power baseline rate depending on the problem type:

        - SINGLE:        One Tx -> one Rx, objective = calc_sum_rate
        - MULTICAST:     One Tx -> many Rx, same message, objective = objective_multicast
        - MULTI:         One Tx -> K Rx, different messages, objective = objective_multicommodity
        - CONVERGE:      K Tx -> one Rx, different messages, objective = objective_multicommodity
        - MULTIUNICAST:  K Tx-Rx pairs, different messages, objective = objective_multicommodity

    Uses the same power normalization rule as the model and centralized ADAM.
    """
    rate_bounds = []
    p_store = []  # keep the actual equal-power P (and Z for multi-message problems)

    for data in dataset:
        device = data.links_matrix.device
        data = data.to(device)

        adj = data.adj_matrix
        h = data.links_matrix       # [B,n,n]
        B = data.B if hasattr(data, "B") else h.shape[0]
        n = adj.shape[0]

        if not sigma_noise:
            sigma = data.sigma
        else:
            sigma = sigma_noise

        sigma_t = torch.tensor(sigma, device=device) if not isinstance(sigma, torch.Tensor) else sigma.to(device)

        # --------------------------
        # SINGLE
        # --------------------------
        if problem == "single":
            paths = find_all_paths(adj.cpu(), data.tx, data.rx)
            if not paths:
                rate_bounds.append(0.0)
                continue

            paths = paths_to_tensor(paths, device)

            # equal power tensor: [B,n,n]
            P = torch.zeros(B, n, n, device=device)
            P = init_equal_power(P, adj)
            P = normalize_power(P, adj).to(device)

            rate = calc_sum_rate(
                h_arr=h,
                p_arr=P,
                sigma=sigma_t,
                paths_tensor=paths,
                B=B,
                tau=0.0,
                eps=1e-12,
                per_band=False
            )
            rate_bounds.append(rate.item())
            p_store.append(P.detach())
            continue

        # --------------------------
        # MULTICAST
        # --------------------------
        if problem == "multicast":
            rx_list = data.rx
            if isinstance(rx_list, torch.Tensor):
                rx_list = rx_list.view(-1).tolist()
            if isinstance(rx_list, (list, tuple)) and len(rx_list) == 1 and isinstance(rx_list[0], (list, tuple)):
                rx_list = rx_list[0]

            subgraphs = find_multicast_subgraphs(adj.cpu(), data.tx, rx_list)
            if (subgraphs is None) or (len(subgraphs) == 0):
                rate_bounds.append(0.0)
                continue

            subgraphs_per_band = [subgraphs for _ in range(B)]

            # equal power [B,n,n]
            P = torch.zeros(B, n, n, device=device)
            P = init_equal_power(P, adj)
            P = normalize_power(P, adj).to(device)

            rate = objective_multicast(
                h=h,
                p=P,
                sigma=sigma_t,
                adj=adj,
                subgraphs_per_band=subgraphs_per_band,
                tau_min=0.0,
                tau_max=0.0,
                per_band=False,
                eps=1e-12,
            )
            rate_bounds.append(rate.item())
            p_store.append(P.detach())
            continue

        # --------------------------
        # MULTI-MESSAGE:
        # "multi" / "converge" / "multiunicast"
        # --------------------------
        if problem in {"multi", "converge", "multiunicast"}:
            tx = data.tx
            rx = data.rx

            if isinstance(tx, torch.Tensor):
                tx = tx.view(-1).tolist() if tx.numel() > 1 else int(tx.item())
            if isinstance(rx, torch.Tensor):
                rx = rx.view(-1).tolist() if rx.numel() > 1 else int(rx.item())

            # Build aligned (tx_k, rx_k) pairs
            if problem == "multi":
                rx_list = list(rx) if isinstance(rx, (list, tuple)) else [rx]
                tx_list = [int(tx)] * len(rx_list)

            elif problem == "converge":
                tx_list = list(tx) if isinstance(tx, (list, tuple)) else [tx]
                if isinstance(rx, (list, tuple)):
                    if len(rx) != 1:
                        raise ValueError("For problem='converge', rx must contain exactly one receiver.")
                    rx_scalar = int(rx[0])
                else:
                    rx_scalar = int(rx)
                rx_list = [rx_scalar] * len(tx_list)

            else:  # multiunicast
                tx_list = list(tx) if isinstance(tx, (list, tuple)) else [tx]
                rx_list = list(rx) if isinstance(rx, (list, tuple)) else [rx]
                if len(tx_list) != len(rx_list):
                    raise ValueError(
                        f"For problem='multiunicast', len(tx) must equal len(rx), "
                        f"got {len(tx_list)} and {len(rx_list)}."
                    )

            K = len(tx_list)

            paths_k = []
            has_path = False
            for tx_k, rx_k in zip(tx_list, rx_list):
                pk = find_all_paths(adj.cpu(), int(tx_k), int(rx_k))
                paths_k.append(
                    paths_to_tensor(pk, device) if pk else
                    torch.empty((0, 0), device=device, dtype=torch.long)
                )
                if pk:
                    has_path = True

            if not has_path:
                rate_bounds.append(0.0)
                continue

            # equal power P: shape [B,K,n,n]
            P = torch.zeros(B, K, n, n, device=device)
            for k in range(K):
                P[:, k] = init_equal_power(torch.zeros(B, n, n, device=device), adj)

            # normalize per-message
            for k in range(K):
                P[:, k] = normalize_power(P[:, k], adj).to(device)

            # routing matrix initialized to binary adjacency mask
            Z = adj.bool().float().unsqueeze(0).unsqueeze(0).expand(B, K, n, n).to(device)

            rate = objective_multicommodity(
                h=h,
                p=P,
                z=Z,
                sigma=sigma_t,
                adj=adj,
                paths_k=paths_k,
                tau_min=0.0,
                tau_max=0.0,
                reduce="mean",
                per_band=False,
                outage_as_neg_inf=False,
            )

            rate_bounds.append(rate.item())
            p_store.append((P.detach(), Z.detach()))
            continue

        raise ValueError(f"Unknown problem='{problem}'")

    return np.array(rate_bounds), p_store


# ======================================= Centralized Greedy Power Allocation ==========================================

def compute_centralized_greedy_power_rate(dataset, sigma_noise=False, problem="single"):
    """
    Greedy-power benchmark:

    SINGLE:
      - Use find_all_paths to get all Tx->Rx paths.
      - Take the path with the fewest hops (if multiple, choose randomly).
      - For each hop (u->v) on that path and each band b, set P[b,u,v] = 1/sqrt(B).
      - Feed P into calc_sum_rate (after normalize_power).

    MULTICAST:
      - Use find_multicast_subgraphs to get candidate subgraphs.
      - Pick the subgraph with the fewest links (if ties, pick randomly).
      - Use that single subgraph in objective_multicast.
      - Allocate P[b,u,v] = 1/sqrt(B) for edges (u->v) in this subgraph.
      - Feed P and subgraphs_per_band into objective_multicast (after normalize_power).

    MULTI / CONVERGE / MULTIUNICAST:
      - Build one message-specific shortest path per commodity/message.
      - Allocate P[b,k,u,v] = 1/sqrt(B) on each hop of the chosen path.
      - Pass P (and routing Z from adjacency) into objective_multicommodity
        (after per-message normalization).

        * MULTI:         one Tx, K receivers
        * CONVERGE:      K transmitters, one Rx
        * MULTIUNICAST:  K Tx-Rx pairs

    Returns:
      rate_bounds: np.ndarray of shape [len(dataset)]
      p_store:     list of P (and Z for multi-message problems) tensors.
    """
    rate_bounds = []
    p_store = []

    for data in dataset:
        device = data.links_matrix.device
        data = data.to(device)

        adj = data.adj_matrix
        h = data.links_matrix  # [B,n,n]
        B = data.B if hasattr(data, "B") else h.shape[0]
        n = adj.shape[0]

        sigma = data.sigma if (sigma_noise is False) else sigma_noise
        sigma_t = torch.tensor(sigma, device=device) if not isinstance(sigma, torch.Tensor) else sigma.to(device)

        # --------------------------
        # SINGLE
        # --------------------------
        if problem == "single":
            all_paths = find_all_paths(adj.cpu(), data.tx, data.rx)
            if not all_paths:
                rate_bounds.append(0.0)
                continue

            lengths = [len(p) - 1 for p in all_paths]
            min_len = min(lengths)
            shortest_paths = [p for p, L in zip(all_paths, lengths) if L == min_len]
            chosen_path = random.choice(shortest_paths)

            P = _build_P_from_path_single(h, chosen_path)   # [B,n,n]
            P = normalize_power(P, adj).to(device)

            paths_tensor = paths_to_tensor(all_paths, device)
            rate = calc_sum_rate(
                h_arr=h,
                p_arr=P,
                sigma=sigma_t,
                paths_tensor=paths_tensor,
                B=B,
                tau=0.0,
                eps=1e-12,
                per_band=False,
            )
            rate_bounds.append(rate.item())
            p_store.append(P.detach())
            continue

        # --------------------------
        # MULTICAST
        # --------------------------
        if problem == "multicast":
            rx_list = data.rx
            if isinstance(rx_list, torch.Tensor):
                rx_list = rx_list.view(-1).tolist()
            if isinstance(rx_list, (list, tuple)) and len(rx_list) == 1 and isinstance(rx_list[0], (list, tuple)):
                rx_list = rx_list[0]

            subgraphs = find_multicast_subgraphs(adj.cpu(), data.tx, rx_list)
            if (subgraphs is None) or (len(subgraphs) == 0):
                rate_bounds.append(0.0)
                continue

            best_sg = _select_shortest_subgraph(subgraphs)
            if best_sg is None:
                rate_bounds.append(0.0)
                continue

            subgraphs_per_band = [[best_sg] for _ in range(B)]

            P = _build_P_from_multicast_subgraph(h, adj, best_sg)   # [B,n,n]
            P = normalize_power(P, adj).to(device)

            rate = objective_multicast(
                h=h,
                p=P,
                sigma=sigma_t,
                adj=adj,
                subgraphs_per_band=subgraphs_per_band,
                tau_min=0.0,
                tau_max=0.0,
                per_band=False,
                eps=1e-12,
            )
            rate_bounds.append(rate.item())
            p_store.append(P.detach())
            continue

        # --------------------------
        # MULTI-MESSAGE:
        # "multi" / "converge" / "multiunicast"
        # --------------------------
        if problem in {"multi", "converge", "multiunicast"}:
            tx = data.tx
            rx = data.rx

            if isinstance(tx, torch.Tensor):
                tx = tx.view(-1).tolist() if tx.numel() > 1 else int(tx.item())
            if isinstance(rx, torch.Tensor):
                rx = rx.view(-1).tolist() if rx.numel() > 1 else int(rx.item())

            # Build aligned (tx_k, rx_k) pairs
            if problem == "multi":
                rx_list = list(rx) if isinstance(rx, (list, tuple)) else [rx]
                tx_list = [int(tx)] * len(rx_list)

            elif problem == "converge":
                tx_list = list(tx) if isinstance(tx, (list, tuple)) else [tx]
                if isinstance(rx, (list, tuple)):
                    if len(rx) != 1:
                        raise ValueError("For problem='converge', rx must contain exactly one receiver.")
                    rx_scalar = int(rx[0])
                else:
                    rx_scalar = int(rx)
                rx_list = [rx_scalar] * len(tx_list)

            else:  # multiunicast
                tx_list = list(tx) if isinstance(tx, (list, tuple)) else [tx]
                rx_list = list(rx) if isinstance(rx, (list, tuple)) else [rx]
                if len(tx_list) != len(rx_list):
                    raise ValueError(
                        f"For problem='multiunicast', len(tx) must equal len(rx), "
                        f"got {len(tx_list)} and {len(rx_list)}."
                    )

            K = len(tx_list)

            # Collect all paths for each message
            paths_k_lists = []
            has_path = False
            for tx_k, rx_k in zip(tx_list, rx_list):
                pk = find_all_paths(adj.cpu(), int(tx_k), int(rx_k))
                paths_k_lists.append(pk if pk else [])
                if pk:
                    has_path = True

            if not has_path:
                rate_bounds.append(0.0)
                continue

            # Build P from shortest paths per message
            # Assumes helper accepts one list of paths per message, regardless of problem semantics
            P = _build_P_from_paths_multi(h, paths_k_lists, K)  # [B,K,n,n]

            # normalize per message
            for k in range(K):
                P[:, k] = normalize_power(P[:, k], adj).to(device)

            # paths_k in tensor form for the objective
            paths_k_tensors = [
                paths_to_tensor(pk, device) if pk else torch.empty((0, 0), device=device, dtype=torch.long)
                for pk in paths_k_lists
            ]

            Z = adj.bool().float().unsqueeze(0).unsqueeze(0).expand(B, K, n, n).to(device)

            rate = objective_multicommodity(
                h=h,
                p=P,
                z=Z,
                sigma=sigma_t,
                adj=adj,
                paths_k=paths_k_tensors,
                tau_min=0.0,
                tau_max=0.0,
                reduce="mean",
                per_band=False,
                outage_as_neg_inf=False,
            )
            rate_bounds.append(rate.item())
            p_store.append((P.detach(), Z.detach()))
            continue

        raise ValueError(f"Unknown problem='{problem}'")

    return np.array(rate_bounds), p_store

# ======================================= Decentralized Greedy Power Allocation ========================================

def compute_decentralized_greedy_power_rate(
    dataset,
    sigma_noise=False,
    problem="single",
    max_iters=None,
):
    """
    Distributed greedy/shortest-path benchmark.

    This is the decentralized counterpart of compute_greedy_power_rate:
    it replaces global path enumeration with distributed Bellman-Ford
    shortest-path routing, then evaluates with the same objective functions.

    Parameters
    ----------
    dataset : iterable
        Dataset of graph samples.

    sigma_noise : float or False
        If False, use data.sigma. Otherwise use the provided value.

    problem : str
        "single", "multicast", "multi", "converge", or "multiunicast".

    max_iters : int or None
        Number of distributed Bellman-Ford update rounds.
        If None, uses n-1.

    Returns
    -------
    rate_bounds : np.ndarray
        Achieved rate per sample.

    p_store : list
        Stored allocations. For multi-message cases, stores (P, Z).
    """
    rate_bounds = []
    p_store = []

    for data in dataset:
        device = data.links_matrix.device
        data = data.to(device)

        adj = data.adj_matrix
        h = data.links_matrix
        B = data.B if hasattr(data, "B") else h.shape[0]
        n = adj.shape[0]

        sigma = data.sigma if (sigma_noise is False) else sigma_noise
        sigma_t = (
            torch.tensor(sigma, device=device)
            if not isinstance(sigma, torch.Tensor)
            else sigma.to(device)
        )

        # --------------------------
        # SINGLE
        # tx: int, rx: int
        # --------------------------
        if problem == "single":
            tx = int(data.tx)
            rx = int(data.rx)

            chosen_path = distributed_shortest_path(
                adj=adj,
                tx=tx,
                rx=rx,
                max_iters=max_iters,
            )

            if chosen_path is None:
                rate_bounds.append(0.0)
                p_store.append(torch.zeros_like(h))
                continue

            P = _build_P_from_path_single(h, chosen_path)
            P = normalize_power(P, adj).to(device)

            paths_tensor = paths_to_tensor([chosen_path], device)

            rate = calc_sum_rate(
                h_arr=h,
                p_arr=P,
                sigma=sigma_t,
                paths_tensor=paths_tensor,
                B=B,
                tau=0.0,
                eps=1e-12,
                per_band=False,
            )

            rate_bounds.append(rate.item())
            p_store.append(P.detach())
            continue

        # --------------------------
        # MULTICAST
        # tx: int, rx: list[int]
        # --------------------------
        if problem == "multicast":
            tx = int(data.tx)

            rx_list = data.rx
            if isinstance(rx_list, torch.Tensor):
                rx_list = rx_list.view(-1).tolist()
            if isinstance(rx_list, (list, tuple)) and len(rx_list) == 1 and isinstance(rx_list[0], (list, tuple)):
                rx_list = rx_list[0]
            rx_list = [int(r) for r in rx_list]

            paths = [
                distributed_shortest_path(adj, tx, r, max_iters=max_iters)
                for r in rx_list
            ]

            if any(path is None for path in paths):
                rate_bounds.append(0.0)
                p_store.append(torch.zeros_like(h))
                continue

            S = _paths_to_multicast_subgraph(paths, n, device)

            subgraphs_per_band = [[S] for _ in range(B)]

            P = torch.zeros(B, n, n, device=device)
            amp = 1.0 / (B ** 0.5)

            for path in paths:
                for u, v in zip(path[:-1], path[1:]):
                    P[:, int(u), int(v)] = amp

            P = normalize_power(P, adj).to(device)

            rate = objective_multicast(
                h=h,
                p=P,
                sigma=sigma_t,
                adj=adj,
                subgraphs_per_band=subgraphs_per_band,
                tau_min=0.0,
                tau_max=0.0,
                per_band=False,
                eps=1e-12,
            )

            rate_bounds.append(rate.item())
            p_store.append(P.detach())
            continue

        # --------------------------
        # MULTI
        # tx: int, rx: list[int]
        # --------------------------
        if problem == "multi":
            tx = int(data.tx)

            rx_list = data.rx
            if isinstance(rx_list, torch.Tensor):
                rx_list = rx_list.view(-1).tolist()
            rx_list = [int(r) for r in rx_list]

            paths_k = [
                distributed_shortest_path(adj, tx, r, max_iters=max_iters)
                for r in rx_list
            ]
            K = len(rx_list)

        # --------------------------
        # CONVERGE
        # tx: list[int], rx: int
        # --------------------------
        elif problem == "converge":
            tx_list = data.tx
            if isinstance(tx_list, torch.Tensor):
                tx_list = tx_list.view(-1).tolist()
            tx_list = [int(t) for t in tx_list]

            rx = int(data.rx)

            paths_k = [
                distributed_shortest_path(adj, t, rx, max_iters=max_iters)
                for t in tx_list
            ]
            K = len(tx_list)

        # --------------------------
        # MULTIUNICAST
        # tx: list[int], rx: list[int]
        # --------------------------
        elif problem == "multiunicast":
            tx_list = data.tx
            rx_list = data.rx

            if isinstance(tx_list, torch.Tensor):
                tx_list = tx_list.view(-1).tolist()
            if isinstance(rx_list, torch.Tensor):
                rx_list = rx_list.view(-1).tolist()

            tx_list = [int(t) for t in tx_list]
            rx_list = [int(r) for r in rx_list]

            if len(tx_list) != len(rx_list):
                raise ValueError("For multiunicast, tx and rx must have the same length.")

            paths_k = [
                distributed_shortest_path(adj, t, r, max_iters=max_iters)
                for t, r in zip(tx_list, rx_list)
            ]
            K = len(tx_list)

        else:
            raise ValueError(f"Unknown problem='{problem}'")

        # --------------------------
        # MULTI-MESSAGE EVALUATION
        # --------------------------
        if problem in {"multi", "converge", "multiunicast"}:
            if not any(path is not None for path in paths_k):
                rate_bounds.append(0.0)
                p_store.append(torch.zeros(B, len(paths_k), n, n, device=device))
                continue

            P = _build_P_from_paths_multi(h, paths_k, K)

            for k in range(len(paths_k)):
                P[:, k] = normalize_power(P[:, k], adj).to(device)

            paths_k_tensors = [
                paths_to_tensor([path], device)
                if path is not None else torch.empty((0, 0), device=device, dtype=torch.long)
                for path in paths_k
            ]

            Z = adj.bool().float().unsqueeze(0).unsqueeze(0).expand(B, len(paths_k), n, n).to(device)

            rate = objective_multicommodity(
                h=h,
                p=P,
                z=Z,
                sigma=sigma_t,
                adj=adj,
                paths_k=paths_k_tensors,
                tau_min=0.0,
                tau_max=0.0,
                reduce="mean",
                per_band=False,
                outage_as_neg_inf=False,
            )

            rate_bounds.append(rate.item())
            p_store.append((P.detach(), Z.detach()))

    return np.array(rate_bounds), p_store

# ======================================== Aux Functions ===============================================================

def _normalize_tx_rx_for_problem(tx, rx, problem) -> tuple[list[int], list[int]]:
    """
    Convert tx/rx fields into aligned commodity pairs.

    Returns
    -------
    tx_list : list[int]
    rx_list : list[int]
    """
    if isinstance(tx, torch.Tensor):
        tx = tx.view(-1).tolist() if tx.numel() > 1 else int(tx.item())
    if isinstance(rx, torch.Tensor):
        rx = rx.view(-1).tolist() if rx.numel() > 1 else int(rx.item())

    if problem == "single":
        return [int(tx)], [int(rx)]

    if problem == "multicast":
        rx_list = list(rx) if isinstance(rx, (list, tuple)) else [rx]
        return [int(tx)] * len(rx_list), [int(r) for r in rx_list]

    if problem == "multi":
        rx_list = list(rx) if isinstance(rx, (list, tuple)) else [rx]
        return [int(tx)] * len(rx_list), [int(r) for r in rx_list]

    if problem == "converge":
        tx_list = list(tx) if isinstance(tx, (list, tuple)) else [tx]

        if isinstance(rx, (list, tuple)):
            if len(rx) != 1:
                raise ValueError("For converge, rx must contain exactly one receiver.")
            rx_scalar = int(rx[0])
        else:
            rx_scalar = int(rx)

        return [int(t) for t in tx_list], [rx_scalar] * len(tx_list)

    if problem == "multiunicast":
        tx_list = list(tx) if isinstance(tx, (list, tuple)) else [tx]
        rx_list = list(rx) if isinstance(rx, (list, tuple)) else [rx]

        if len(tx_list) != len(rx_list):
            raise ValueError("For multiunicast, len(tx) must equal len(rx).")

        return [int(t) for t in tx_list], [int(r) for r in rx_list]

    raise ValueError(f"Unknown problem={problem}")


def _build_P_from_path_single(h, path_nodes):
    """
    For SINGLE / MULTICAST (per-path) case:
    Given a node-sequence path [v0, v1, ..., vL], construct P of shape [B,n,n] such that:
        - For each hop (u -> v) along the path and for each band b, P[b,u,v] = 1/sqrt(B)
        - All other entries are 0.
    """
    device = h.device
    B, n, _ = h.shape

    P = torch.zeros((B, n, n), device=device)
    amp = 1.0 / (B ** 0.5)

    # path_nodes is a list like [tx, ..., rx]
    for u, v in zip(path_nodes[:-1], path_nodes[1:]):
        P[:, u, v] = amp

    return P


def _build_P_from_paths_multi(h, paths_k, K):
    """
    MULTI-COMMODITY case:
    paths_k is a list length K, each element either:
      - list of node-sequence paths [v0,...,vL] (from find_all_paths)
      - or empty list / None if no path.

    For each commodity k with at least one path:
      - select a shortest path
      - allocate 1/sqrt(B) on each hop of that path for all bands
      - P has shape [B,K,n,n]
    """
    device = h.device
    B, n, _ = h.shape
    P = torch.zeros((B, K, n, n), device=device)
    amp = 1.0 / (B ** 0.5)

    for k, pk_raw in enumerate(paths_k):
        if pk_raw is None or len(pk_raw) == 0:
            continue

        # Case 1: pk_raw is already one selected path, e.g. [tx, ..., rx]
        if isinstance(pk_raw[0], int):
            chosen_path = pk_raw

        # Case 2: pk_raw is a list of candidate paths, e.g. [[tx,...,rx], ...]
        else:
            path_lengths = [len(p) - 1 for p in pk_raw]
            min_len = min(path_lengths)
            candidates = [p for p, L in zip(pk_raw, path_lengths) if L == min_len]
            chosen_path = random.choice(candidates)

        for u, v in zip(chosen_path[:-1], chosen_path[1:]):
            P[:, k, int(u), int(v)] = amp

    return P


def _select_shortest_subgraph(subgraphs):
    """
    Choose the subgraph with the smallest number of links.

    You MUST adapt this depending on how a 'subgraph' is represented
    in your code. The simplest case is that each subgraph is a dict
    with an 'edges' field listing (u,v) pairs.
    """
    if not subgraphs:
        return None

    # Example assumption: each subgraph is dict-like: {'edges': [(u,v), ...], ...}
    def num_edges(sg):
        if isinstance(sg, dict) and "edges" in sg:
            return len(sg["edges"])
        # Fallback: if sg itself is a list of edges
        if isinstance(sg, (list, tuple)) and sg and isinstance(sg[0], (list, tuple)):
            return len(sg)
        # Otherwise, treat as 0 (you should refine this to match your real structure)
        return 0

    edge_counts = [num_edges(sg) for sg in subgraphs]
    min_edges = min(edge_counts)
    candidates = [sg for sg, c in zip(subgraphs, edge_counts) if c == min_edges]

    return random.choice(candidates)


def _build_P_from_multicast_subgraph(h, adj, subgraph):
    """
    Build P for multicast from a chosen subgraph.

    Assumes 'subgraph' exposes its edges as either:
      - subgraph['edges'] = list of (u,v) pairs, or
      - subgraph is itself a list of (u,v) pairs.

    For each edge (u->v) in the chosen subgraph and each band b,
    P[b,u,v] = 1/sqrt(B). Others are 0.
    """
    device = h.device
    B, n, _ = h.shape
    P = torch.zeros((B, n, n), device=device)
    amp = 1.0 / (B ** 0.5)

    # Extract edges list from subgraph
    edges = [tuple(i.tolist()) for i in torch.nonzero(subgraph)]


    for (u, v) in edges:
        P[:, u, v] = amp

    return P

def _paths_to_multicast_subgraph(paths, n, device):
    """
    Convert several receiver paths into one multicast subgraph adjacency mask.

    Parameters
    ----------
    paths : list[list[int]]
        One selected path per receiver.

    n : int
        Number of nodes.

    device : torch.device
        Target device.

    Returns
    -------
    torch.Tensor
        Binary subgraph adjacency matrix [n, n].
    """
    S = torch.zeros(n, n, device=device)

    for path in paths:
        if path is None:
            continue

        for u, v in zip(path[:-1], path[1:]):
            S[int(u), int(v)] = 1.0

    return S

def distributed_shortest_path(adj, tx, rx, max_iters=None):
    """
    Distributed Bellman-Ford shortest path.

    Returns
    -------
    path : list[int] or None
        Minimum-hop path from tx to rx.
    """
    device = adj.device
    n = adj.shape[0]
    tx, rx = int(tx), int(rx)

    if max_iters is None:
        max_iters = n - 1

    dist = torch.full((n,), float("inf"), device=device)
    dist[rx] = 0.0
    next_hop = -torch.ones(n, dtype=torch.long, device=device)

    for _ in range(max_iters):
        dist_old = dist.clone()

        for i in range(n):
            if i == rx:
                continue

            neigh = torch.where(adj[i] > 0)[0]
            if neigh.numel() == 0:
                continue

            vals = 1.0 + dist_old[neigh]
            best_val, best_idx = vals.min(dim=0)

            if best_val < dist[i]:
                dist[i] = best_val
                next_hop[i] = neigh[best_idx]

        if torch.equal(dist, dist_old):
            break

    if not torch.isfinite(dist[tx]):
        return None

    path = [tx]
    cur = tx
    visited = {cur}

    while cur != rx:
        nh = int(next_hop[cur].item())
        if nh < 0 or nh in visited:
            return None

        path.append(nh)
        visited.add(nh)
        cur = nh

    return path


