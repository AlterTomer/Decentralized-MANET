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


#=======================================================================================================================
# Centralized Optimization
#=======================================================================================================================
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
    problem: str = "single",  # "single", "multicast", or "multi"
):
    """
    Evaluate centralized ADAM-based power allocation over a dataset for different
    problem types (single, multicast, multi-commodity), using objective wrappers.

    This function constructs a centralized optimization baseline that:
      1. Initializes a learnable power tensor P (and Z for multi-commodity).
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
              - data.tx:            transmitter node index (int).
              - data.rx:            for:
                                      * "single": int (single Rx),
                                      * "multicast": list[int] (receivers),
                                      * "multi": list[int] (commodities).
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
              - "single"    : single Tx→Rx unicast.
              - "multicast" : single Tx, K receivers, same message.
              - "multi"     : single Tx, K receivers, distinct messages (multi-commodity).

    Returns:
        rate_results (list[float]):
            For each graph, the final objective value (e.g., sum-rate or multicast
            bottleneck rate) obtained by centralized optimization.

        p_opt_results (list[torch.Tensor]):
            For each graph, the optimized power tensor:
              - "single" / "multicast": [B, n, n].
              - "multi"              : [B, K, n, n].
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

        # ----- problem-specific structures + p_arr shape + objective wrapper -----
        if problem == "single":
            # Single Tx→Rx: scalar rx
            paths = find_all_paths(adj.cpu(), data.tx, data.rx)
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
            # Multicast: rx is list of receivers
            rx_list = data.rx
            # Fix DataLoader nesting for batch_size=1
            if isinstance(rx_list, (list, tuple)) and len(rx_list) == 1 and isinstance(rx_list[0], (list, tuple)):
                rx_list = rx_list[0]

            # Multicast subgraphs (Steiner-like) from Tx to all rx_list
            subgraphs = find_multicast_subgraphs(adj.cpu(), data.tx, rx_list)
            # Same subgraphs for each band (channel differs per band, topology same)
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
            # Multi-commodity: Tx→rx_k, each k a different message
            rx_list = data.rx
            # Fix DataLoader nesting for batch_size=1
            if isinstance(rx_list, (list, tuple)) and len(rx_list) == 1 and isinstance(rx_list[0], (list, tuple)):
                rx_list = rx_list[0]
            K = len(rx_list)

            # Build paths_k: list length K, each tensor of paths Tx→rx_k
            paths_k = []
            for rx_k in rx_list:
                paths_k_k = find_all_paths(adj.cpu(), data.tx, rx_k)
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
                        dim=0,  # [K, n, n]
                    )
                    for _ in range(B)
                ],
                dim=0,          # [B, K, n, n]
            )

            # Initialize Z: routing/splitting variables, [B, K, n, n]
            Z0 = (
                adj.bool()
                .unsqueeze(0)  # [1, n, n]
                .unsqueeze(0)  # [1, 1, n, n]
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

        # ----- create optimizer (note: for multi we optimize P and Z) -----
        p_arr = nn.Parameter(p0, requires_grad=True)
        if problem == "multi":
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


def evaluate_centralized_adam_single(data, B, lr=0.1, num_iterations=100):
    """
    Evaluate the centralized ADAM optimization strategy for a set of graphs.

    This function performs ADAM-based power allocation optimization for each
    graph in the loader, aiming to maximize the sum rate. For each graph:
    - It initializes a learnable power allocation tensor `p_arr` of shape [B, n, n].
    - Optimizes it over `num_iterations` steps using the given learning rate.
    - Stores the final optimized power allocation and the corresponding rate.

    Args:
        loader (DataLoader or Iterable): A loader or iterable yielding graph data objects.
            Each data object is expected to contain:
                - data.adj_matrix: [n, n] binary adjacency matrix
                - data.links_matrix: [B, n, n] complex link gains
                - data.sigma: scalar noise standard deviation
                - data.tx: transmitter node index (int)
                - data.rx: receiver node index (int)
        B (int): Number of frequency bands
        lr (float): Learning rate for the ADAM optimizer (default 0.1)
        num_iterations (int): Number of optimization steps per graph (default 100)

    Returns:
        rate_results (List[float]): Achieved sum rate for each graph after optimization
        p_opt_results (List[Tensor]): Optimized power allocation tensors of shape [B, n, n] for each graph
    """

    adj = data.adj_matrix                # [n, n] binary adjacency matrix
    links = data.links_matrix            # [B, n, n] complex channel gains
    sigma = data.sigma                   # noise std
    tx = data.tx                         # transmitter index
    rx = data.rx                         # receiver index
    paths = find_all_paths(adj, tx, rx)  # list of paths from Tx to Rx
    paths = paths_to_tensor(paths, data.links_matrix.device)

    # Initialize power allocation tensor [B, n, n] with masked entries
    p_arr = torch.stack([
        create_normalized_tensor(adj.shape[0], adj.shape[1], mask=adj) for _ in range(B)
    ])
    p_arr = nn.Parameter(p_arr, requires_grad=True)

    # ADAM optimizer on p_arr
    optimizer = optim.AdamW([p_arr], lr=lr)

    # Optimize with respect to sum rate (maximize via negative loss)
    p_opt = classic_opt(num_iterations, optimizer, adj, links, p_arr, sigma, paths, B)

    # Evaluate and store the achieved rate
    rate = calc_sum_rate(links, p_opt, sigma, paths, B)
    # print(f'lower bound = {compute_lower_bound_single(data, sigma)}')

    return rate, p_opt



def compute_strongest_bottleneck_rate(
    dataset,
    problem: str = "single",          # "single", "multicast", "multi"
    sigma_noise=None,
    multi_mode: str = "global",       # kept for compatibility, not used
):
    """
    Compute a bottleneck-based lower bound for each graph in the dataset.

    Convention:
      - links_matrix: complex channel links [B, n, n]
      - We work with |h|^2 at the chosen bottleneck edge(s).

    Problems:

    1) problem == "single":
         Use the calc_sum_rate objective (mean over bands of best path),
         evaluated at a one-hot power tensor p_min (one active edge).

    2) problem == "multicast":
         Use the multicast objective (mean over bands of max-subgraph
         bottleneck), evaluated at a one-hot p_min.

    3) problem == "multi" (multi-commodity, K receivers, K messages):
         For each receiver k, choose a bottleneck link as in the single case,
         assign amplitude 1/K on that link for that commodity, and evaluate
         the multi-commodity objective (reduced with 'mean' over commodities).
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
        # Make sure sigma is a plain float for this analytic bound
        if isinstance(sigma, torch.Tensor):
            sigma = float(sigma.item())

        # ------------------------------------------------------------------
        # 1) SINGLE-UNICAST: bottleneck baseline
        # ------------------------------------------------------------------
        if problem == "single":
            paths = find_all_paths(adj, tx, rx)
            if not paths:
                lower_bounds.append(0.0)
                p_min.append(torch.zeros_like(links))
                continue

            h_band_vals   = []   # per-band worst-path bottleneck
            idx_band_best = []   # per-band (b,i,j) realizing h_b

            for b in range(B):
                band_path_mins = []   # path bottlenecks in band b
                band_path_idx  = []   # (b,i,j) for each path's bottleneck

                for path in paths:
                    if len(path) < 2:
                        continue

                    edge_list = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
                    rows, cols = zip(*edge_list)
                    rows_t = torch.tensor(rows, dtype=torch.long)
                    cols_t = torch.tensor(cols, dtype=torch.long)

                    gains = torch.abs(links[b][rows_t, cols_t]) ** 2  # [L]
                    if gains.numel() == 0:
                        continue

                    # bottleneck along this path
                    min_val, argmin = gains.min(dim=0)
                    min_val = float(min_val.item())
                    i_min, j_min = edge_list[argmin.item()]

                    band_path_mins.append(min_val)
                    band_path_idx.append((b, i_min, j_min))

                if band_path_mins:
                    h_b = min(band_path_mins)
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

            # Pick best band according to the bottleneck definition
            h_best = max(h_band_vals)

            # Build one-hot power tensor
            p = torch.zeros_like(links)
            best_band_idx = h_band_vals.index(h_best)
            b_best, i_best, j_best = idx_band_best[best_band_idx]
            p[b_best, i_best, j_best] = 1.0
            p_min.append(p)

            # NEW: evaluate with the *same* objective as the GNN,
            #      but ignoring zero-power edges in the min.
            paths_tensor = paths_to_tensor(paths, adj.device)
            rate = calc_sum_rate(
                h_arr=links,
                p_arr=p,
                sigma=sigma,
                paths_tensor=paths_tensor,
                B=B,
                tau=0.0,
                per_band=False,
                ignore_zero_edges=True,     # <<< important for one-hot p
                power_threshold=1e-8,
            )
            lower_bounds.append(float(rate.item()))
            continue  # next graph

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

            h_band_vals   = []  # per-band worst-subgraph bottleneck
            idx_band_best = []  # per-band (b,i,j)

            for b in range(B):
                band_sub_min_vals = []
                band_sub_idx      = []

                for S in subgraphs:
                    if S is None or not torch.any(S):
                        continue
                    edge_idx = S.nonzero(as_tuple=False)  # [E, 2]
                    if edge_idx.numel() == 0:
                        continue

                    rows = edge_idx[:, 0]
                    cols = edge_idx[:, 1]

                    gains = torch.abs(links[b][rows, cols]) ** 2  # [E]
                    min_val, argmin = gains.min(dim=0)
                    min_val = float(min_val.item())
                    i_min = int(rows[argmin].item())
                    j_min = int(cols[argmin].item())

                    band_sub_min_vals.append(min_val)
                    band_sub_idx.append((b, i_min, j_min))

                if band_sub_min_vals:
                    # Worst multicast subgraph in band b
                    h_b = min(band_sub_min_vals)
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

            # One-hot power tensor on the decisive edge
            p = torch.zeros_like(links)
            best_band_idx = h_band_vals.index(h_best)
            b_best, i_best, j_best = idx_band_best[best_band_idx]
            p[b_best, i_best, j_best] = 1.0
            p_min.append(p)

            # Subgraphs list is the same for all bands in this baseline
            subgraphs_per_band = [subgraphs for _ in range(B)]

            # NEW: use multicast objective with ignore_zero_edges=True
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
                ignore_zero_edges=True,     # <<< key
                power_threshold=1e-8,
            )
            lower_bounds.append(float(rate.item()))
            continue  # next graph

        # ------------------------------------------------------------------
        # 3) MULTI-COMMODITY: K commodities baseline
        # ------------------------------------------------------------------
        if problem == "multi":
            # rx can be list or single index
            if isinstance(rx, (list, tuple)):
                rx_list = list(rx)
            else:
                rx_list = [rx]
            K = len(rx_list)

            if K == 0:
                lower_bounds.append(0.0)
                p_min.append(torch.zeros_like(links))
                continue

            # Precompute all paths for each commodity and store as tensors
            paths_k_list = []
            for rx_k in rx_list:
                paths_k = find_all_paths(adj, tx, rx_k)
                if not paths_k:
                    paths_k_list.append(None)
                else:
                    paths_k_list.append(paths_to_tensor(paths_k, adj.device))

            # For each commodity k, find its bottleneck edge (b*, i*, j*)
            chosen_edges = [None] * K  # entries: (b_best, i_best, j_best) or None

            for k, paths_k_tensor in enumerate(paths_k_list):
                if paths_k_tensor is None:
                    continue  # this commodity has no route

                edge_start = paths_k_tensor[:, :-1]  # [P_k, L-1]
                edge_end   = paths_k_tensor[:,  1:]  # [P_k, L-1]
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
                        # valid edges in this path
                        valid_mask = (es >= 0) & (ee >= 0)
                        if not torch.any(valid_mask):
                            continue

                        rows = es[valid_mask]
                        cols = ee[valid_mask]

                        gains = torch.abs(links[b][rows, cols]) ** 2  # [L_path]
                        if gains.numel() == 0:
                            continue

                        min_val, argmin = gains.min(dim=0)
                        min_val = float(min_val.item())
                        i_min = int(rows[argmin].item())
                        j_min = int(cols[argmin].item())

                        band_path_mins.append(min_val)
                        band_path_idx.append((b, i_min, j_min))

                    if band_path_mins:
                        h_b = min(band_path_mins)  # worst path in band b
                        best_idx = band_path_mins.index(h_b)
                        h_band_vals_k.append(h_b)
                        idx_band_best_k.append(band_path_idx[best_idx])
                    else:
                        h_band_vals_k.append(float("-inf"))
                        idx_band_best_k.append(None)

                if all(v == float("-inf") for v in h_band_vals_k):
                    continue  # no feasible band for this commodity

                h_k = max(h_band_vals_k)   # best band for commodity k
                best_band_idx = h_band_vals_k.index(h_k)
                chosen_edges[k] = idx_band_best_k[best_band_idx]  # (b_best, i_best, j_best)

            # Build p and z for multi-commodity objective: [B, K, n, n]
            p_multi = torch.zeros((B, K, n, n), dtype=links.dtype, device=links.device)
            z_multi = torch.zeros_like(p_multi, dtype=links.dtype)

            for k, edge in enumerate(chosen_edges):
                if edge is None:
                    continue
                b_best, i_best, j_best = edge
                # amplitude 1/K for each commodity's chosen link
                p_multi[b_best, k, i_best, j_best] = 1.0 / K
                z_multi[b_best, k, i_best, j_best] = 1.0

            # If no commodity has a path: rate = 0
            if not any(edge is not None for edge in chosen_edges):
                lower_bounds.append(0.0)
                p_min.append(torch.zeros_like(links))
                continue

            # NEW: evaluate multi-commodity objective, ignoring zero-power edges
            rate = objective_multicommodity(
                h=links,
                p=p_multi,
                z=z_multi,
                sigma=sigma,
                adj=adj,
                paths_k=paths_k_list,
                tau_min=0.0,
                tau_max=0.0,
                reduce="mean",              # average over commodities
                per_band=False,
                outage_as_neg_inf=False,
                ignore_zero_edges=True,     # <<< key
                power_threshold=1e-8,
            )

            lower_bounds.append(float(rate.item()))
            # For consistency with single/multicast, store total power per edge:
            p_min.append(p_multi.sum(dim=1))   # [B, n, n]
            continue

        # ------------------------------------------------------------------
        # Fallback if problem is unknown
        # ------------------------------------------------------------------
        raise ValueError(f"Unknown problem type: {problem}")

    return np.array(lower_bounds, dtype=float), p_min




def compute_equal_power_bound(dataset, sigma_noise=False, problem="single"):
    """
    Compute the equal-power baseline rate depending on the problem type:

        - SINGLE:      One Tx → one Rx, objective=calc_sum_rate
        - MULTICAST:   One Tx → many Rx, objective=objective_multicast
        - MULTI:       One Tx → K Rx w/ different flows, objective=objective_multicommodity

    Uses the same power normalization rule as the model and centralized ADAM.
    """
    rate_bounds = []
    p_store = []  # keep the actual equal-power P (and Z if multi)

    for data in dataset:
        device = data.links_matrix.device
        data = data.to(device)

        adj = data.adj_matrix
        h = data.links_matrix       # [B,n,n]
        B = data.B if hasattr(data, "B") else h.shape[0]
        n = adj.shape[0]

        if sigma_noise is False:
            sigma = data.sigma
        else:
            sigma = sigma_noise

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
                sigma=torch.tensor(sigma, device=device),
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
                sigma=torch.tensor(sigma, device=device),
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
        # MULTI-COMMODITY
        # --------------------------
        if problem == "multi":
            rx_list = data.rx
            if isinstance(rx_list, (list, tuple)) and len(rx_list) == 1 and isinstance(rx_list[0], (list, tuple)):
                rx_list = rx_list[0]
            K = len(rx_list)

            paths_k = []
            has_path = False
            for rx_k in rx_list:
                pk = find_all_paths(adj.cpu(), data.tx, rx_k)
                paths_k.append(paths_to_tensor(pk, device) if pk else torch.empty((0,0), device=device, dtype=torch.long))
                if pk:
                    has_path = True

            if not has_path:
                rate_bounds.append(0.0)
                continue

            # equal power P: shape [B,K,n,n]
            P = torch.zeros(B, K, n, n, device=device)
            for k in range(K):
                P[:, k] = init_equal_power(torch.zeros(B, n, n), adj)

            P = normalize_power(P, adj).to(device)

            # routing matrix initialized to binary adjacency mask
            Z = adj.bool().float().unsqueeze(0).unsqueeze(0).expand(B, K, n, n).to(device)

            rate = objective_multicommodity(
                h=h,
                p=P,
                z=Z,
                sigma=torch.tensor(sigma, device=device),
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


def compute_greedy_power_rate(dataset, sigma_noise=False, problem="single"):
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

    MULTI (multicommodity):
      - For each receiver rx_k:
          - Use find_all_paths to get Tx->rx_k paths.
          - Pick a shortest path (ties broken randomly).
          - Allocate P[b,k,u,v] = 1/sqrt(B) on each hop of that path.
      - Pass P (and routing Z from adjacency) into objective_multicommodity (after normalize_power).

    Returns:
      rate_bounds: np.ndarray of shape [len(dataset)]
      p_store:     list of P (and Z for multi) tensors for debugging/inspection.
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

        # --------------------------
        # SINGLE
        # --------------------------
        if problem == "single":
            all_paths = find_all_paths(adj.cpu(), data.tx, data.rx)
            if not all_paths:
                rate_bounds.append(0.0)
                continue

            # shortest by number of edges
            lengths = [len(p) - 1 for p in all_paths]
            min_len = min(lengths)
            shortest_paths = [p for p, L in zip(all_paths, lengths) if L == min_len]
            chosen_path = random.choice(shortest_paths)

            P = _build_P_from_path_single(h, adj, chosen_path)   # [B,n,n]
            P = normalize_power(P, adj).to(device)

            paths_tensor = paths_to_tensor(all_paths, device)    # objective still uses all paths
            rate = calc_sum_rate(
                h_arr=h,
                p_arr=P,
                sigma=torch.tensor(sigma, device=device),
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
            if isinstance(rx_list, (list, tuple)) and len(rx_list) == 1 and isinstance(rx_list[0], (list, tuple)):
                rx_list = rx_list[0]

            subgraphs = find_multicast_subgraphs(adj.cpu(), data.tx, rx_list)
            if (subgraphs is None) or (len(subgraphs) == 0):
                rate_bounds.append(0.0)
                continue

            # Choose the subgraph with the smallest number of links
            best_sg = _select_shortest_subgraph(subgraphs)
            if best_sg is None:
                rate_bounds.append(0.0)
                continue

            # Use ONLY this subgraph in all bands
            subgraphs_per_band = [[best_sg] for _ in range(B)]

            # Build P based on that subgraph
            P = _build_P_from_multicast_subgraph(h, adj, best_sg)   # [B,n,n]
            P = normalize_power(P, adj).to(device)

            rate = objective_multicast(
                h=h,
                p=P,
                sigma=torch.tensor(sigma, device=device),
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
        # MULTI-COMMODITY
        # --------------------------
        if problem == "multi":
            rx_list = data.rx
            if isinstance(rx_list, (list, tuple)) and len(rx_list) == 1 and isinstance(rx_list[0], (list, tuple)):
                rx_list = rx_list[0]
            K = len(rx_list)

            # Collect all paths for each commodity (same as in equal-power)
            paths_k_lists = []
            has_path = False
            for rx_k in rx_list:
                pk = find_all_paths(adj.cpu(), data.tx, rx_k)
                paths_k_lists.append(pk if pk else [])
                if pk:
                    has_path = True

            if not has_path:
                rate_bounds.append(0.0)
                continue

            # Build P from shortest paths per commodity
            P = _build_P_from_paths_multi(h, adj, paths_k_lists, K)  # [B,K,n,n]
            P = normalize_power(P, adj).to(device)

            # paths_k in tensor form for the objective
            paths_k_tensors = [
                paths_to_tensor(pk, device) if pk else torch.empty((0, 0), device=device, dtype=torch.long)
                for pk in paths_k_lists
            ]

            # routing matrix same as in equal-power baseline
            Z = adj.bool().float().unsqueeze(0).unsqueeze(0).expand(B, K, n, n).to(device)

            rate = objective_multicommodity(
                h=h,
                p=P,
                z=Z,
                sigma=torch.tensor(sigma, device=device),
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


def _build_P_from_path_single(h, adj, path_nodes):
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


def _build_P_from_paths_multi(h, adj, paths_k, K):
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
        # pk_raw was created earlier via find_all_paths; here we re-use those lists.
        if pk_raw is None or len(pk_raw) == 0:
            continue

        # pk_raw is a list of node lists, e.g., [[tx,...,rx], ...]
        path_lengths = [len(p) - 1 for p in pk_raw]  # number of edges
        min_len = min(path_lengths)
        candidates = [p for p, L in zip(pk_raw, path_lengths) if L == min_len]

        chosen_path = random.choice(candidates)

        # Set P[:,k,u,v] = 1/sqrt(B) for each hop
        for u, v in zip(chosen_path[:-1], chosen_path[1:]):
            P[:, k, u, v] = amp

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







