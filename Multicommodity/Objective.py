import torch


def edge_rates_multicommodity(h: torch.Tensor,
                              p: torch.Tensor,
                              z: torch.Tensor,
                              sigma,
                              adj: torch.Tensor,
                              eps: float = 1e-12) -> torch.Tensor:
    """
    Compute per-edge rates R^{(b,k)}_{i->j}(P,Z)
        SNR^{(b,k)}_{i->j} = (|h^{(b)}_{i->j}|^2 * p^{(b,k)}_{i->j}^2 * z^{(b,k)}_{i->j}) / sigma_b^2

    Args:
        h:     [B, n, n] complex channel gains.
        p:     [B, K, n, n] nonnegative amplitudes.
        z:     [B, K, n, n] activations in [0,1] (soft or hard).
        sigma: float or [B] noise std per band.
        adj:   [n, n] {0,1} adjacency (1 = edge exists).
        eps:   small positive for numerical safety.

    Returns:
        R: [B, K, n, n] real, masked by adj (0 on nonexistent edges).
    """
    assert h.dim() == 3 and p.dim() == 4 and z.dim() == 4, "Shapes: h[B,n,n], p[B,K,n,n], z[B,K,n,n]"
    B, n, n2 = h.shape
    assert n == n2 and p.shape[0] == B and z.shape[0] == B and p.shape[2:] == (n, n) and z.shape[2:] == (n, n)
    device = h.device

    # Real adjacency mask
    adj_mask = (adj != 0).to(device=device, dtype=torch.float32)  # [n,n]

    # |h|^2 and mask
    h_abs2 = torch.nan_to_num(h, nan=0.0+0.0j).abs().pow(2).to(torch.float32)  # [B,n,n]
    h_abs2 = h_abs2 * adj_mask  # [B,n,n]

    # Powers and activations
    p2 = torch.nan_to_num(p, nan=0.0).clamp_min(0.0).pow(2)  # [B,K,n,n]
    zc = torch.nan_to_num(z, nan=0.0).clamp(0.0, 1.0)  # [B,K,n,n]

    # Signal term
    sig = h_abs2[:, None, :, :] * p2 * zc  # [B,K,n,n]

    # Noise per band (std -> variance)
    sigma = torch.as_tensor(sigma, dtype=torch.float32, device=device).reshape(-1)
    if sigma.numel() == 1:
        sigma = sigma.repeat(B)
    noise2 = (sigma ** 2)[:, None, None, None]  # [B,1,1,1]

    # SNR and rate
    snr = sig / (noise2 + eps)
    R = torch.log2(1.0 + snr)  # [B,K,n,n]

    # Zero nonexistent edges
    R *= adj_mask  # broadcast to [B,K,n,n]
    R = torch.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)
    return R


def objective_multicommodity(
    h: torch.Tensor,
    p: torch.Tensor,
    z: torch.Tensor,
    sigma,
    adj: torch.Tensor,
    paths_k: list,
    *,
    tau_min: float = 0.0,
    tau_max: float = 0.0,
    reduce: str = "sum",
    per_band: bool = False,
    outage_as_neg_inf: bool = False,
    ignore_zero_edges: bool = False,
    power_threshold: float = 1e-8,
) -> torch.Tensor:
    """
    Multi-commodity objective:

        J(P,Z) = sum_b sum_k max_{path ∈ Φ_{b,k}} min_{(i→j) ∈ path} R^{(b,k)}_{i→j}(P,Z),

    with optional soft-min (tau_min) over edges and soft-max (tau_max) over paths.

    Args:
        h:      [B, n, n] complex channels.
        p:      [B, K, n, n] real, nonnegative amplitudes/powers.
        z:      [B, K, n, n] (or whatever shape edge_rates_multicommodity expects).
        sigma:  noise std, broadcastable.
        adj:    [n, n] adjacency.
        paths_k: list of length K; paths_k[k] is a [P_k, L] tensor of node indices
                 (with -1 padding).
        tau_min: >0 for soft-min over edges in a path; 0 for hard min.
        tau_max: >0 for soft-max over paths; 0 for hard max.
        reduce:  "sum" or "mean" across commodities and bands.
        per_band: if True, return [B]; else scalar.
        outage_as_neg_inf:
                 If True, commodities with no paths (or degenerate paths) are set to -inf.
        ignore_zero_edges:
                 If True, when computing the bottleneck (min) over edges in a path,
                 only consider edges with p > power_threshold. Edges with
                 p <= power_threshold are treated like “non-edges” for the min.
                 This is useful for bottleneck baselines where p is one-hot and
                 we want the bottleneck over the actually powered edges.
        power_threshold:
                 Threshold for deciding whether an edge is “active”.

    Returns:
        If per_band=False:
            scalar: sum or mean over b,k (depending on `reduce`).
        If per_band=True:
            [B] tensor: per-band reduced value over commodities.
    """
    # Edge rates: [B, K, n, n]
    R = edge_rates_multicommodity(h, p, z, sigma, adj)
    B, K, n, _ = R.shape
    device = R.device

    r_bk = torch.zeros(B, K, device=device)  # per-band, per-commodity

    for k in range(K):
        paths = paths_k[k]
        if paths is None or paths.numel() == 0:
            if outage_as_neg_inf:
                r_bk[:, k] = float("-inf")
            continue

        edge_start = paths[:, :-1]  # [P_k, L-1]
        edge_end   = paths[:,  1:]  # [P_k, L-1]

        if edge_start.numel() == 0:  # degenerate paths with no edges
            if outage_as_neg_inf:
                r_bk[:, k] = float("-inf")
            continue

        # Structural validity of edges (not padding)
        valid_mask = (edge_start >= 0) & (edge_end >= 0)  # [P_k, L-1]

        # Gather edge rates along paths: [B, P_k, L-1]
        link_rates = R[:, k][:, edge_start, edge_end]

        # Base mask from topology (valid edges only)
        base_mask = valid_mask.unsqueeze(0)  # [B, P_k, L-1]

        if ignore_zero_edges:
            # Use p to decide which edges are "active"
            # p_k: [B, n, n]
            p_k = p[:, k]  # [B, n, n]
            p_edges = p_k[:, edge_start, edge_end]  # [B, P_k, L-1]

            # Active edges: valid in topology AND have power > threshold
            active_mask = base_mask & (p_edges > power_threshold)

            # Mask inactive edges with +inf so min/soft-min ignores them
            masked_link_rates = torch.where(
                active_mask,
                link_rates,
                torch.full_like(link_rates, float("inf")),
            )

            # Min over edges in each path
            if tau_min and tau_min > 0.0:
                path_vals = (-1.0 / tau_min) * torch.logsumexp(
                    -tau_min * masked_link_rates, dim=2
                )  # [B, P_k]
            else:
                path_vals, _ = masked_link_rates.min(dim=2)  # [B, P_k]

            # Paths with no active edges at all → set their value to 0
            # so they don't win against real paths.
            all_inactive = ~active_mask.any(dim=2)  # [B, P_k]
            path_vals = torch.where(
                all_inactive,
                torch.zeros_like(path_vals),
                path_vals,
            )
        else:
            # Original behavior: ignore only padding edges, but zero-power edges DO count in the min
            masked_link_rates = torch.where(
                base_mask,
                link_rates,
                torch.full_like(link_rates, float("inf")),
            )

            if tau_min and tau_min > 0.0:
                path_vals = (-1.0 / tau_min) * torch.logsumexp(
                    -tau_min * masked_link_rates, dim=2
                )  # [B, P_k]
            else:
                path_vals, _ = masked_link_rates.min(dim=2)  # [B, P_k]

        # Max over paths (per band)
        if tau_max and tau_max > 0.0:
            r_bk[:, k] = (1.0 / tau_max) * torch.logsumexp(tau_max * path_vals, dim=1)  # [B]
        else:
            r_bk[:, k], _ = path_vals.max(dim=1)  # [B]

    # Reduce across commodities and bands
    if per_band:
        # per band, reduce over commodities k
        if reduce == "sum":
            return r_bk.sum(dim=1)  # [B]
        else:
            return r_bk.mean(dim=1)  # [B]
    else:
        # scalar over b,k
        if reduce == "sum":
            return r_bk.sum()
        else:
            return r_bk.mean()




def objective_multicommodity_wrapper(
    links_mat,
    P,
    sigma_noise,
    paths_k,
    B,
    tau_min: float = 0.0,
    tau_max: float = 0.0,
    per_band: bool = False,
    eps: float = 1e-12,
    *,
    adj_mat=None,
    Z=None,
    **kwargs,
):
    """
    Wrapper around objective_multicommodity for the 1→K multi-commodity problem.
    Expects:
      - links_mat: h, shape [B, n, n] (or [B, K, n, n] if you designed it so)
      - P:         p, shape [B, K, n, n]
      - Z:         z, shape [B, K, n, n]
      - paths_k:   list length K of padded path tensors
      - adj_mat:   adjacency [n, n]
    """
    if Z is None:
        raise ValueError("objective_multicommodity_wrapper: Z must be provided for multi-commodity.")
    if adj_mat is None:
        raise ValueError("objective_multicommodity_wrapper: adj_mat must be provided for multi-commodity.")

    return objective_multicommodity(
        h=links_mat,
        p=P,
        z=Z,
        sigma=sigma_noise,
        adj=adj_mat,
        paths_k=paths_k,
        tau_min=tau_min,
        tau_max=tau_max,
        reduce="sum",
        per_band=per_band,
        outage_as_neg_inf=False,
    )


