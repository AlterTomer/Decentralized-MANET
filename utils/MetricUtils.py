import torch
import torch.nn.functional as F


#=======================================================================================================================
# Rate and Loss
#=======================================================================================================================
def link_rate(h, p, sigma):
    """
    Computes the link rate for a single channel.

    Args:
        h (tensor): Channel gain.
        p (tensor): Power allocation.
        sigma (float): Noise power.

    Returns:
        torch.Tensor: Link rate (scalar).
    """
    snr = ((torch.abs(h) * p) ** 2) / (sigma ** 2)
    return torch.log2(1 + snr)


def calc_sum_rate(
    h_arr,
    p_arr,
    sigma,
    paths_tensor,
    B,
    tau: float = 0.0,
    eps: float = 1e-12,
    per_band: bool = False,
    ignore_zero_edges: bool = False,
    power_threshold: float = 1e-8,
):
    """
    Fully vectorized sum-rate over all paths and bands.
    Any NaNs/Infs in h_arr/p_arr are treated as zeros (i.e., contribute 0 rate).
    When num_paths == 0, returns 0.0 (interpretable as outage).

    Args:
        h_arr:   [B, n, n] complex channel matrices (can contain NaNs/Infs).
        p_arr:   [B, n, n] real, nonnegative power allocations (can contain NaNs/Infs).
        sigma:   float or 0-D/1-D tensor noise std (not variance!) per-graph; broadcastable.
        paths_tensor: [num_paths, max_path_len] with -1 padding.
        B:       int, number of frequency bands.
        tau:     float, if >0 uses soft-min over edges with temperature tau.
        eps:     float, numerical floor.
        per_band: Boolean, if True return per band rates [R1, R2, ..., RB],
                  else return mean rate across bands.
        ignore_zero_edges:
                  If True, when computing the bottleneck (min) over edges in a path,
                  only consider edges with p > power_threshold. Edges with
                  p <= power_threshold are treated like “non-edges” for the min.
                  Useful for bottleneck baselines with one-hot p.
        power_threshold:
                  Threshold for deciding whether an edge is “active”.

    Returns:
        torch.Tensor scalar, or [B] if per_band=True.
    """
    device = h_arr.device

    # Sigma handling
    if not isinstance(sigma, torch.Tensor):
        sigma = torch.tensor(sigma, dtype=torch.float32, device=device)
    sigma = torch.nan_to_num(sigma, nan=1.0, posinf=1.0, neginf=1.0).clamp_min(eps)

    # Sanitize inputs
    h_arr = torch.nan_to_num(h_arr, nan=0.0, posinf=0.0, neginf=0.0)
    p_arr = torch.nan_to_num(p_arr, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)

    num_paths, _ = paths_tensor.shape
    if num_paths == 0:
        # No candidate paths at all
        return torch.tensor(0.0, device=device)

    # Edges per path & valid mask (topological validity)
    edge_start = paths_tensor[:, :-1]                       # [num_paths, path_len-1]
    edge_end   = paths_tensor[:,  1:]                       # [num_paths, path_len-1]
    valid_mask = (edge_start >= 0) & (edge_end >= 0)        # [num_paths, path_len-1]

    # Index for bands
    B_idx = torch.arange(B, device=device)[:, None, None]   # [B,1,1]
    es = edge_start.unsqueeze(0).expand(B, -1, -1)          # [B, num_paths, path_len-1]
    ee = edge_end.unsqueeze(0).expand(B, -1, -1)

    # Gather values along path edges
    h_values = h_arr[B_idx, es, ee]     # [B, num_paths, path_len-1], complex
    p_values = p_arr[B_idx, es, ee]     # [B, num_paths, path_len-1], real

    # SNR and per-link rates
    snr = (h_values.abs() * p_values) ** 2 / (sigma ** 2)
    snr = torch.nan_to_num(snr, nan=0.0, posinf=0.0, neginf=0.0)
    link_rates = torch.log2(1.0 + snr)                      # [B, num_paths, path_len-1]

    # Base mask for "structurally valid" edges (not padding)
    base_mask = valid_mask.unsqueeze(0)                     # [B, num_paths, path_len-1]

    if ignore_zero_edges:
        # Only edges that are both valid and have p > threshold are "active" for the min
        active_mask = base_mask & (p_values > power_threshold)

        # Mask everything else with +inf so min/soft-min ignores them
        masked_link_rates = torch.where(
            active_mask,
            link_rates,
            torch.full_like(link_rates, float("inf")),
        )

        if tau and tau > 0.0:
            # soft-min over active edges
            path_min_rates = (-1.0 / tau) * torch.logsumexp(-tau * masked_link_rates, dim=2)
        else:
            path_min_rates, _ = masked_link_rates.min(dim=2)  # [B, num_paths]

        # Paths with no active edges at all -> set to 0 (so they never beat real paths)
        all_inactive = ~active_mask.any(dim=2)                # [B, num_paths]
        path_min_rates = torch.where(
            all_inactive,
            torch.zeros_like(path_min_rates),
            path_min_rates,
        )
    else:
        # Original behavior: ignore only padding edges, but zero-power edges DO count in the min
        masked_link_rates = torch.where(
            base_mask,
            link_rates,
            torch.full_like(link_rates, float("inf")),
        )

        if tau and tau > 0.0:
            path_min_rates = (-1.0 / tau) * torch.logsumexp(-tau * masked_link_rates, dim=2)
        else:
            path_min_rates, _ = masked_link_rates.min(dim=2)  # [B, num_paths]

    # Best path per band (max over paths)
    max_path_rates, _ = path_min_rates.max(dim=1)  # [B]

    if per_band:
        return max_path_rates
    return max_path_rates.mean()


def objective_single_wrapper(
    links_mat,
    P,
    sigma_noise,
    paths,
    B,
    tau: float = 0.0,
    eps: float = 1e-12,
    per_band: bool = False,
    **kwargs,
):
    """
    Wrapper around calc_sum_rate for the single unicast problem.
    """
    return calc_sum_rate(
        h_arr=links_mat,
        p_arr=P,
        sigma=sigma_noise,
        paths_tensor=paths,
        B=B,
        tau=tau,
        eps=eps,
        per_band=per_band,
    )

