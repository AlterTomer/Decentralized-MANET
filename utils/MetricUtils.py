import torch
import torch.nn.functional as F


def link_rate(h, p, interference_power, sigma, eps: float = 1e-12):
    """
    Computes the interference-aware link rate for a single frequency channel.

    Args:
        h: Desired channel gain.
        p: Desired power allocation.
        interference_power: Same-band neighbor interference power.
        sigma: Noise std, not variance.
        eps: Numerical floor.

    Returns:
        torch.Tensor: Link rate.
    """
    desired_power = (torch.abs(h) * p) ** 2
    sinr = desired_power / (sigma ** 2 + interference_power + eps)
    sinr = torch.nan_to_num(sinr, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.log2(1.0 + sinr)


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
    Fully vectorized interference-aware sum-rate over all paths and bands.

    Interference is computed per frequency band and only from neighboring
    transmitters of the receiving node. There is no cross-band interference.

    For link i -> j on band b:

        SINR[b,i,j] =
            |h[b,i,j]|^2 p[b,i,j]^2
            / (sigma^2 + sum_{k in N(j), k != i} |h[b,k,j]|^2 sum_l p[b,k,l]^2)

    Args:
        h_arr: [B, n, n] complex channel matrices.
        p_arr: [B, n, n] real, nonnegative power allocations.
        sigma: float or tensor noise std, not variance.
        paths_tensor: [num_paths, max_path_len] with -1 padding.
        B: Number of frequency bands.
        tau: If > 0, uses soft-min over path edges.
        eps: Numerical floor.
        per_band: If True, return [B]; otherwise return mean over bands.
        ignore_zero_edges: If True, ignore inactive edges in bottleneck min.
        power_threshold: Activity threshold for ignore_zero_edges.

    Returns:
        torch.Tensor scalar, or [B] if per_band=True.
    """
    device = h_arr.device

    if not isinstance(sigma, torch.Tensor):
        sigma = torch.tensor(sigma, dtype=torch.float32, device=device)
    sigma = torch.nan_to_num(sigma, nan=1.0, posinf=1.0, neginf=1.0).clamp_min(eps)

    h_arr = torch.nan_to_num(h_arr, nan=0.0, posinf=0.0, neginf=0.0)
    p_arr = p_arr.real
    p_arr = torch.nan_to_num(p_arr, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)

    num_paths, _ = paths_tensor.shape
    if num_paths == 0:
        return torch.tensor(0.0, device=device)

    edge_start = paths_tensor[:, :-1]
    edge_end = paths_tensor[:, 1:]
    valid_mask = (edge_start >= 0) & (edge_end >= 0)

    B_idx = torch.arange(B, device=device)[:, None, None]
    es = edge_start.unsqueeze(0).expand(B, -1, -1)
    ee = edge_end.unsqueeze(0).expand(B, -1, -1)

    # Desired received power on all directed links: [B, n, n]
    channel_power = h_arr.abs() ** 2
    desired_power_all = channel_power * (p_arr ** 2)

    # Same-band transmitted power per transmitter: [B, n]
    tx_power_per_node = (p_arr ** 2).sum(dim=2)

    # Neighbor mask: k interferes with receiver j only if k -> j exists.
    neighbor_mask = channel_power > eps

    # Total same-band received power at each receiver from neighboring transmitters: [B, n]
    rx_total_power = torch.einsum(
        "bkj,bk->bj",
        channel_power * neighbor_mask,
        tx_power_per_node,
    )

    # Interference for desired link i -> j excludes the desired signal itself.
    interference_all = rx_total_power[:, None, :] - desired_power_all
    interference_all = interference_all.clamp_min(0.0)

    desired_power_values = desired_power_all[B_idx, es, ee]
    interference_values = interference_all[B_idx, es, ee]

    sinr = desired_power_values / (sigma ** 2 + interference_values + eps)
    sinr = torch.nan_to_num(sinr, nan=0.0, posinf=0.0, neginf=0.0)

    link_rates = torch.log2(1.0 + sinr)

    base_mask = valid_mask.unsqueeze(0)

    if ignore_zero_edges:
        active_mask = base_mask & (p_arr[B_idx, es, ee] > power_threshold)

        masked_link_rates = torch.where(
            active_mask,
            link_rates,
            torch.full_like(link_rates, float("inf")),
        )

        if tau and tau > 0.0:
            path_min_rates = (-1.0 / tau) * torch.logsumexp(
                -tau * masked_link_rates,
                dim=2,
            )
        else:
            path_min_rates, _ = masked_link_rates.min(dim=2)

        all_inactive = ~active_mask.any(dim=2)
        path_min_rates = torch.where(
            all_inactive,
            torch.zeros_like(path_min_rates),
            path_min_rates,
        )
    else:
        masked_link_rates = torch.where(
            base_mask,
            link_rates,
            torch.full_like(link_rates, float("inf")),
        )

        if tau and tau > 0.0:
            path_min_rates = (-1.0 / tau) * torch.logsumexp(
                -tau * masked_link_rates,
                dim=2,
            )
        else:
            path_min_rates, _ = masked_link_rates.min(dim=2)

    max_path_rates, _ = path_min_rates.max(dim=1)

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
