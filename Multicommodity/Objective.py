import torch

def edge_rates_multicommodity(
    h: torch.Tensor,
    p: torch.Tensor,
    z: torch.Tensor,
    sigma,
    adj: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Compute interference-aware per-edge rates R^{(b,k)}_{i->j}(P,Z).

    Interference is same-band and neighbor-only.

    Args:
        h: [B, n, n] complex channel gains.
        p: [B, K, n, n] non-negative amplitudes.
        z: [B, K, n, n] activations in [0,1].
        sigma: float or [B] noise std per band.
        adj: [n, n] {0,1} adjacency.
        eps: numerical safety.

    Returns:
        R: [B, K, n, n] real, masked by adj.
    """
    assert h.dim() == 3 and p.dim() == 4 and z.dim() == 4, \
        "Shapes: h[B,n,n], p[B,K,n,n], z[B,K,n,n]"

    B, n, n2 = h.shape
    assert n == n2
    assert p.shape[0] == B and z.shape[0] == B
    assert p.shape[2:] == (n, n) and z.shape[2:] == (n, n)

    device = h.device

    adj_mask = (adj != 0).to(device=device, dtype=torch.float32)  # [n,n]

    h = torch.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)
    h_abs2 = h.abs().pow(2).to(torch.float32) * adj_mask  # [B,n,n]

    p = torch.nan_to_num(
        p.to(torch.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).clamp_min(0.0)

    z = torch.nan_to_num(
        z.to(torch.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).clamp(0.0, 1.0)

    p2z = p.pow(2) * z * adj_mask  # [B,K,n,n]

    # Desired signal power for each commodity/link: [B,K,n,n]
    sig = h_abs2[:, None, :, :] * p2z

    # Total same-band transmit power per node over all commodities and outgoing links:
    # tx_power[b, i] = sum_k sum_l p[b,k,i,l]^2 z[b,k,i,l]
    tx_power_per_node = p2z.sum(dim=(1, 3))  # [B,n]

    # Total same-band received power at each receiver j from neighboring transmitters i.
    rx_total_power = torch.einsum(
        "bij,bi->bj",
        h_abs2,
        tx_power_per_node,
    )  # [B,n]

    # Interference for commodity k on link i -> j:
    # total same-band neighbor received power at j minus the desired stream.
    interference = rx_total_power[:, None, None, :] - sig  # [B,K,n,n]
    interference = interference.clamp_min(0.0)

    sigma = torch.as_tensor(sigma, dtype=torch.float32, device=device).reshape(-1)
    if sigma.numel() == 1:
        sigma = sigma.repeat(B)

    noise2 = sigma.clamp_min(eps).pow(2)[:, None, None, None]  # [B,1,1,1]

    sinr = sig / (noise2 + interference + eps)
    sinr = torch.nan_to_num(sinr, nan=0.0, posinf=0.0, neginf=0.0)

    R = torch.log2(1.0 + sinr)
    R = R * adj_mask
    R = torch.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)

    return R.real


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
    reduce: str = "mean",
    per_band: bool = False,
    outage_as_neg_inf: bool = False,
    ignore_zero_edges: bool = False,
    power_threshold: float = 1e-8,
) -> torch.Tensor:
    """
    Interference-aware multicommodity objective.

        J(P, Z) = sum_b sum_k max_{path in Phi_{b,k}}
                  min_{(i->j) in path} R^{(b,k)}_{i->j}(P,Z)

    with optional soft-min over edges and soft-max over paths.
    """
    R = edge_rates_multicommodity(h, p, z, sigma, adj)
    B, K, n, _ = R.shape
    device = R.device

    p = p.to(torch.float32)

    r_bk = torch.zeros(B, K, device=device)

    for k in range(K):
        paths = paths_k[k]

        if paths is None or paths.numel() == 0:
            if outage_as_neg_inf:
                r_bk[:, k] = float("-inf")
            continue

        paths = paths.to(device)

        edge_start = paths[:, :-1]
        edge_end = paths[:, 1:]

        if edge_start.numel() == 0:
            if outage_as_neg_inf:
                r_bk[:, k] = float("-inf")
            continue

        valid_mask = (edge_start >= 0) & (edge_end >= 0)

        link_rates = R[:, k][:, edge_start, edge_end]  # [B,P_k,L-1]
        base_mask = valid_mask.unsqueeze(0)

        if ignore_zero_edges:
            p_k = p[:, k]
            p_edges = p_k[:, edge_start, edge_end]

            active_mask = base_mask & (p_edges > power_threshold)

            masked_link_rates = torch.where(
                active_mask,
                link_rates,
                torch.full_like(link_rates, float("inf")),
            )

            if tau_min and tau_min > 0.0:
                path_vals = (-1.0 / tau_min) * torch.logsumexp(
                    -tau_min * masked_link_rates,
                    dim=2,
                )
            else:
                path_vals, _ = masked_link_rates.min(dim=2)

            all_inactive = ~active_mask.any(dim=2)
            path_vals = torch.where(
                all_inactive,
                torch.zeros_like(path_vals),
                path_vals,
            )

        else:
            masked_link_rates = torch.where(
                base_mask,
                link_rates,
                torch.full_like(link_rates, float("inf")),
            )

            if tau_min and tau_min > 0.0:
                path_vals = (-1.0 / tau_min) * torch.logsumexp(
                    -tau_min * masked_link_rates,
                    dim=2,
                )
            else:
                path_vals, _ = masked_link_rates.min(dim=2)

        if tau_max and tau_max > 0.0:
            r_bk[:, k] = (1.0 / tau_max) * torch.logsumexp(
                tau_max * path_vals,
                dim=1,
            )
        else:
            r_bk[:, k], _ = path_vals.max(dim=1)

    if per_band:
        if reduce == "sum":
            return r_bk.sum(dim=1)
        return r_bk.mean(dim=1)

    if reduce == "sum":
        return r_bk.sum()

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
    Wrapper around objective_multicommodity for the 1→K multicommodity problem.
    Expects:
      - links_mat: h, shape [B, n, n] (or [B, K, n, n] if you designed it so)
      - P:         p, shape [B, K, n, n]
      - Z:         z, shape [B, K, n, n]
      - paths_k:   list length K of padded path tensors
      - adj_mat:   adjacency [n, n]
    """
    if Z is None:
        raise ValueError("objective_multicommodity_wrapper: Z must be provided for multicommodity.")
    if adj_mat is None:
        raise ValueError("objective_multicommodity_wrapper: adj_mat must be provided for multicommodity.")

    return objective_multicommodity(
        h=links_mat,
        p=P,
        z=Z,
        sigma=sigma_noise,
        adj=adj_mat,
        paths_k=paths_k,
        tau_min=tau_min,
        tau_max=tau_max,
        reduce="mean",
        per_band=per_band,
        outage_as_neg_inf=False,
    )


