import torch

def objective_multicast(
    h: torch.Tensor,
    p: torch.Tensor,
    sigma,
    adj: torch.Tensor,
    subgraphs_per_band: list,
    eps: float = 1e-12,
    tau_min: float = 0.0,
    tau_max: float = 0.0,
    per_band: bool = False,
    ignore_zero_edges: bool = False,
    power_threshold: float = 1e-8,
) -> torch.Tensor:
    """
    Implements the multicast (multiple receivers, one shared message) objective.

        J(P) = mean_b  max_{S in S_b}  min_{(i->j) in S} R^{(b)}_{i->j}(P)

    where
        R^{(b)}_{i->j}(P) = log2(1 + |h^{(b)}_{i->j}|^2 * p^{(b)2}_{i->j} / sigma_b^2).

    subgraphs_per_band: list of length B; each element is a list of subgraphs,
        and each subgraph S is an [n, n] {0,1} mask; nonzero entries indicate
        edges (i->j) belonging to S.

    ignore_zero_edges:
        If True, when computing the min over edges in S, only consider edges
        with p^{(b)}_{i->j} > power_threshold. Subgraphs that contain no such
        edges are ignored. If an entire band has no subgraph with any active
        edge, that band contributes 0 to the mean.
    """
    # Shapes / device
    B, n, _ = h.shape
    device = h.device

    # Prepare sigma^2 per band
    sigma = torch.as_tensor(sigma, dtype=torch.float32, device=device).reshape(-1)
    if sigma.numel() == 1:
        sigma = sigma.repeat(B)
    sigma = sigma.clamp_min(eps)
    sigma2 = (sigma**2).view(B, 1, 1)  # [B,1,1]

    # Sanitize inputs and mask non-edges
    adj_mask = (adj != 0).to(p.dtype).to(device)           # [n,n]
    p = torch.nan_to_num(torch.real(p), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    h_abs2 = (torch.nan_to_num(h, nan=0.0).abs() ** 2) * adj_mask  # [B,n,n]
    p2 = (p ** 2) * adj_mask

    # Per-edge rates (no interference)
    snr = h_abs2 * p2 / sigma2  # [B,n,n]
    snr = torch.nan_to_num(snr, nan=0.0, posinf=0.0, neginf=0.0)
    R = torch.log2(1.0 + snr) * adj_mask  # [B,n,n]
    R = R.real

    # Aggregate: max over subgraphs of min-over-edges
    r_band = torch.zeros(B, device=device)

    for b in range(B):
        subgraphs = subgraphs_per_band[b]
        if not subgraphs:
            # No multicast structures at all in this band -> contributes 0
            r_band[b] = 0.0
            continue

        sub_vals = []

        for S in subgraphs:
            if (S is None) or (isinstance(S, torch.Tensor) and not torch.any(S)):
                continue

            edge_idx = S.nonzero(as_tuple=False)  # [E, 2]
            if edge_idx.numel() == 0:
                continue

            ii = edge_idx[:, 0]
            jj = edge_idx[:, 1]

            edge_rates = R[b, ii, jj]  # [|S|]

            if ignore_zero_edges:
                # Only keep edges that actually carry power on this band
                edge_p = p[b, ii, jj]
                active_mask = edge_p > power_threshold
                if not torch.any(active_mask):
                    # This subgraph has no active edges under p -> skip it
                    continue
                edge_rates = edge_rates[active_mask]

            # Now edge_rates contains the rates of the edges we consider
            if edge_rates.numel() == 0:
                continue

            if tau_min > 0.0:
                sub_val = (-1.0 / tau_min) * torch.logsumexp(-tau_min * edge_rates, dim=0)
            else:
                sub_val = edge_rates.min()
            sub_vals.append(sub_val)

        if len(sub_vals) == 0:
            # No subgraph in this band had any active edge (or any edges at all)
            # -> band contributes 0 to the mean.
            r_band[b] = 0.0
            continue

        sub_vals = torch.stack(sub_vals)  # [num_valid_subgraphs]

        if tau_max > 0.0:
            r_band[b] = (1.0 / tau_max) * torch.logsumexp(tau_max * sub_vals, dim=0)
        else:
            r_band[b] = sub_vals.max()

    return r_band if per_band else r_band.mean()


def objective_multicast_wrapper(
    links_mat,
    P,
    sigma_noise,
    subgraphs_per_band,
    B,
    tau_min: float = 0.0,
    tau_max: float = 0.0,
    per_band: bool = False,
    eps: float = 1e-12,
    **kwargs,
):
    return objective_multicast(
        h=links_mat,
        p=P,
        sigma=sigma_noise,
        adj=kwargs.get("adj_mat"),  # passed in centralized optimizer
        subgraphs_per_band=subgraphs_per_band,
        tau_min=tau_min,
        tau_max=tau_max,
        eps=eps,
        per_band=per_band,
    )

