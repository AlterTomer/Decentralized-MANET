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


def calc_sum_rate(h_arr, p_arr, sigma, paths_tensor, B, tau=0.0, eps=1e-12):
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

    Returns:
        torch.Tensor scalar: average best-path rate across bands.
    """
    device = h_arr.device
    if not isinstance(sigma, torch.Tensor):
        sigma = torch.tensor(sigma, dtype=torch.float32, device=device)
    # Ensure sigma is finite & > 0
    sigma = torch.nan_to_num(sigma, nan=1.0, posinf=1.0, neginf=1.0).clamp_min(eps)

    # Sanitize inputs: NaN/Inf -> 0 so they contribute no rate
    h_arr = torch.nan_to_num(h_arr, nan=0.0, posinf=0.0, neginf=0.0)
    p_arr = torch.nan_to_num(p_arr, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)

    num_paths, _ = paths_tensor.shape
    if num_paths == 0:
        return torch.tensor(0.0, device=device)

    # Edges per path & valid mask
    edge_start = paths_tensor[:, :-1]
    edge_end   = paths_tensor[:,  1:]
    valid_mask = (edge_start >= 0) & (edge_end >= 0)  # [num_paths, path_len-1]

    # Index for bands
    B_idx = torch.arange(B, device=device)[:, None, None]  # [B,1,1]
    es = edge_start.unsqueeze(0).expand(B, -1, -1)         # [B, num_paths, path_len-1]
    ee = edge_end.unsqueeze(0).expand(B, -1, -1)

    # Gather values along path edges
    h_values = h_arr[B_idx, es, ee]     # [B, num_paths, path_len-1], complex
    p_values = p_arr[B_idx, es, ee]     # [B, num_paths, path_len-1], real

    # SNR and per-link rates; sanitize again after ops
    snr = (h_values.abs() * p_values) ** 2 / (sigma ** 2)
    snr = torch.nan_to_num(snr, nan=0.0, posinf=0.0, neginf=0.0)
    link_rates = torch.log2(1.0 + snr)

    # Mask padding edges with +inf so min() ignores them; keep valid edges unchanged
    link_rates = torch.where(
        valid_mask.unsqueeze(0),
        link_rates,
        torch.full_like(link_rates, float('inf'))
    )

    # Min over edges in each path (hard min or soft-min)
    if tau and tau > 0.0:
        # soft-min across edges
        path_min_rates = (-1.0 / tau) * torch.logsumexp(-tau * link_rates, dim=2)  # [B, num_paths]
    else:
        path_min_rates, _ = link_rates.min(dim=2)  # [B, num_paths]

    # Best path per band (max over paths), then average across bands
    max_path_rates, _ = path_min_rates.max(dim=1)  # [B]
    return max_path_rates.mean()


def cosine_p(pred_p: torch.Tensor,
             target_p: torch.Tensor,
             adj: torch.Tensor,
             eps: float = 1e-12,
             reduction: str = 'mean',
             treat_isolated_as_perfect: bool = True) -> torch.Tensor:
    """
    Per-node cosine loss over rows p[:, i, :], masked by adjacency.
    Each node i compares its outgoing powers (across bands and neighbors).

    Args:
        pred_p:   [B, n, n]
        target_p: [B, n, n]
        adj:      [n, n] boolean or {0,1}
        eps:      numerical stability
        reduction: 'mean' | 'sum' | 'none'
        treat_isolated_as_perfect: if True, nodes with no outgoing edges contribute 0 loss

    Returns:
        scalar if reduction != 'none', else [n] per-node losses
    """
    assert pred_p.shape == target_p.shape, "pred/target shapes must match [B,n,n]"
    B, n, _ = pred_p.shape

    # Broadcastable mask
    mask3 = (adj > 0).to(dtype=pred_p.dtype, device=pred_p.device).unsqueeze(0)  # [1,n,n]

    # Masked rows for each node i
    Pm = pred_p * mask3           # [B,n,n]
    Tm = target_p * mask3         # [B,n,n]

    # Sum over bands and destination nodes -> per-node scalars
    # dot_i = <Pm[:,i,:], Tm[:,i,:]> summed over both B and n dims
    dot = (Pm * Tm).sum(dim=(0, 2))                      # [n]
    p_norm = torch.sqrt((Pm * Pm).sum(dim=(0, 2)).clamp_min(eps))  # [n]
    t_norm = torch.sqrt((Tm * Tm).sum(dim=(0, 2)).clamp_min(eps))  # [n]

    # Identify isolated rows (no outgoing edges)
    deg = adj.sum(dim=1)   # [n]
    isolated = (deg == 0)

    # Cosine similarity per node
    # If you want sign-invariance, wrap dot in .abs(); otherwise, leave it raw
    cos = dot / (p_norm * t_norm)
    cos = cos.clamp(-1.0, 1.0)

    # Handle isolated rows
    if treat_isolated_as_perfect:
        cos = torch.where(isolated.to(cos.device), torch.ones_like(cos), cos)
    else:
        # Exclude isolated from reduction by masking later
        pass

    loss_per_node = 1.0 - cos  # [n]

    if reduction == 'none':
        return loss_per_node
    elif reduction == 'sum':
        if treat_isolated_as_perfect:
            return loss_per_node.sum()
        else:
            valid = (~isolated).to(loss_per_node.dtype).to(loss_per_node.device)
            return (loss_per_node * valid).sum()
    else:  # 'mean'
        if treat_isolated_as_perfect:
            return loss_per_node.mean()
        else:
            valid = (~isolated).to(loss_per_node.dtype).to(loss_per_node.device)
            denom = valid.sum().clamp_min(1)
            return (loss_per_node * valid).sum() / denom

def mse_p(pred_P, target_P, normalize=False):
    """Mean squared error between predicted and target power allocations."""
    if normalize:
        num = torch.norm(pred_P - target_P, p='fro') ** 2
        denom = torch.norm(target_P, p='fro') ** 2 + 1e-12
        return num / denom
    else:
        return torch.mean((pred_P - target_P) ** 2)


def hybrid_power_loss(pred_P, target_P, adj, alpha=0.75, normalize_mse=False):
    """
    Hybrid loss: alpha * cosine + (1-alpha) * MSE (default unnormalized).
    """
    cos_loss = cosine_p(pred_P, target_P, adj)
    mse_loss = mse_p(pred_P, target_P, normalize=normalize_mse)
    return alpha * cos_loss + (1 - alpha) * mse_loss
