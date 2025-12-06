import torch

#=======================================================================================================================
# Tensors and Normalization
#=======================================================================================================================
def create_normalized_tensor(m, n, mask=None, device=None):
    """
    Creates a random [m, n] tensor with each row normalized (L2 norm = 1).

    Args:
        m (int): Number of rows.
        n (int): Number of columns.
        mask (torch.Tensor): Optional binary mask to zero-out elements.
        device (torch.device): Target device.

    Returns:
        torch.Tensor: Normalized random tensor.
    """
    device = device or (mask.device if isinstance(mask, torch.Tensor) else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    A = torch.rand(m, n, device=device)
    if isinstance(mask, torch.Tensor):
        A *= mask.to(device)
    norm_A = torch.norm(A, dim=1, keepdim=True)
    normalized_A = A / norm_A
    return normalized_A


def normalize_power(p_arr: torch.Tensor, adj: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Project power allocation onto a per-node L2 ball (≤ 1), masked by adjacency.

    Supported shapes:
      • [B, n, n]       (bands, src, dst)
      • [B, K, n, n]    (bands, commodities, src, dst)

    Semantics: For each node i, the L2 norm of *all outgoing* powers
    across all active (band × [commodity] × neighbors) is ≤ 1.

    Args:
        p_arr: Tensor [B,n,n] or [B,K,n,n].
        adj:   Tensor [n,n] binary adjacency.
        eps:   Small floor to avoid division by zero.

    Returns:
        Tensor with the same shape as p_arr, row-wise scaled per source node.
    """
    device = p_arr.device
    adj_mask = adj.bool().to(device)  # [n,n]

    if p_arr.dim() == 3:
        B, n, _ = p_arr.shape
        mask_f = adj_mask.repeat(B, 1).T.reshape(n, B * n).float()
        p_flat = p_arr.permute(1, 0, 2).reshape(n, B * n)  # [n, B*n]
        p_masked = p_flat * mask_f
        norms = p_masked.norm(p=2, dim=1, keepdim=True).clamp_min(eps)
        scale = torch.minimum(torch.ones_like(norms), 1.0 / norms)
        p_proj = (p_masked * scale) * mask_f
        return p_proj.reshape(n, B, n).permute(1, 0, 2).contiguous()

    elif p_arr.dim() == 4:
        B, K, n, _ = p_arr.shape
        mask_f = adj_mask.repeat(B * K, 1).T.reshape(n, B * K * n).float()
        p_flat = p_arr.permute(2, 0, 1, 3).reshape(n, B * K * n)  # [n, B*K*n]
        p_masked = p_flat * mask_f
        norms = p_masked.norm(p=2, dim=1, keepdim=True).clamp_min(eps)
        scale = torch.minimum(torch.ones_like(norms), 1.0 / norms)
        p_proj = (p_masked * scale) * mask_f
        return p_proj.reshape(n, B, K, n).permute(1, 2, 0, 3).contiguous()

    else:
        raise ValueError(f"normalize_power: unsupported p_arr.dim() = {p_arr.dim()} (expected 3 or 4)")


def init_equal_power(p, adj):
    """
    Initialize the power tensor with equal power allocation
    per user and per frequency band based on adjacency.

    Args:
        p (torch.Tensor): Power tensor [B, n, n], modified in place.
        adj (torch.Tensor): Adjacency matrix [n, n], 0/1 values.

    Returns:
        torch.Tensor: Updated power tensor [B, n, n].
    """
    B, n, _ = p.shape
    device, dtype = adj.device, p.dtype

    # Create a mask to zero-out non-existing links
    mask = adj.bool().to(device=device)

    # Count outgoing links per transmitter
    num_links = adj.sum(dim=1).clamp(min=1)  # avoid division by zero

    # Compute equal power per existing link
    equal_power = 1.0 / torch.sqrt(num_links.float())  # L2-normalized
    equal_power = equal_power.to(device=device, dtype=dtype)

    # Broadcast to match p shape
    equal_power_expanded = equal_power.unsqueeze(0).unsqueeze(-1)  # [1, n, 1]
    p[:] = torch.zeros_like(p)
    p[:, :, :] = equal_power_expanded * mask.unsqueeze(0)

    return p