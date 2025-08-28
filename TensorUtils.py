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


def normalize_power(p_arr, adj, eps=1e-12):
    """ Projects a power allocation matrix onto the constraint set:
    - For each row (i.e., each node), L2 norm of outgoing powers (masked by adjacency) <= 1.
     - Ensures stability for disconnected nodes by clamping norms.
      Args: p_arr (torch.Tensor): [B, n, n] power allocation tensor.
      adj (torch.Tensor): [n, n] binary adjacency matrix.
      eps (float): Small value to prevent division by zero.
    Returns: torch.Tensor: Normalized power allocation tensor.
    """

    B, n, _ = p_arr.shape
    device = p_arr.device

    # Create adjacency mask [n, n] → [n, B * n] → match p_flat shape
    mask = adj.bool().to(device)
    mask_f = mask.repeat(B, 1).T.reshape(n, B * n).float()

    # Flatten [B, n, n] to [n, B * n] so each row is outgoing power for a node across batches
    p_flat = p_arr.permute(1, 0, 2).reshape(n, B * n)

    # Apply mask
    p_masked = p_flat * mask_f

    # Compute L2 norm per row (with clamping to avoid div/0)
    norms = p_masked.norm(p=2, dim=1, keepdim=True)  # [n, 1]
    norms_clamped = norms.clamp(min=eps)

    # Scale down only if norm > 1
    scale = torch.minimum(torch.ones_like(norms), 1.0 / norms_clamped)

    # Apply scaling
    p_projected = p_masked * scale

    # Zero out any values not in adjacency (safety, in case scale * eps ≠ 0)
    p_projected = p_projected * mask_f

    # Reshape back to [B, n, n]
    return p_projected.reshape(n, B, n).permute(1, 0, 2).contiguous()


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