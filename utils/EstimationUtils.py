import torch

@torch.no_grad()
def lmmse_estimate_masked(
    Y: torch.Tensor,                 # [B, n, n] complex noisy observations (on masked edges)
    adj: torch.Tensor,               # [n, n] bool/0-1 adjacency mask
    noise_var: torch.Tensor | float, # scalar noise variance (per complex entry)
    prior_var: torch.Tensor | float, # [B] per-band prior var OR scalar
    pilots_M: int = 1,               # number of pilot symbols per link
    pilot_power: float = 1.0,        # power per pilot symbol
) -> torch.Tensor:
    """
    Mask-aware LMMSE: \hat{h} = alpha * Y with alpha = S_h / (S_h + S_w / (M*P_pilot))
    Off-mask entries are returned as exactly zero.

    Y, prior_var, noise_var can be on CPU or CUDA. Complex dtype supported.
    """
    device = Y.device
    B, n, _ = Y.shape

    # Ensure types
    adj_bool = (adj > 0).to(device=device)
    mask = adj_bool.unsqueeze(0).expand(B, n, n)  # [B,n,n]

    # Broadcast shapes for alpha (per-band or scalar)
    if isinstance(prior_var, torch.Tensor):
        prior_var = prior_var.to(device=device, dtype=torch.float32).view(-1)  # [B] or [1]
        if prior_var.numel() == 1:
            prior_var = prior_var.expand(B)
    else:
        prior_var = torch.tensor(prior_var, device=device, dtype=torch.float32).expand(B)  # [B]

    noise_var = torch.as_tensor(noise_var, device=device, dtype=torch.float32)

    eff_snr_scale = float(pilots_M) * float(pilot_power)           # M * P_pilot
    denom = prior_var + noise_var / max(eff_snr_scale, 1e-12)      # [B]
    alpha = (prior_var / denom).view(B, 1, 1)                      # [B,1,1] for broadcast

    # Apply only on masked edges; keep others at 0
    Y_masked = torch.where(mask, Y, torch.zeros((), device=device, dtype=Y.dtype))
    H_hat = alpha * Y_masked
    # Safety: keep off-mask exactly 0
    H_hat = torch.where(mask, H_hat, torch.zeros((), device=device, dtype=Y.dtype))
    return H_hat

@torch.no_grad()
def lmmse_from_truth_masked(
    H_true: torch.Tensor,            # [B, n, n] complex ground-truth
    adj: torch.Tensor,               # [n, n] bool/0-1 adjacency
    noise_var: torch.Tensor | float, # scalar noise variance (per complex entry)
    prior_var: torch.Tensor | float, # [B] per-band prior var OR scalar
    pilots_M: int = 1,
    pilot_power: float = 1.0,
    rng: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Simulate pilot observation Y = H + W_eff on masked edges, then run masked LMMSE.
    W_eff ~ CN(0, noise_var / (M * P_pilot)).
    """
    device = H_true.device
    B, n, _ = H_true.shape
    dtype = H_true.dtype

    adj_bool = (adj > 0).to(device=device)
    mask = adj_bool.unsqueeze(0).expand(B, n, n)

    eff_noise_var = torch.as_tensor(noise_var, device=device, dtype=torch.float32) / max(pilots_M * pilot_power, 1e-12)

    # Complex Gaussian noise: real/imag ~ N(0, eff_noise_var/2)
    std = (eff_noise_var / 2.0).sqrt()
    real = torch.zeros_like(H_true.real).normal_(0.0, float(std), generator=rng)
    imag = torch.zeros_like(H_true.imag).normal_(0.0, float(std), generator=rng)
    W = torch.complex(real, imag)

    # Observation only on masked edges
    Y = torch.where(mask, H_true + W, torch.zeros((), device=device, dtype=dtype))

    # LMMSE on Y
    return lmmse_estimate_masked(Y, adj, noise_var, prior_var, pilots_M, pilot_power)

@torch.no_grad()
def masked_band_variance_from_dataset(dataset, use_abs2=True) -> torch.Tensor:
    """
    Estimate per-band prior variance S_h[b] over masked edges from a dataset.
    Returns: [B] float32 tensor.
    """
    # Assume all samples have same B
    B = int(dataset[0].B)
    num = torch.zeros(B, dtype=torch.float64)
    den = torch.zeros(B, dtype=torch.float64)

    for d in dataset:
        H = d.links_matrix  # [B,n,n] complex
        adj = (d.adj_matrix > 0)
        m = adj.unsqueeze(0)  # [1,n,n]

        if use_abs2:
            vals = (H.abs() ** 2).to(torch.float64)   # power
        else:
            # For zero-mean Rayleigh, E[H]≈0, so var≈E[|H|^2]; this branch would compute var via mean-subtraction.
            vals = (H.abs() ** 2).to(torch.float64)

        num += vals.masked_select(m).view(B, -1).sum(dim=1)
        den += m.view(1, -1).expand(B, -1).sum(dim=1).to(torch.float64)

    prior_var = (num / den.clamp(min=1)).to(torch.float32)
    return prior_var  # [B]
