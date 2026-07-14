import torch
import torch.nn as nn
import torch.nn.functional as F


class FFNPowerAllocator(nn.Module):
    """
    Fully connected power-allocation baseline.

    The model receives the complete dense CSI tensor, flattens it, and predicts
    raw non-negative power amplitudes on all possible directed edges.

    Parameters
    ----------
    n_nodes : int
        Number of MANET nodes, n.
    n_bands : int
        Number of frequency bands, B.
    K : int
        Number of messages/commodities for multi-message settings.
    problem : str
        Problem type: {"single", "multicast", "multi", "converge", "multiunicast"}.
    hidden_dim : int
        Width of each hidden layer.
    num_layers : int
        Total number of linear layers. The last one is the output layer.
    dropout : float
        Dropout probability used after hidden activations.
    use_layernorm : bool
        If True, applies LayerNorm after hidden linear layers.

    Input
    -----
    h : torch.Tensor
        Complex CSI tensor with shape [B, n, n] or [batch, B, n, n].

    Output
    ------
    torch.Tensor
        Raw non-negative power amplitudes:
            single/multicast: [batch, B, n, n]
            multi-like:       [batch, B, K, n, n]
    """

    def __init__(
        self,
        n_nodes: int,
        n_bands: int,
        K: int = 1,
        problem: str = "single",
        hidden_dim: int = 512,
        num_layers: int = 4,
        dropout: float = 0.1,
        use_layernorm: bool = True,
        noise_conditioning: bool = False,
    ):
        super().__init__()

        self.n_nodes = int(n_nodes)
        self.n_bands = int(n_bands)
        self.K = int(K)
        self.problem = problem.lower()
        self.multi_like = self.problem in {"multi", "converge", "multiunicast"}
        # noise_conditioning: append a global SNR feature (-log10(sigma^2)/5 ~ SNR/50) to the
        # flattened CSI input so the FFN can adapt its allocation to the operating SNR. Without
        # it the FFN is noise-blind (CSI carries no sigma) -> one fixed allocation at every SNR,
        # the same blind spot the GNN had. Off by default (existing checkpoints load unchanged).
        self.noise_conditioning = bool(noise_conditioning)

        # Input = flattened [Re(H), Im(H)], where H has shape [B, n, n] (+1 for the SNR feature).
        input_dim = 2 * self.n_bands * self.n_nodes * self.n_nodes + (1 if self.noise_conditioning else 0)

        # Output = one raw amplitude per band/edge, and per commodity if needed.
        if self.multi_like:
            output_dim = self.n_bands * self.K * self.n_nodes * self.n_nodes
        else:
            output_dim = self.n_bands * self.n_nodes * self.n_nodes

        layers = []
        dim = input_dim

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(dim, hidden_dim))
            if use_layernorm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            dim = hidden_dim

        layers.append(nn.Linear(dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, h: torch.Tensor, sigma=None):
        """
        Forward pass.

        Parameters
        ----------
        h : torch.Tensor
            Complex CSI tensor with shape [B, n, n] or [batch, B, n, n].
        sigma : float | torch.Tensor | None
            Noise std for the sample(s). Used only when noise_conditioning is enabled;
            appended as a global SNR feature. Ignored otherwise (safe to always pass).

        Returns
        -------
        torch.Tensor
            Raw non-negative power amplitudes:
                [batch, B, n, n] for single/multicast.
                [batch, B, K, n, n] for multi/converge/multiunicast.
        """
        if h.dim() == 3:
            h = h.unsqueeze(0)

        batch_size = h.shape[0]

        # Convert complex CSI into real-valued features.
        x = torch.cat([h.real, h.imag], dim=1)  # [batch, 2B, n, n]
        x = x.reshape(batch_size, -1)

        # Noise conditioning: append a global SNR feature (-log10(sigma^2)/5 ~ SNR_dB/50).
        if self.noise_conditioning:
            if sigma is None:
                snr = x.new_zeros(batch_size, 1)
            else:
                s = torch.as_tensor(sigma, device=x.device, dtype=x.dtype).reshape(-1)
                snr_val = -torch.log10(s * s + 1e-12) / 5.0
                snr = (snr_val.reshape(1, 1).expand(batch_size, 1)
                       if snr_val.numel() == 1 else snr_val.reshape(batch_size, 1))
            x = torch.cat([x, snr], dim=1)

        # Softplus enforces smooth non-negative amplitudes.
        out = F.softplus(self.net(x))

        if self.multi_like:
            return out.reshape(batch_size, self.n_bands, self.K, self.n_nodes, self.n_nodes)

        return out.reshape(batch_size, self.n_bands, self.n_nodes, self.n_nodes)