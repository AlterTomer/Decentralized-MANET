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
    ):
        super().__init__()

        self.n_nodes = int(n_nodes)
        self.n_bands = int(n_bands)
        self.K = int(K)
        self.problem = problem.lower()
        self.multi_like = self.problem in {"multi", "converge", "multiunicast"}

        # Input = flattened [Re(H), Im(H)], where H has shape [B, n, n].
        input_dim = 2 * self.n_bands * self.n_nodes * self.n_nodes

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

    def forward(self, h: torch.Tensor):
        """
        Forward pass.

        Parameters
        ----------
        h : torch.Tensor
            Complex CSI tensor with shape [B, n, n] or [batch, B, n, n].

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

        # Softplus enforces smooth non-negative amplitudes.
        out = F.softplus(self.net(x))

        if self.multi_like:
            return out.reshape(batch_size, self.n_bands, self.K, self.n_nodes, self.n_nodes)

        return out.reshape(batch_size, self.n_bands, self.n_nodes, self.n_nodes)