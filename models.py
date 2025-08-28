from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

# ---------------------------------------------- GatedGCN --------------------------------------------------------------
class GatedGCNLayer(MessagePassing):
    r"""
    Node-centric, edge-conditioned (FiLM) GCN layer with PreNorm + PostNorm + residuals.

    Updates:
      • Node features x ∈ ℝ^{n×D_n}
      • Edge features e ∈ ℝ^{E×D_e}

    Normalization:
      • PreNorm: LayerNorm before any linear/MLP computation.
      • PostNorm: LayerNorm after residual connection.
      • Residuals only if input/output dims match.
    """
    def __init__(self,
                 node_in_dim: int,
                 node_out_dim: int,
                 edge_in_dim: int,
                 edge_out_dim: int,
                 dropout: float = 0.0):
        super().__init__(aggr='add', node_dim=0)

        self.node_in_dim = node_in_dim
        self.node_out_dim = node_out_dim
        self.edge_in_dim = edge_in_dim
        self.edge_out_dim = edge_out_dim
        self.dropout = dropout

        # ---- PreNorms ----
        self.node_prenorm = nn.LayerNorm(node_in_dim)
        self.edge_prenorm = nn.LayerNorm(edge_in_dim)

        # ---- Node message transform ----
        self.W_msg = nn.Linear(node_in_dim, node_out_dim, bias=False)

        # ---- Edge-conditioned FiLM ----
        self.film_scale = nn.Linear(edge_out_dim, node_out_dim)
        self.film_shift = nn.Linear(edge_out_dim, node_out_dim)

        # ---- Node FFN ----
        self.node_ff = nn.Sequential(
            nn.Linear(node_out_dim, node_out_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(node_out_dim, node_out_dim),
        )
        self.node_residual = (node_in_dim == node_out_dim)
        self.node_postnorm = nn.LayerNorm(node_out_dim)

        # ---- Edge refinement MLP ----
        self.edge_refine_mlp = nn.Sequential(
            nn.Linear(edge_in_dim + 2 * node_in_dim, edge_out_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(edge_out_dim, edge_out_dim),
        )
        self.edge_gate = nn.Sigmoid()
        self.edge_residual = (edge_in_dim == edge_out_dim)
        self.edge_postnorm = nn.LayerNorm(edge_out_dim)

    def forward(self, x, edge_index, edge_attr):
        """
        Args:
            x (Tensor):             [n, D_n_in] node features.
            edge_index (LongTensor):[2, E] (src=j, dst=i).
            edge_attr (Tensor):     [E, D_e_in] edge features.

        Returns:
            (x_out, e_out): ([n, D_n_out], [E, D_e_out])
        """
        # ---- PreNorm ----
        x_ln = self.node_prenorm(x)
        e_ln = self.edge_prenorm(edge_attr)

        j, i = edge_index
        x_j = x_ln[j]  # [E, D_n_in]
        x_i = x_ln[i]  # [E, D_n_in]

        # ---- Edge update ----
        edge_in = torch.cat([e_ln, x_j, x_i], dim=-1)  # [E, D_e_in + 2*D_n_in]
        e_delta = self.edge_refine_mlp(edge_in)        # [E, D_e_out]
        gate = self.edge_gate(e_delta)

        if self.edge_residual:
            e_out = edge_attr + gate * e_delta
        else:
            e_out = gate * e_delta

        # ---- PostNorm for edges ----
        e_out = self.edge_postnorm(e_out)

        # ---- Node messages (FiLM-modulated) ----
        scale = self.film_scale(e_out)  # [E, D_n_out]
        shift = self.film_shift(e_out)  # [E, D_n_out]
        base  = self.W_msg(x_j)         # [E, D_n_out]
        msg   = (1.0 + scale) * base + shift

        # ---- Aggregate messages ----
        x_aggr = self.propagate(edge_index, msg=msg)

        # ---- Post-aggregation FF + residual ----
        x_upd = self.node_ff(x_aggr)
        x_out = x + x_upd if self.node_residual else x_upd

        # ---- PostNorm for nodes ----
        x_out = self.node_postnorm(x_out)

        return x_out, e_out

    def message(self, msg):
        """Return the precomputed per-edge message."""
        return msg



# ------------------------------------------- Chained Net --------------------------------------------------------------
class ChainedGNN(nn.Module):
    r"""
    Node-centric Chained GNN for decentralized OFDM MANET power allocation.

    Overview per sample (variable n, fixed B):
      1) Build node features from P0 ∈ ℝ^{B×n×n}:
           x_out[i] = Σ_j P0[:, i, j] ∈ ℝ^B   (bandwise outgoing power)
           x_in[i]  = Σ_j P0[:, j, i] ∈ ℝ^B   (bandwise incoming power)
           role[i]  ∈ {Tx, Rx, Relay} one-hot (size 3)
           x_node[i] = concat[x_out[i], x_in[i], role[i]] ∈ ℝ^{2B+3}
      2) Run L FiLM/Gated message-passing layers (each does PreNorm + residuals).
      3) Jumping Knowledge (JK):
           - "max": elementwise max over {x^(0), …, x^(ℓ)} → fixed dim = node_hidden
           - "concat": concat[x^(0)||…||x^(ℓ)] then a small Linear projects to node_hidden.
      4) Shared per-edge decoder:
           dec_in = [ e^(ℓ)_{i→j},  x_i^{JK(ℓ)},  x_j^{JK(ℓ)} ]  → LayerNorm → MLP → Softplus
           p_edge ∈ ℝ^B, then scatter into dense [B, n, n].

    Returns:
        List[Tensor]: length L; each is a per-band power matrix [B, n, n].
    """

    def __init__(self,
                 num_layers: int,
                 B: int,
                 dropout: float = 0.2,
                 use_jk: bool = True,
                 jk_mode: str = "concat"):
        super().__init__()
        assert jk_mode in ("max", "concat"), "jk_mode must be 'max' or 'concat'"

        self.B = B
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_jk = use_jk
        self.jk_mode = jk_mode

        # Trunk dims
        node_hidden = B          # keep simple & band-aligned
        edge_hidden = B
        node_in_dim = 2 * B + 3  # [x_out||x_in||role]
        edge_in_dim = 2 * B      # CSI: [Re(H_b), Im(H_b)] for b=1..B

        # GNN trunk (assumes GatedGCNLayer is available in scope or imported at file top)
        self.layers = nn.ModuleList([
            GatedGCNLayer(
                node_in_dim=node_in_dim if i == 0 else node_hidden,
                node_out_dim=node_hidden,
                edge_in_dim=edge_in_dim if i == 0 else edge_hidden,
                edge_out_dim=edge_hidden,
                dropout=dropout
            )
            for i in range(num_layers)
        ])

        # JK projection heads for "concat" to keep decoder input fixed-size
        if self.use_jk and self.jk_mode == "concat":
            self.jk_projs = nn.ModuleList([
                nn.Linear((l + 1) * node_hidden, node_hidden) for l in range(num_layers)
            ])
        else:
            self.jk_projs = None

        # Shared decoder: input is always [edge_hidden + 2*node_hidden]
        dec_in_dim = edge_hidden + 2 * node_hidden
        self.dec_in_norm = nn.LayerNorm(dec_in_dim)  # <-- stabilizes decoder input scale
        # self.decoder = nn.Sequential(
        #     nn.Linear(dec_in_dim, B),
        #     nn.Softplus()  # nonnegative power
        # )
        self.head = nn.Linear(dec_in_dim, B)
        self.out_act = nn.Softplus()

    # ---------- feature builder ----------
    def _build_node_features(self, P0: torch.Tensor, tx_idx=None, rx_idx=None) -> torch.Tensor:
        """
        Build node features from initial power guess and roles.

        Args:
            P0:     [B, n, n] initial power guess
            tx_idx: int or None (index of Tx)
            rx_idx: int or None (index of Rx)

        Returns:
            x_node: [n, 2B+3] = [sum_out || sum_in || role(Tx,Rx,Relay)]
        """
        B, n, _ = P0.shape
        # bandwise outgoing/incoming
        x_out = P0.sum(dim=2).transpose(0, 1)  # [n, B]
        x_in  = P0.sum(dim=1).transpose(0, 1)  # [n, B]

        # roles one-hot
        role = torch.zeros(n, 3, device=P0.device, dtype=x_out.dtype)
        if tx_idx is not None and 0 <= int(tx_idx) < n:
            role[int(tx_idx), 0] = 1.0
        if rx_idx is not None and 0 <= int(rx_idx) < n:
            role[int(rx_idx), 1] = 1.0
        relay_mask = (role.sum(dim=1, keepdim=True) == 0.0).float()
        role[:, 2:3] = relay_mask  # third column = Relay

        return torch.cat([x_out, x_in, role], dim=-1)  # [n, 2B+3]

    # ---------- forward ----------
    def forward(self, data):
        """
        Args (PyG Data):
            data.x          : [B, n, n] initial power guess
            data.edge_index : [2, E] directed edges (src, dst)
            data.edge_attr  : [E, 2B] CSI real/imag per band
            data.tx, data.rx: optional ints (Tx/Rx indices)

        Returns:
            List[Tensor]: length L; each [B, n, n] per-band power for layer ℓ.
        """
        P0, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        B, n, _ = P0.shape
        src, dst = edge_index

        # Build initial node features
        tx_idx = int(data.tx) if hasattr(data, 'tx') else None
        rx_idx = int(data.rx) if hasattr(data, 'rx') else None
        x = self._build_node_features(P0, tx_idx=tx_idx, rx_idx=rx_idx)  # [n, 2B+3]

        xs_per_layer, es_per_layer = [], []
        e = edge_attr  # [E, 2B]

        # GNN trunk
        for layer in self.layers:
            x, e = layer(x, edge_index, e)
            x = F.relu(x)
            e = F.relu(e)
            if self.training and self.dropout > 0:
                x = F.dropout(x, p=self.dropout)
                e = F.dropout(e, p=self.dropout)
            xs_per_layer.append(x)
            es_per_layer.append(e)

        outputs = []
        for l in range(self.num_layers):
            # Node reps (with JK)
            if not self.use_jk:
                x_src = xs_per_layer[l][src]
                x_dst = xs_per_layer[l][dst]
            else:
                if self.jk_mode == "max":
                    x_stack = torch.stack(xs_per_layer[:l + 1], dim=0)  # [(l+1), n, node_hidden]
                    x_src = torch.max(x_stack[:, src, :], dim=0).values
                    x_dst = torch.max(x_stack[:, dst, :], dim=0).values
                else:  # "concat"
                    x_src_cat = torch.cat(xs_per_layer[:l + 1], dim=-1)[src]
                    x_dst_cat = torch.cat(xs_per_layer[:l + 1], dim=-1)[dst]
                    x_src = self.jk_projs[l](x_src_cat)
                    x_dst = self.jk_projs[l](x_dst_cat)

            e_l = es_per_layer[l]
            dec_in = torch.cat([e_l, x_src, x_dst], dim=-1)
            dec_in = self.dec_in_norm(dec_in)

            # Per-edge powers → scatter to dense
            p_edge = self.out_act(self.head(dec_in))  # Softplus
            p_full = P0.new_zeros((B, n, n))
            p_full[:, src, dst] = p_edge.t()
            outputs.append(p_full)

        return outputs


