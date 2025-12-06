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
    """
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
           dec_in = [ e^(ℓ)_{i→j},  x_i^{JK(ℓ)},  x_j^{JK(ℓ)} ]  → LayerNorm → MLP → Softplus.

    Modes:
      - problem="single":    one Tx→Rx, outputs per layer: [B, n, n]
      - problem="multicast": shared message to K receivers, outputs per layer: [B, n, n]
      - problem="multi":     K distinct messages (commodities), outputs per layer: (P[ B,K,n,n ], Z[ B,K,n,n ])
    """

    def __init__(self,
                 num_layers: int,
                 B: int,
                 K: int = 1,
                 problem: str = "single",     # NEW
                 dropout: float = 0.2,
                 use_jk: bool = True,
                 jk_mode: str = "concat"):
        super().__init__()
        assert jk_mode in ("max", "concat"), "jk_mode must be 'max' or 'concat'"
        assert problem in {"single", "multicast", "multi"}, "problem must be 'single'|'multicast'|'multi'"

        self.B = B
        self.K = K
        self.problem = problem        # NEW
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_jk = use_jk
        self.jk_mode = jk_mode

        # Trunk dims
        node_hidden = B
        edge_hidden = B
        # roles: per-receiver channels for multicast/multi, 3 channels for single
        roles_dim = (1 + K + 1) if problem in {"multicast", "multi"} and K > 0 else 3
        node_in_dim = 2 * B + roles_dim
        edge_in_dim = 2 * B  # CSI: [Re(H_b), Im(H_b)] for b=1..B

        # GNN trunk
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

        # JK projection heads for "concat"
        if self.use_jk and self.jk_mode == "concat":
            self.jk_projs = nn.ModuleList([
                nn.Linear((l + 1) * node_hidden, node_hidden) for l in range(num_layers)
            ])
        else:
            self.jk_projs = None

        # Decoder
        dec_in_dim = edge_hidden + 2 * node_hidden
        self.dec_in_norm = nn.LayerNorm(dec_in_dim)

        # p_head size depends on PROBLEM (not just K):
        # - single/multicast → B
        # - multi            → B*K
        if self.problem == "multi":
            p_out = B * K
        else:
            p_out = B
        self.p_head = nn.Linear(dec_in_dim, p_out)
        self.p_act  = nn.Softplus()

        # z_head is only used in multi (K commodities): per-band (K+1)-way (K targets + 1 null)
        if self.problem == "multi":
            self.z_head = nn.Linear(dec_in_dim, B * (K + 1))
            self.z_act  = nn.Softmax(dim=-1)
        else:
            self.z_head = None
            self.z_act  = None

    # ---------- feature builder ----------
    def _build_node_features(self, P0: torch.Tensor, tx_idx=None, rx_idx=None, *, per_receiver=False,
                             K=None) -> torch.Tensor:
        """
        If per_receiver=True, roles dim = (1 + K + 1); else roles dim = 3.
        K can be inferred from len(rx_idx) if not provided.
        """
        B, n, _ = P0.shape
        x_out = P0.sum(dim=2).transpose(0, 1)  # [n,B]
        x_in = P0.sum(dim=1).transpose(0, 1)  # [n,B]

        if per_receiver:
            K_model = int(K) if K is not None else self.K
            role = torch.zeros(n, 1 + K_model + 1, device=P0.device,
                               dtype=x_out.dtype)  # [Tx, Rx1..RxK_model, Relay]

            # Tx (always set if valid)
            if tx_idx is not None and 0 <= int(tx_idx) < n:
                role[int(tx_idx), 0] = 1.0

            # Fill up to K_eff receiver channels, leave the rest zero
            if rx_idx is not None:
                rx_list = rx_idx if isinstance(rx_idx, (list, tuple)) else [int(rx_idx)]
                K_eff = min(len(rx_list), K_model)
                for k, r in enumerate(rx_list[:K_eff]):
                    r = int(r)
                    if 0 <= r < n:
                        role[r, 1 + k] = 1.0

            # Relay = 1 − max over all previous role channels
            role[:, -1] = (role[:, :-1].max(dim=1).values == 0).float()
        else:
            role = torch.zeros(n, 3, device=P0.device, dtype=x_out.dtype)
            if tx_idx is not None and 0 <= int(tx_idx) < n:
                role[int(tx_idx), 0] = 1.0
            if rx_idx is not None:
                if isinstance(rx_idx, (list, tuple)):
                    for r in rx_idx:
                        r = int(r)
                        if 0 <= r < n:
                            role[r, 1] = 1.0
                else:
                    r = int(rx_idx)
                    if 0 <= r < n:
                        role[r, 1] = 1.0
            role[:, 2] = (role[:, :2].max(dim=1).values == 0).float()

        return torch.cat([x_out, x_in, role], dim=-1)

    def forward(self, data):
        """
        Expects:
          - data.x:
              * [B, n, n]   for problem in {"single", "multicast"}
              * [B, K, n, n] for problem == "multi"   (model can ignore its values if unused)
          - data.K (int), data.tx (int), data.rx (int | list[int])
          - data.edge_index: [2, E], data.edge_attr: [E, 2B]

        Returns:
          - single/multicast: List[Tensor], each [B, n, n]
          - multi:            List[Tuple[Tensor, Tensor]], each (P[ B,K,n,n ], Z[ B,K,n,n ])
        """
        P0, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        problem = self.problem
        K_eff = self.K

        # Shapes depend on problem:
        if problem == "multi":
            assert P0.dim() == 4, "For problem='multi', data.x must be [B,K,n,n]"
            B, K, n, _ = P0.shape
            assert K == K_eff, "data.K must match data.x.shape[1]"
        else:
            assert P0.dim() == 3, "For problem='single'/'multicast', data.x must be [B,n,n]"
            B, n, _ = P0.shape

        src, dst = edge_index

        # ---- Build roles (per-receiver for multicast/multi; simple 3-channel for single) ----
        tx_idx = int(data.tx) if hasattr(data, 'tx') else None
        rx_idx = getattr(data, 'rx', None)
        if isinstance(rx_idx, torch.Tensor):
            rx_idx = rx_idx.view(-1).tolist()

        use_per_receiver = (problem in {"multicast", "multi"}) and (K_eff > 0)
        x = self._build_node_features(
            P0 if problem != "multi" else P0.sum(dim=1),  # use [B,n,n]-shaped proxy for sum_out/in (aggregate across K)
            tx_idx=tx_idx,
            rx_idx=rx_idx,
            per_receiver=use_per_receiver,
            K=K_eff if use_per_receiver else None
        )
        # Note: For "multi", we pass a [B,n,n] proxy (sum over K) to keep x_out/x_in semantics consistent.

        # ---- GNN trunk ----
        xs_per_layer, es_per_layer = [], []
        e = edge_attr  # [E, 2B]
        for layer in self.layers:
            x, e = layer(x, edge_index, e)
            x = F.relu(x)
            e = F.relu(e)
            if self.training and self.dropout > 0:
                x = F.dropout(x, p=self.dropout)
                e = F.dropout(e, p=self.dropout)
            xs_per_layer.append(x)
            es_per_layer.append(e)

        # ---- Decode per layer ----
        outputs = []
        for l in range(self.num_layers):
            # JK node reps
            if not self.use_jk:
                x_src = xs_per_layer[l][src]
                x_dst = xs_per_layer[l][dst]
            else:
                if self.jk_mode == "max":
                    x_stack = torch.stack(xs_per_layer[:l + 1], dim=0)
                    x_src = torch.max(x_stack[:, src, :], dim=0).values
                    x_dst = torch.max(x_stack[:, dst, :], dim=0).values
                else:  # "concat"
                    x_src_cat = torch.cat(xs_per_layer[:l + 1], dim=-1)[src]
                    x_dst_cat = torch.cat(xs_per_layer[:l + 1], dim=-1)[dst]
                    x_src = self.jk_projs[l](x_src_cat)
                    x_dst = self.jk_projs[l](x_dst_cat)

            e_l   = es_per_layer[l]
            dec_in = torch.cat([e_l, x_src, x_dst], dim=-1)   # [E, dec_in_dim]
            dec_in = self.dec_in_norm(dec_in)
            E      = dec_in.size(0)

            if problem in {"single", "multicast"}:
                # ----- single & multicast: output [B,n,n] -----
                p_edge = self.p_act(self.p_head(dec_in))       # [E, B]
                p_full = (P0 if P0.dim()==3 else P0.sum(dim=1)).new_zeros((B, n, n))
                p_full[:, src, dst] = p_edge.t()               # [B,E] → scatter
                outputs.append(p_full)

            else:
                # ----- multi-commodity: output (P[ B,K,n,n ], Z[ B,K,n,n ]) -----
                # P: [E, B*K] → [E,B,K] → scatter to [B,K,n,n]
                p_edge_bk = self.p_act(self.p_head(dec_in)).view(E, B, K_eff)   # [E,B,K]

                # Z: [E, B*(K+1)] → [E,B,K+1] → softmax over K+1 → take first K
                z_logits = self.z_head(dec_in).view(E, B, K_eff + 1)            # [E,B,K+1]
                z_probs  = self.z_act(z_logits)                                 # [E,B,K+1]
                z_edge_bk = z_probs[..., :K_eff]                                 # [E,B,K]

                p_full = P0.new_zeros((B, K_eff, n, n))
                z_full = P0.new_zeros((B, K_eff, n, n))
                p_full[:, :, src, dst] = p_edge_bk.permute(1, 2, 0)             # [B,K,E] → scatter
                z_full[:, :, src, dst] = z_edge_bk.permute(1, 2, 0)

                outputs.append((p_full, z_full))

        return outputs
