# from __future__ import annotations
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import MessagePassing
#
# # ---------------------------------------------- GatedGCN --------------------------------------------------------------
# class GatedGCNLayer(MessagePassing):
#     r"""
#     Node-centric, edge-conditioned (FiLM) GCN layer with PreNorm + PostNorm + residuals.
#
#     Updates:
#       • Node features x ∈ ℝ^{n×D_n}
#       • Edge features e ∈ ℝ^{E×D_e}
#
#     Normalization:
#       • PreNorm: LayerNorm before any linear/MLP computation.
#       • PostNorm: LayerNorm after residual connection.
#       • Residuals only if input/output dims match.
#     """
#     def __init__(self,
#                  node_in_dim: int,
#                  node_out_dim: int,
#                  edge_in_dim: int,
#                  edge_out_dim: int,
#                  dropout: float = 0.0):
#         super().__init__(aggr='add', node_dim=0)
#
#         self.node_in_dim = node_in_dim
#         self.node_out_dim = node_out_dim
#         self.edge_in_dim = edge_in_dim
#         self.edge_out_dim = edge_out_dim
#         self.dropout = dropout
#
#         # ---- PreNorms ----
#         self.node_prenorm = nn.LayerNorm(node_in_dim)
#         self.edge_prenorm = nn.LayerNorm(edge_in_dim)
#
#         # ---- Node message transform ----
#         self.W_msg = nn.Linear(node_in_dim, node_out_dim, bias=False)
#
#         # ---- Edge-conditioned FiLM ----
#         self.film_scale = nn.Linear(edge_out_dim, node_out_dim)
#         self.film_shift = nn.Linear(edge_out_dim, node_out_dim)
#
#         # ---- Node FFN ----
#         self.node_ff = nn.Sequential(
#             nn.Linear(node_out_dim, node_out_dim),
#             nn.LeakyReLU(0.1),
#             nn.Dropout(dropout),
#             nn.Linear(node_out_dim, node_out_dim),
#         )
#         self.node_residual = (node_in_dim == node_out_dim)
#         self.node_postnorm = nn.LayerNorm(node_out_dim)
#
#         # ---- Edge refinement MLP ----
#         self.edge_refine_mlp = nn.Sequential(
#             nn.Linear(edge_in_dim + 2 * node_in_dim, edge_out_dim),
#             nn.LeakyReLU(0.1),
#             nn.Dropout(dropout),
#             nn.Linear(edge_out_dim, edge_out_dim),
#         )
#         self.edge_gate = nn.Sigmoid()
#         self.edge_residual = (edge_in_dim == edge_out_dim)
#         self.edge_postnorm = nn.LayerNorm(edge_out_dim)
#
#     def forward(self, x, edge_index, edge_attr):
#         """
#         Args:
#             x (Tensor): [n, D_n_in] node features.
#             edge_index (LongTensor):[2, E] (src=j, dst=i).
#             edge_attr (Tensor): [E, D_e_in] edge features.
#
#         Returns:
#             (x_out, e_out): ([n, D_n_out], [E, D_e_out])
#         """
#         # ---- PreNorm ----
#         x_ln = self.node_prenorm(x)
#         e_ln = self.edge_prenorm(edge_attr)
#
#         j, i = edge_index
#         x_j = x_ln[j]  # [E, D_n_in]
#         x_i = x_ln[i]  # [E, D_n_in]
#
#         # ---- Edge update ----
#         edge_in = torch.cat([e_ln, x_j, x_i], dim=-1)  # [E, D_e_in + 2*D_n_in]
#         e_delta = self.edge_refine_mlp(edge_in)        # [E, D_e_out]
#         gate = self.edge_gate(e_delta)
#
#         if self.edge_residual:
#             e_out = edge_attr + gate * e_delta
#         else:
#             e_out = gate * e_delta
#
#         # ---- PostNorm for edges ----
#         e_out = self.edge_postnorm(e_out)
#
#         # ---- Node messages (FiLM-modulated) ----
#         scale = self.film_scale(e_out)  # [E, D_n_out]
#         shift = self.film_shift(e_out)  # [E, D_n_out]
#         base  = self.W_msg(x_j)         # [E, D_n_out]
#         msg   = (1.0 + scale) * base + shift
#
#         # ---- Aggregate messages ----
#         x_aggr = self.propagate(edge_index, msg=msg)
#
#         # ---- Post-aggregation FF + residual ----
#         x_upd = self.node_ff(x_aggr)
#         x_out = x + x_upd if self.node_residual else x_upd
#
#         # ---- PostNorm for nodes ----
#         x_out = self.node_postnorm(x_out)
#
#         return x_out, e_out
#
#     def message(self, msg):
#         """Return the precomputed per-edge message."""
#         return msg
#
#
#
# class CandidateRoutingHead(nn.Module):
#     r"""
#     Route-consistent routing tensor Z built from CANDIDATE PATHS (not per-edge decisions).
#
#     For each commodity k, band b, candidate path p_{k,m}, it scores the option (k,b,m) from
#     pooled edge embeddings + route CSI features, takes an (annealed) softmax over the (b,m)
#     options per commodity, and lays the probabilities onto edges by path incidence:
#
#         Z[b,k,i,j] = sum_m  pi[k,b,m] * 1{(i,j) in p_{k,m}}.
#
#     Because Z is assembled from whole feasible paths, it is route-consistent by construction:
#     a low-temperature / argmax selection puts power on ONE connected route per commodity and
#     exactly ZERO off it — the property per-edge softmax Z could never learn (the source of the
#     transmit-everywhere failure). Route features:
#       bottleneck gain (min |h|^2 on path), mean gain, length, and exposure
#       E = sum_{(i,j) in p} sum_{l != i} |h^(b)_{l,j}|^2  (receiver interference exposure).
#     """
#     N_ROUTE_FEATS = 4  # [bottleneck, mean_gain, length, exposure]
#
#     def __init__(self, edge_hidden: int, hidden: int = 64):
#         super().__init__()
#         self.scorer = nn.Sequential(
#             nn.Linear(edge_hidden + self.N_ROUTE_FEATS, hidden),
#             nn.LeakyReLU(0.1),
#             nn.Linear(hidden, 1),
#         )
#
#     def forward(self, e_edge, edge_index, links, adj, paths_k, B, n, K, tau, device):
#         """e_edge:[E,edge_hidden]  links:[B,n,n] complex  paths_k: list len K of [M,L] (-1 pad).
#         Returns Z: [B,K,n,n]."""
#         src, dst = edge_index
#         emap = {(int(src[i]), int(dst[i])): i for i in range(src.shape[0])}
#         gains = (links.abs() ** 2).to(torch.float32)                 # [B,n,n]
#         col_sum = gains.sum(dim=1)                                    # [B,n]  sum_l |h_{l,j}|^2
#         Z = torch.zeros(B, K, n, n, device=device)
#         tau = max(float(tau), 1e-6)
#
#         for k in range(K):
#             paths = paths_k[k]
#             if paths is None or paths.numel() == 0:
#                 continue
#             feats, opt_edges = [], []
#             for b in range(B):
#                 for row in range(paths.shape[0]):
#                     seq = [int(x) for x in paths[row].tolist() if int(x) >= 0]
#                     if len(seq) < 2:
#                         continue
#                     edges = list(zip(seq[:-1], seq[1:]))
#                     if any(e not in emap for e in edges):
#                         continue
#                     eidx = torch.tensor([emap[e] for e in edges], device=device)
#                     emb = e_edge[eidx].mean(dim=0)                    # [edge_hidden]
#                     ii = torch.tensor([e[0] for e in edges], device=device)
#                     jj = torch.tensor([e[1] for e in edges], device=device)
#                     g = gains[b, ii, jj]                             # [len]
#                     exposure = (col_sum[b, jj] - g).sum()            # receiver interference exposure
#                     rfeat = torch.stack([g.min(), g.mean(),
#                                          torch.tensor(float(len(edges)), device=device), exposure])
#                     feats.append(torch.cat([emb, rfeat]))
#                     opt_edges.append((b, edges))
#             if not feats:
#                 continue
#             scores = self.scorer(torch.stack(feats)).squeeze(-1)     # [n_opts]
#             if (not self.training) or tau < 1e-3:
#                 # eval (or near-zero temperature) -> HARD argmax: single route/band per
#                 # commodity, so power lands on ONE connected route (true concentration).
#                 pi = torch.zeros_like(scores)
#                 pi[int(scores.argmax())] = 1.0
#             else:
#                 pi = torch.softmax(scores / tau, dim=0)             # annealed soft over (b,path)
#             for (b, edges), p in zip(opt_edges, pi):
#                 for (i, j) in edges:
#                     Z[b, k, i, j] = Z[b, k, i, j] + p
#         return Z
#
#
# # ------------------------------------------- Chained Net --------------------------------------------------------------
#
# class ChainedGNN(nn.Module):
#     """
#     Node-centric Chained GNN for decentralized OFDM MANET power allocation.
#
#
#     Overview per sample (variable n, fixed B):
#       1) Build node features from P0 ∈ ℝ^{B×n×n}:
#            x_out[i] = Σ_j P0[:, i, j] ∈ ℝ^B   (band-wise outgoing power)
#            x_in[i]  = Σ_j P0[:, j, i] ∈ ℝ^B   (band-wise incoming power)
#            role[i]  ∈ {Tx, Rx, Relay} one-hot (size 3 for single, K+2 else)
#            x_node[i] = concat[x_out[i], x_in[i], role[i]] ∈ ℝ^{2B+3}
#       2) Run L FiLM/Gated message-passing layers (each does PreNorm + residuals).
#       3) Jumping Knowledge (JK):
#            - "max": elementwise max over {x^(0), …, x^(ℓ)} → fixed dim = node_hidden
#            - "concat": concat[x^(0)||…||x^(ℓ)] then a small Linear projects to node_hidden.
#       4) Shared per-edge decoder:
#            dec_in = [ e^(ℓ)_{i→j}, x_i^{JK(ℓ)}, x_j^{JK(ℓ)} ] → LayerNorm → MLP → Softplus.
#
#     Modes:
#       - problem="single": one Tx→Rx, outputs per layer: [B, n, n]
#       - problem="multicast": shared message to K receivers, outputs per layer: [B, n, n]
#       - problem="multi", "converge", "multiunicast": K distinct messages (commodities), outputs per layer: (P[ B,K,n,n ], Z[ B,K,n,n ])
#     """
#
#     def __init__(self,
#                  num_layers: int,
#                  B: int,
#                  K: int = 1,
#                  problem: str = "single",
#                  dropout: float = 0.2,
#                  use_jk: bool = True,
#                  jk_mode: str = "concat",
#                  p_head_bias_init: float = 0.0,
#                  z_null_bias_init: float = 0.0,
#                  z_mode: str = "edge"):
#         super().__init__()
#         # z_mode: "edge" = per-edge softmax routing (original); "candidate" = route-consistent
#         # Z assembled from candidate paths (multi family only). See CandidateRoutingHead.
#         self.z_mode = z_mode
#         assert jk_mode in ("max", "concat"), "jk_mode must be 'max' or 'concat'"
#         assert problem in {"single", "multicast", "multi", "converge", "multiunicast"}, "problem must be 'single'|'multicast'|'multi'|'converge'| 'multiunicast'"
#
#         self.B = B
#         self.K = K
#         self.problem = problem
#         self.num_layers = num_layers
#         self.dropout = dropout
#         self.use_jk = use_jk
#         self.jk_mode = jk_mode
#
#         # Trunk dims
#         node_hidden = B
#         edge_hidden = B
#         roles_dim = 3 if problem == "single" else K + 2
#         node_in_dim = 2 * B + roles_dim
#         edge_in_dim = 2 * B  # CSI: [Re(H_b), Im(H_b)] for b=1..B
#
#         # GNN trunk
#         self.layers = nn.ModuleList([
#             GatedGCNLayer(
#                 node_in_dim=node_in_dim if i == 0 else node_hidden,
#                 node_out_dim=node_hidden,
#                 edge_in_dim=edge_in_dim if i == 0 else edge_hidden,
#                 edge_out_dim=edge_hidden,
#                 dropout=dropout
#             )
#             for i in range(num_layers)
#         ])
#
#         # JK projection heads for "concat"
#         if self.use_jk and self.jk_mode == "concat":
#             self.jk_projs = nn.ModuleList([
#                 nn.Linear((l + 1) * node_hidden, node_hidden) for l in range(num_layers)
#             ])
#         else:
#             self.jk_projs = None
#
#         # Decoder
#         dec_in_dim = edge_hidden + 2 * node_hidden
#         self.dec_in_norm = nn.LayerNorm(dec_in_dim)
#
#         # p_head size depends on PROBLEM (not just K):
#         # - single/multicast → B
#         # - multi → B*K
#         if self.problem in {"multi", "converge", "multiunicast"}:
#             p_out = B * K
#         else:
#             p_out = B
#         self.p_head = nn.Linear(dec_in_dim, p_out)
#         self.p_act  = nn.Softplus()
#         # Positive bias on the power head starts every edge near full power, so after the
#         # (down-only) per-node projection each transmitting node saturates its budget from
#         # the start. The model then learns to BACK OFF where interference hurts, instead of
#         # fighting the weak high-SNR gradient to push power UP (the observed 1-hop
#         # under-powering). This is an INIT shift, not a loss term, so it cannot persistently
#         # fight the routing objective the way a utilization penalty did. 0.0 = default init.
#         if p_head_bias_init != 0.0:
#             nn.init.constant_(self.p_head.bias, float(p_head_bias_init))
#
#         # z_head is only used in multi (K commodities): per-band (K+1)-way (K targets + 1 null)
#         self.z_head = nn.Linear(dec_in_dim, B * (K + 1))
#         self.z_act  = nn.Softmax(dim=-1)
#         # Candidate-path routing head (z_mode="candidate", multi family only): builds a
#         # route-consistent Z from candidate paths instead of per-edge decisions.
#         self.multi_like = self.problem in {"multi", "converge", "multiunicast"}
#         if self.z_mode == "candidate" and self.multi_like:
#             self.route_head = CandidateRoutingHead(edge_hidden=edge_hidden)
#         # Annealed selection temperature for candidate routing (set per-epoch by the trainer;
#         # small -> near-hard single-route selection). Ignored in "edge" mode.
#         self.routing_tau = 1.0
#         # Positive bias on the NULL slot (index K per band) starts every edge routing to "off"
#         # (softmax favors null) instead of the diffuse uniform init that spreads ~1/(K+1) power
#         # to EVERY commodity on EVERY edge (the transmit-everywhere start). The model then
#         # learns to switch specific routes ON, rather than fighting a diffuse init to switch
#         # edges OFF. Init shift, not a loss term. 0.0 = default (uniform). Multi-family only.
#         if z_null_bias_init != 0.0 and self.problem in {"multi", "converge", "multiunicast"}:
#             with torch.no_grad():
#                 nn.init.zeros_(self.z_head.bias)
#                 self.z_head.bias.view(B, K + 1)[:, K] = float(z_null_bias_init)
#
#
#     # ---------- feature builder ----------
#     def _build_node_features(
#             self,
#             P0: torch.Tensor,
#             tx_idx=None,
#             rx_idx=None,
#             *,
#             problem: str = "single",
#             K=None,
#     ) -> torch.Tensor:
#         """
#         Construct node feature vectors from the current power allocation tensor and
#         the communication roles of nodes.
#         Each node feature is composed of three parts:
#         1. Outgoing power features
#            Sum of transmitted power from node i to all other nodes across each band.
#                x_out[i, b] = sum_j P0[b, i, j]
#         2. Incoming power features
#            Sum of received power at node i from all other nodes across each band.
#                x_in[i, b] = sum_j P0[b, j, i]
#         3. Role indicators
#            Binary indicators describing the communication role of each node,
#            determined by the communication problem type.
#         The final node feature vector is
#             x_i = [x_out_i, x_in_i, role_i]
#         with dimension:
#             2B + d_role
#         where B is the number of frequency bands and d_role depends on the problem.
#
#         Parameters
#         ----------
#         P0 : torch.Tensor
#             Initial power allocation tensor of shape [B, n, n], where
#                 B = number of frequency bands
#                 n = number of nodes
#             and P0[b, i, j] is the power allocated by node i to node j on band b.
#         tx_idx : int | list[int] | None
#             Index (or list of indices) of transmitter nodes.
#         rx_idx : int | list[int] | None
#             Index (or list of indices) of receiver nodes.
#         problem : str
#             Communication pattern defining how roles are encoded. Supported values:
#             'single'
#                 One transmitter and one receiver.
#                 Role layout:
#                     [Tx, Rx, Relay]
#                 Role dimension:
#                     3
#             'multicast' or 'multi'
#                 One transmitter sending to multiple receivers.
#                 Role layout:
#                     [Tx_group, Rx_1, ..., Rx_K, Relay]
#                 Role dimension:
#                     K + 2
#                 The transmitter column indicates all transmitting nodes,
#                 while each receiver has its own role column corresponding
#                 to a distinct message/commodity.
#             'converge'
#                 Multiple transmitters sending toward a single receiver
#                 (convergencecast).
#                 Role layout:
#                     [Tx_1, ..., Tx_K, Rx_group, Relay]
#                 Role dimension:
#                     K + 2
#                 Each transmitter corresponds to a distinct message source,
#                 while all receivers share the same receiver column.
#             'multiunicast'
#                 Multiple transmitter–receiver pairs, each carrying an
#                 independent message.
#                 Role layout:
#                     [Pair_1, ..., Pair_K, Endpoint, Relay]
#                 Role dimension:
#                     K + 2
#                 Column k represents message pair k. Both the transmitter
#                 and receiver of that pair activate the same column.
#                 The 'Endpoint' column indicates nodes participating in
#                 any Tx–Rx pair.
#
#             K : int | None
#                 Number of messages/commodities (used to determine the number
#                 of role columns). If None, it is inferred from tx_idx or rx_idx
#                 depending on the problem.
#
#             Returns
#             -------
#             torch.Tensor
#                 Node feature tensor of shape [n, 2B + d_role]
#
#             """
#
#         B, n, _ = P0.shape
#         x_out = P0.sum(dim=2).transpose(0, 1)  # [n, B]
#         x_in = P0.sum(dim=1).transpose(0, 1)  # [n, B]
#
#         def _to_list(idx):
#             if idx is None:
#                 return []
#             if isinstance(idx, (list, tuple)):
#                 return [int(i) for i in idx]
#             return [int(idx)]
#
#         tx_list = _to_list(tx_idx)
#         rx_list = _to_list(rx_idx)
#
#         if problem == "single":
#             role = torch.zeros(n, 3, device=P0.device, dtype=x_out.dtype)
#             # [Tx, Rx, Relay]
#
#             if len(tx_list) > 0:
#                 t = tx_list[0]
#                 if 0 <= t < n:
#                     role[t, 0] = 1.0
#
#             if len(rx_list) > 0:
#                 r = rx_list[0]
#                 if 0 <= r < n:
#                     role[r, 1] = 1.0
#
#             role[:, 2] = (role[:, :2].max(dim=1).values == 0).float()
#
#         elif problem in {"multicast", "multi"}:
#             K_model = int(K) if K is not None else len(rx_list)
#             role = torch.zeros(n, K_model + 2, device=P0.device, dtype=x_out.dtype)
#             # [Tx_group, Rx1, ..., RxK, Relay]
#
#             for t in tx_list:
#                 if 0 <= t < n:
#                     role[t, 0] = 1.0
#
#             K_eff = min(len(rx_list), K_model)
#             for k, r in enumerate(rx_list[:K_eff]):
#                 if 0 <= r < n:
#                     role[r, 1 + k] = 1.0
#
#             role[:, -1] = (role[:, :-1].max(dim=1).values == 0).float()
#
#         elif problem == "converge":
#             K_model = int(K) if K is not None else len(tx_list)
#             role = torch.zeros(n, K_model + 2, device=P0.device, dtype=x_out.dtype)
#             # [Tx1, ..., TxK, Rx_group, Relay]
#
#             K_eff = min(len(tx_list), K_model)
#             for k, t in enumerate(tx_list[:K_eff]):
#                 if 0 <= t < n:
#                     role[t, k] = 1.0
#
#             for r in rx_list:
#                 if 0 <= r < n:
#                     role[r, K_model] = 1.0
#
#             role[:, -1] = (role[:, :-1].max(dim=1).values == 0).float()
#
#         elif problem == "multiunicast":
#             if len(tx_list) != len(rx_list):
#                 raise ValueError(
#                     f"For problem='multiunicast', tx_idx and rx_idx must have the same length, "
#                     f"got len(tx_idx)={len(tx_list)} and len(rx_idx)={len(rx_list)}."
#                 )
#
#             K_model = int(K) if K is not None else len(tx_list)
#             role = torch.zeros(n, K_model + 2, device=P0.device, dtype=x_out.dtype)
#             # [Pair1, ..., PairK, IsSource, Relay]. The pair column k identifies WHICH
#             # commodity a node belongs to; the IsSource column encodes DIRECTION within
#             # the pair (set for the Tx only). Previously BOTH endpoints set this column,
#             # so Tx_k and Rx_k had identical roles and flow direction was invisible to the
#             # model — it could not tell which end to route from. Now Tx_k=[pair-k, source]
#             # and Rx_k=[pair-k], so the two endpoints are distinguishable.
#             K_eff = min(len(tx_list), K_model)
#             for k in range(K_eff):
#                 t = tx_list[k]
#                 r = rx_list[k]
#
#                 if 0 <= t < n:
#                     role[t, k] = 1.0
#                     role[t, K_model] = 1.0   # source (direction) bit — Tx only
#                 if 0 <= r < n:
#                     role[r, k] = 1.0         # receiver: pair membership only, no source bit
#
#             role[:, -1] = (role[:, :-1].max(dim=1).values == 0).float()
#
#         else:
#             raise ValueError(
#                 f"Invalid problem='{problem}'. "
#                 f"Expected one of: 'single', 'multicast', 'multi', 'converge', 'multiunicast'."
#             )
#
#         return torch.cat([x_out, x_in, role], dim=-1)
#
#     def forward(self, data, paths_k=None):
#         """
#         Forward pass of the MANET-GNN model.
#
#         paths_k: list length K of candidate-path tensors (multi family). REQUIRED when
#         z_mode="candidate" (used to build the route-consistent Z); ignored otherwise.
#
#         Inputs
#         ---------------
#         data.x : torch.Tensor
#             Initial power tensor. Its shape depends on the communication problem:
#
#             - problem == 'single':
#                   [B, n, n]
#             - problem == 'multicast':
#                   [B, n, n]
#             - problem in {'multi', 'converge', 'multiunicast'}:
#                   [B, K, n, n]
#
#             where:
#                 B = number of frequency bands,
#                 n = number of nodes,
#                 K = number of messages / commodities.
#
#         data.tx : int | list[int]
#             Transmitter index or indices, depending on the problem.
#
#         data.rx : int | list[int]
#             Receiver index or indices, depending on the problem.
#
#         data.edge_index : torch.Tensor
#             Graph connectivity tensor of shape [2, E], where E is the number of
#             directed edges.
#
#         data.edge_attr : torch.Tensor
#             Edge feature tensor of shape [E, 2B].
#
#         Returns
#         -------
#         list
#             A list containing one output per GNN layer.
#             - For problem in {'single', 'multicast'}:
#                   each element is a tuple (P, Z), where
#                       P : [B, n, n]
#                       Z : [B, n, n] - all ones
#             - For problem in {'multi', 'converge', 'multiunicast'}:
#                   each element is a tuple (P, Z), where
#                       P : [B, K, n, n]
#                       Z : [B, K, n, n] - learned
#
#         """
#         P0, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
#         problem = self.problem
#         K_eff = self.K
#
#         # Shapes depend on problem
#         if problem in {"multi", "converge", "multiunicast"}:
#             assert P0.dim() == 4, f"For problem='{problem}', data.x must be [B,K,n,n]"
#             B, K, n, _ = P0.shape
#             assert K == K_eff, "data.K must match data.x.shape[1]"
#         else:
#             assert P0.dim() == 3, f"For problem='{problem}', data.x must be [B,n,n]"
#             B, n, _ = P0.shape
#
#         src, dst = edge_index
#
#         # ---- Build roles ----
#         tx_idx = getattr(data, 'tx', None)
#         rx_idx = getattr(data, 'rx', None)
#
#         x = self._build_node_features(
#             P0 if problem not in {"multi", "converge", "multiunicast"} else P0.sum(dim=1),
#             tx_idx=tx_idx,
#             rx_idx=rx_idx,
#             problem=problem,
#             K=K_eff if problem != "single" else None
#         )
#         # For multi-message problems, pass a [B,n,n] proxy obtained by summing over
#         # K so that x_out / x_in remain aggregate node-level traffic features.
#
#         # ---- GNN trunk ----
#         xs_per_layer, es_per_layer = [], []
#         e = edge_attr  # [E, 2B]
#         for layer in self.layers:
#             x, e = layer(x, edge_index, e)
#             x = F.relu(x)
#             e = F.relu(e)
#             if self.training and self.dropout > 0:
#                 x = F.dropout(x, p=self.dropout)
#                 e = F.dropout(e, p=self.dropout)
#             xs_per_layer.append(x)
#             es_per_layer.append(e)
#
#         # ---- Decode per layer ----
#         outputs = []
#         for l in range(self.num_layers):
#             # JK node reps
#             if not self.use_jk:
#                 x_src = xs_per_layer[l][src]
#                 x_dst = xs_per_layer[l][dst]
#             else:
#                 if self.jk_mode == "max":
#                     x_stack = torch.stack(xs_per_layer[:l + 1], dim=0)
#                     x_src = torch.max(x_stack[:, src, :], dim=0).values
#                     x_dst = torch.max(x_stack[:, dst, :], dim=0).values
#                 else:  # "concat"
#                     x_src_cat = torch.cat(xs_per_layer[:l + 1], dim=-1)[src]
#                     x_dst_cat = torch.cat(xs_per_layer[:l + 1], dim=-1)[dst]
#                     x_src = self.jk_projs[l](x_src_cat)
#                     x_dst = self.jk_projs[l](x_dst_cat)
#
#             e_l = es_per_layer[l]
#             dec_in = torch.cat([e_l, x_src, x_dst], dim=-1)  # [E, dec_in_dim]
#             dec_in = self.dec_in_norm(dec_in)
#             E = dec_in.size(0)
#
#             if problem in {"single", "multicast"}:
#                 # ----- single / multicast: output [B,n,n] -----
#                 p_edge = self.p_act(self.p_head(dec_in))  # [E, B]
#                 p_full = P0.new_zeros((B, n, n))
#                 p_full[:, src, dst] = p_edge.t()  # [B,E] -> scatter
#                 z_full = torch.ones_like(p_full)
#
#             else:
#                 # ----- multi / converge / multiunicast: output [B,K,n,n] -----
#                 p_edge_bk = self.p_act(self.p_head(dec_in)).view(E, B, K_eff)  # [E,B,K] amplitude A
#                 p_full = P0.new_zeros((B, K_eff, n, n))
#                 p_full[:, :, src, dst] = p_edge_bk.permute(1, 2, 0)  # [B,K,E] -> scatter
#
#                 if self.z_mode == "candidate":
#                     # Route-consistent Z from candidate paths (uses this layer's edge embeddings).
#                     if paths_k is None:
#                         raise ValueError("z_mode='candidate' requires paths_k in forward().")
#                     z_full = self.route_head(
#                         e_l, edge_index, data.links_matrix, data.adj_matrix, paths_k,
#                         B, n, K_eff, self.routing_tau, P0.device,
#                     )
#                 else:
#                     # Per-edge softmax routing: [E,B*(K+1)] -> softmax over K+1 -> take first K
#                     z_logits = self.z_head(dec_in).view(E, B, K_eff + 1)
#                     z_probs = self.z_act(z_logits)
#                     z_edge_bk = z_probs[..., :K_eff]  # [E,B,K]
#                     z_full = P0.new_zeros((B, K_eff, n, n))
#                     z_full[:, :, src, dst] = z_edge_bk.permute(1, 2, 0)
#
#             outputs.append((p_full, z_full))
#
#         return outputs
#
#
#
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
            x (Tensor): [n, D_n_in] node features.
            edge_index (LongTensor):[2, E] (src=j, dst=i).
            edge_attr (Tensor): [E, D_e_in] edge features.

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



class CandidateRoutingHead(nn.Module):
    r"""
    Fast route-consistent routing tensor Z built from candidate paths.

    This is a vectorized drop-in replacement for the original loop-heavy head.
    It keeps the same model idea and the same four route features:
      [bottleneck gain, mean gain, length, exposure].

    Decentralized inference support:
      pass sources_k and max_hops to restrict commodity k to paths that start
      at its source and contain at most max_hops hops. In simulation we still
      scatter the local decisions into a global Z tensor for evaluation, but the
      decision for each k is made independently from its local candidate set.
    """
    N_ROUTE_FEATS = 4  # [bottleneck, mean_gain, length, exposure]

    def __init__(self, edge_hidden: int, hidden: int = 64, *, exclude_desired_from_exposure: bool = True):
        super().__init__()
        self.exclude_desired_from_exposure = exclude_desired_from_exposure
        self.scorer = nn.Sequential(
            nn.Linear(edge_hidden + self.N_ROUTE_FEATS, hidden),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden, 1),
        )

    @staticmethod
    def _dense_edge_id(edge_index: torch.Tensor, n: int, device) -> torch.Tensor:
        """Map directed edge (i,j) -> PyG edge index, or -1 if absent."""
        src, dst = edge_index.to(device)
        edge_id = torch.full((n, n), -1, dtype=torch.long, device=device)
        edge_id[src, dst] = torch.arange(src.numel(), device=device, dtype=torch.long)
        return edge_id

    @staticmethod
    def _filter_paths_for_source(paths: torch.Tensor, source: int | None, max_hops: int | None) -> torch.Tensor:
        """Keep only paths that are source-local under the L-hop inference rule."""
        if paths.numel() == 0:
            return paths
        if paths.dim() == 1:
            paths = paths.unsqueeze(0)

        keep = torch.ones(paths.size(0), dtype=torch.bool, device=paths.device)

        if source is not None:
            keep &= (paths[:, 0] == int(source))

        if max_hops is not None:
            # Number of hops in the padded path = number of valid nodes - 1.
            num_valid_nodes = (paths >= 0).sum(dim=1)
            hops = num_valid_nodes - 1
            keep &= (hops <= int(max_hops))

        return paths[keep]

    def _score_and_scatter_one_commodity(
            self,
            *,
            e_edge: torch.Tensor,
            edge_id: torch.Tensor,
            gains: torch.Tensor,
            col_sum: torch.Tensor,
            paths: torch.Tensor,
            B: int,
            n: int,
            tau: float,
            hard: bool,
            source: int | None,
            max_hops: int | None,
    ) -> torch.Tensor:
        """Vectorized scoring/scattering for a single commodity k."""
        device = e_edge.device
        dtype = e_edge.dtype
        Zk = e_edge.new_zeros((B, n, n))

        if paths is None or paths.numel() == 0:
            return Zk

        paths = paths.to(device=device, dtype=torch.long)
        if paths.dim() == 1:
            paths = paths.unsqueeze(0)
        paths = self._filter_paths_for_source(paths, source=source, max_hops=max_hops)
        if paths.numel() == 0 or paths.shape[1] < 2:
            return Zk

        # Candidate edge sequences: u -> v, padded entries have -1.
        u = paths[:, :-1]  # [M,T]
        v = paths[:, 1:]   # [M,T]
        pair_mask = (u >= 0) & (v >= 0)  # true only for real path edges
        if not pair_mask.any():
            return Zk

        # Safe indexing for padded entries; they will be masked out later.
        u_safe = u.clamp(min=0, max=n - 1)
        v_safe = v.clamp(min=0, max=n - 1)

        eidx = edge_id[u_safe, v_safe]  # [M,T]
        edge_ok = pair_mask & (eidx >= 0)

        # A valid path must contain at least one real edge and every real edge must exist in edge_index.
        path_ok = ((edge_ok | ~pair_mask).all(dim=1)) & pair_mask.any(dim=1)
        if not path_ok.any():
            return Zk

        u_safe = u_safe[path_ok]
        v_safe = v_safe[path_ok]
        pair_mask = pair_mask[path_ok]
        eidx = eidx[path_ok]
        M, T = pair_mask.shape

        edge_len = pair_mask.sum(dim=1).clamp(min=1).to(dtype=dtype)  # [M]

        # Route embedding: mean of edge embeddings along the candidate path.
        eidx_safe = eidx.clamp(min=0)
        edge_emb = e_edge[eidx_safe]  # [M,T,D]
        mask_f = pair_mask.to(dtype=dtype).unsqueeze(-1)
        route_emb = (edge_emb * mask_f).sum(dim=1) / edge_len.unsqueeze(-1)  # [M,D]

        # Route CSI features for all B bands at once.
        # gains_flat[:, u*n+v] avoids slow Python loops and awkward advanced-index semantics.
        flat_uv = (u_safe * n + v_safe).reshape(-1)  # [M*T]
        gains_flat = gains.reshape(B, n * n)
        g = gains_flat[:, flat_uv].reshape(B, M, T)  # [B,M,T]

        col = col_sum[:, v_safe.reshape(-1)].reshape(B, M, T)  # [B,M,T]
        if self.exclude_desired_from_exposure:
            exposure_edge = col - g
        else:
            # Use this if the implemented interference denominator follows Eq. (7) literally.
            exposure_edge = col

        valid_bmt = pair_mask.unsqueeze(0)  # [1,M,T]
        g_masked_sum = (g * valid_bmt).sum(dim=-1)  # [B,M]
        mean_gain = g_masked_sum / edge_len.unsqueeze(0)
        bottleneck = g.masked_fill(~valid_bmt, float('inf')).amin(dim=-1)  # [B,M]
        exposure = (exposure_edge * valid_bmt).sum(dim=-1)  # [B,M]
        length_feat = edge_len.unsqueeze(0).expand(B, M)

        route_emb_bm = route_emb.unsqueeze(0).expand(B, M, route_emb.shape[-1])
        route_feats = torch.stack([bottleneck, mean_gain, length_feat, exposure], dim=-1)
        scorer_in = torch.cat([route_emb_bm, route_feats.to(dtype=dtype)], dim=-1)  # [B,M,D+4]
        scores = self.scorer(scorer_in.reshape(B * M, -1)).reshape(B, M)  # [B,M]

        tau = max(float(tau), 1e-6)
        # Per-band route selection (Eq. 9): each band independently picks its best route,
        # and R_k sums over bands. This keeps all B bands live for the commodity instead of
        # collapsing it onto a single (band, path) pair, which forfeits B-1 additive rate terms
        # and forces band collisions between commodities.
        if hard or (not self.training) or tau < 1e-3:
            idx = scores.argmax(dim=1)                 # [B] one route per band
            pi = torch.zeros_like(scores)
            pi.scatter_(1, idx.unsqueeze(1), 1.0)      # [B,M]
        else:
            pi = torch.softmax(scores / tau, dim=1)    # [B,M] per-band distribution over routes

        # Scatter candidate probabilities onto their path-incidence edges.
        b_idx = torch.arange(B, device=device).view(B, 1, 1).expand(B, M, T)
        u_idx = u_safe.view(1, M, T).expand(B, M, T)
        v_idx = v_safe.view(1, M, T).expand(B, M, T)
        vals = pi.unsqueeze(-1).expand(B, M, T) * pair_mask.to(dtype=dtype).unsqueeze(0)
        scatter_mask = pair_mask.unsqueeze(0).expand(B, M, T)

        Zk.index_put_(
            (b_idx[scatter_mask], u_idx[scatter_mask], v_idx[scatter_mask]),
            vals[scatter_mask],
            accumulate=True,
        )
        return Zk

    def forward(
            self,
            e_edge,
            edge_index,
            links,
            adj,
            paths_k,
            B,
            n,
            K,
            tau,
            device,
            *,
            sources_k=None,
            max_hops: int | None = None,
            hard: bool = False,
    ):
        """
        e_edge: [E, edge_hidden]
        links: [B,n,n] complex
        paths_k: list len K of [M,L] padded node sequences, or already-local paths per commodity

        Decentralized inference:
          - pass sources_k[k] = source node of commodity k
          - pass max_hops = L, the number of message-passing rounds / allowed local horizon
          - set hard=True for hard route-band selection in the forward pass
        """
        device = e_edge.device if device is None else device
        edge_id = self._dense_edge_id(edge_index, n, device=device)

        gains = (links.to(device).abs() ** 2).to(torch.float32)  # [B,n,n]

        # Match the interference model's neighborhood sum over l in N_j.
        # If adj is sparse, this prevents dense non-neighbor channels from contaminating exposure.
        if adj is not None:
            adj_f = adj.to(device=device, dtype=gains.dtype)
            gains_for_exposure = gains * adj_f.unsqueeze(0)  # [B,n,n], entries l->j only when adj[l,j]=1
            col_sum = gains_for_exposure.sum(dim=1)          # [B,n], sum_l in N_j |h_lj|^2
        else:
            col_sum = gains.sum(dim=1)

        Z = e_edge.new_zeros((B, K, n, n))
        for k in range(K):
            if paths_k is None or k >= len(paths_k):
                continue
            source = None
            if sources_k is not None and k < len(sources_k):
                source = sources_k[k]
            Z[:, k] = self._score_and_scatter_one_commodity(
                e_edge=e_edge,
                edge_id=edge_id,
                gains=gains,
                col_sum=col_sum,
                paths=paths_k[k],
                B=B,
                n=n,
                tau=tau,
                hard=hard,
                source=source,
                max_hops=max_hops,
            )
        return Z


# ------------------------------------------- Chained Net --------------------------------------------------------------

class ChainedGNN(nn.Module):
    """
    Node-centric Chained GNN for decentralized OFDM MANET power allocation.


    Overview per sample (variable n, fixed B):
      1) Build node features from P0 ∈ ℝ^{B×n×n}:
           x_out[i] = Σ_j P0[:, i, j] ∈ ℝ^B   (band-wise outgoing power)
           x_in[i]  = Σ_j P0[:, j, i] ∈ ℝ^B   (band-wise incoming power)
           role[i]  ∈ {Tx, Rx, Relay} one-hot (size 3 for single, K+2 else)
           x_node[i] = concat[x_out[i], x_in[i], role[i]] ∈ ℝ^{2B+3}
      2) Run L FiLM/Gated message-passing layers (each does PreNorm + residuals).
      3) Jumping Knowledge (JK):
           - "max": elementwise max over {x^(0), …, x^(ℓ)} → fixed dim = node_hidden
           - "concat": concat[x^(0)||…||x^(ℓ)] then a small Linear projects to node_hidden.
      4) Shared per-edge decoder:
           dec_in = [ e^(ℓ)_{i→j}, x_i^{JK(ℓ)}, x_j^{JK(ℓ)} ] → LayerNorm → MLP → Softplus.

    Modes:
      - problem="single": one Tx→Rx, outputs per layer: [B, n, n]
      - problem="multicast": shared message to K receivers, outputs per layer: [B, n, n]
      - problem="multi", "converge", "multiunicast": K distinct messages (commodities), outputs per layer: (P[ B,K,n,n ], Z[ B,K,n,n ])
    """

    def __init__(self,
                 num_layers: int,
                 B: int,
                 K: int = 1,
                 problem: str = "single",
                 dropout: float = 0.2,
                 use_jk: bool = True,
                 jk_mode: str = "concat",
                 p_head_bias_init: float = 0.0,
                 noise_conditioning: bool = False,
                 z_mode: str = "edge"):
        super().__init__()
        # z_mode: "edge" = per-edge softmax routing (original); "candidate" = route-consistent
        # Z assembled from candidate paths (multi family only). See CandidateRoutingHead.
        self.z_mode = z_mode
        # noise_conditioning: append a global SNR feature (-log10(sigma^2)/5 ~ SNR/50) to every
        # node so the amplitude head can adapt its per-band power split to the operating point.
        # Without it the forward pass is noise-blind (data.x/edge CSI/roles carry no sigma), so
        # the model emits ONE fixed allocation at every SNR and can't water-fill across bands at
        # high SNR the way the optimizer does. Off by default (existing checkpoints unchanged).
        self.noise_conditioning = bool(noise_conditioning)
        assert jk_mode in ("max", "concat"), "jk_mode must be 'max' or 'concat'"
        assert problem in {"single", "multicast", "multi", "converge", "multiunicast"}, "problem must be 'single'|'multicast'|'multi'|'converge'| 'multiunicast'"

        self.B = B
        self.K = K
        self.problem = problem
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_jk = use_jk
        self.jk_mode = jk_mode

        # Trunk dims
        node_hidden = B
        edge_hidden = B
        roles_dim = 3 if problem == "single" else K + 2
        node_in_dim = 2 * B + roles_dim + (1 if self.noise_conditioning else 0)
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
        # - multi → B*K
        if self.problem in {"multi", "converge", "multiunicast"}:
            p_out = B * K
        else:
            p_out = B
        self.p_head = nn.Linear(dec_in_dim, p_out)
        self.p_act  = nn.Softplus()
        # Positive bias on the power head starts every edge near full power, so after the
        # (down-only) per-node projection each transmitting node saturates its budget from
        # the start. The model then learns to BACK OFF where interference hurts, instead of
        # fighting the weak high-SNR gradient to push power UP (the observed 1-hop
        # under-powering). This is an INIT shift, not a loss term, so it cannot persistently
        # fight the routing objective the way a utilization penalty did. 0.0 = default init.
        if p_head_bias_init != 0.0:
            nn.init.constant_(self.p_head.bias, float(p_head_bias_init))

        # z_head is only used in multi (K commodities): per-band (K+1)-way (K targets + 1 null)
        self.z_head = nn.Linear(dec_in_dim, B * (K + 1))
        self.z_act  = nn.Softmax(dim=-1)
        # Candidate-path routing head (z_mode="candidate", multi family only): builds a
        # route-consistent Z from candidate paths instead of per-edge decisions.
        self.multi_like = self.problem in {"multi", "converge", "multiunicast"}
        if self.z_mode == "candidate" and self.multi_like:
            self.route_head = CandidateRoutingHead(edge_hidden=edge_hidden)
        # Annealed selection temperature for candidate routing (set per-epoch by the trainer;
        # small -> near-hard single-route selection). Ignored in "edge" mode.
        self.routing_tau = 1.0


    # ---------- feature builder ----------
    def _build_node_features(
            self,
            P0: torch.Tensor,
            tx_idx=None,
            rx_idx=None,
            *,
            problem: str = "single",
            K=None,
    ) -> torch.Tensor:
        """
        Construct node feature vectors from the current power allocation tensor and
        the communication roles of nodes.
        Each node feature is composed of three parts:
        1. Outgoing power features
           Sum of transmitted power from node i to all other nodes across each band.
               x_out[i, b] = sum_j P0[b, i, j]
        2. Incoming power features
           Sum of received power at node i from all other nodes across each band.
               x_in[i, b] = sum_j P0[b, j, i]
        3. Role indicators
           Binary indicators describing the communication role of each node,
           determined by the communication problem type.
        The final node feature vector is
            x_i = [x_out_i, x_in_i, role_i]
        with dimension:
            2B + d_role
        where B is the number of frequency bands and d_role depends on the problem.

        Parameters
        ----------
        P0 : torch.Tensor
            Initial power allocation tensor of shape [B, n, n], where
                B = number of frequency bands
                n = number of nodes
            and P0[b, i, j] is the power allocated by node i to node j on band b.
        tx_idx : int | list[int] | None
            Index (or list of indices) of transmitter nodes.
        rx_idx : int | list[int] | None
            Index (or list of indices) of receiver nodes.
        problem : str
            Communication pattern defining how roles are encoded. Supported values:
            'single'
                One transmitter and one receiver.
                Role layout:
                    [Tx, Rx, Relay]
                Role dimension:
                    3
            'multicast' or 'multi'
                One transmitter sending to multiple receivers.
                Role layout:
                    [Tx_group, Rx_1, ..., Rx_K, Relay]
                Role dimension:
                    K + 2
                The transmitter column indicates all transmitting nodes,
                while each receiver has its own role column corresponding
                to a distinct message/commodity.
            'converge'
                Multiple transmitters sending toward a single receiver
                (convergencecast).
                Role layout:
                    [Tx_1, ..., Tx_K, Rx_group, Relay]
                Role dimension:
                    K + 2
                Each transmitter corresponds to a distinct message source,
                while all receivers share the same receiver column.
            'multiunicast'
                Multiple transmitter–receiver pairs, each carrying an
                independent message.
                Role layout:
                    [Pair_1, ..., Pair_K, Endpoint, Relay]
                Role dimension:
                    K + 2
                Column k represents message pair k. Both the transmitter
                and receiver of that pair activate the same column.
                The 'Endpoint' column indicates nodes participating in
                any Tx–Rx pair.

            K : int | None
                Number of messages/commodities (used to determine the number
                of role columns). If None, it is inferred from tx_idx or rx_idx
                depending on the problem.

            Returns
            -------
            torch.Tensor
                Node feature tensor of shape [n, 2B + d_role]

            """

        B, n, _ = P0.shape
        x_out = P0.sum(dim=2).transpose(0, 1)  # [n, B]
        x_in = P0.sum(dim=1).transpose(0, 1)  # [n, B]

        def _to_list(idx):
            if idx is None:
                return []
            if isinstance(idx, (list, tuple)):
                return [int(i) for i in idx]
            return [int(idx)]

        tx_list = _to_list(tx_idx)
        rx_list = _to_list(rx_idx)

        if problem == "single":
            role = torch.zeros(n, 3, device=P0.device, dtype=x_out.dtype)
            # [Tx, Rx, Relay]

            if len(tx_list) > 0:
                t = tx_list[0]
                if 0 <= t < n:
                    role[t, 0] = 1.0

            if len(rx_list) > 0:
                r = rx_list[0]
                if 0 <= r < n:
                    role[r, 1] = 1.0

            role[:, 2] = (role[:, :2].max(dim=1).values == 0).float()

        elif problem in {"multicast", "multi"}:
            K_model = int(K) if K is not None else len(rx_list)
            role = torch.zeros(n, K_model + 2, device=P0.device, dtype=x_out.dtype)
            # [Tx_group, Rx1, ..., RxK, Relay]

            for t in tx_list:
                if 0 <= t < n:
                    role[t, 0] = 1.0

            K_eff = min(len(rx_list), K_model)
            for k, r in enumerate(rx_list[:K_eff]):
                if 0 <= r < n:
                    role[r, 1 + k] = 1.0

            role[:, -1] = (role[:, :-1].max(dim=1).values == 0).float()

        elif problem == "converge":
            K_model = int(K) if K is not None else len(tx_list)
            role = torch.zeros(n, K_model + 2, device=P0.device, dtype=x_out.dtype)
            # [Tx1, ..., TxK, Rx_group, Relay]

            K_eff = min(len(tx_list), K_model)
            for k, t in enumerate(tx_list[:K_eff]):
                if 0 <= t < n:
                    role[t, k] = 1.0

            for r in rx_list:
                if 0 <= r < n:
                    role[r, K_model] = 1.0

            role[:, -1] = (role[:, :-1].max(dim=1).values == 0).float()

        elif problem == "multiunicast":
            if len(tx_list) != len(rx_list):
                raise ValueError(
                    f"For problem='multiunicast', tx_idx and rx_idx must have the same length, "
                    f"got len(tx_idx)={len(tx_list)} and len(rx_idx)={len(rx_list)}."
                )

            K_model = int(K) if K is not None else len(tx_list)
            role = torch.zeros(n, K_model + 2, device=P0.device, dtype=x_out.dtype)
            # [Pair1, ..., PairK, IsSource, Relay]. The pair column k identifies WHICH
            # commodity a node belongs to; the IsSource column encodes DIRECTION within
            # the pair (set for the Tx only). Previously BOTH endpoints set this column,
            # so Tx_k and Rx_k had identical roles and flow direction was invisible to the
            # model — it could not tell which end to route from. Now Tx_k=[pair-k, source]
            # and Rx_k=[pair-k], so the two endpoints are distinguishable.
            K_eff = min(len(tx_list), K_model)
            for k in range(K_eff):
                t = tx_list[k]
                r = rx_list[k]

                if 0 <= t < n:
                    role[t, k] = 1.0
                    role[t, K_model] = 1.0   # source (direction) bit — Tx only
                if 0 <= r < n:
                    role[r, k] = 1.0         # receiver: pair membership only, no source bit

            role[:, -1] = (role[:, :-1].max(dim=1).values == 0).float()

        else:
            raise ValueError(
                f"Invalid problem='{problem}'. "
                f"Expected one of: 'single', 'multicast', 'multi', 'converge', 'multiunicast'."
            )

        return torch.cat([x_out, x_in, role], dim=-1)


    @staticmethod
    def _as_int_list(idx):
        if idx is None:
            return []
        if isinstance(idx, torch.Tensor):
            return [int(x) for x in idx.detach().cpu().view(-1).tolist()]
        if isinstance(idx, (list, tuple)):
            out = []
            for x in idx:
                if isinstance(x, torch.Tensor):
                    out.extend([int(v) for v in x.detach().cpu().view(-1).tolist()])
                else:
                    out.append(int(x))
            return out
        return [int(idx)]

    def _commodity_sources(self, tx_idx, rx_idx, K: int):
        """Source node for each commodity under the current communication framework."""
        tx_list = self._as_int_list(tx_idx)
        rx_list = self._as_int_list(rx_idx)

        if self.problem in {"multi"}:          # one Tx, K distinct messages to K receivers
            if not tx_list:
                return [None] * K
            return [tx_list[0]] * K

        if self.problem == "converge":         # K Tx nodes to one receiver
            return [tx_list[k] if k < len(tx_list) else None for k in range(K)]

        if self.problem == "multiunicast":     # K Tx-Rx pairs
            return [tx_list[k] if k < len(tx_list) else None for k in range(K)]

        # Candidate routing is not normally used for single/multicast in this implementation.
        if tx_list:
            return [tx_list[0]] * K
        return [None] * K

    def forward(
            self,
            data,
            paths_k=None,
            *,
            decentralized_inference: bool = False,
            max_route_hops: int | None = None,
            local_candidate_routing: bool = False,
            hard_candidate_routing: bool | None = None,
    ):
        """
        Forward pass of the MANET-GNN model.

        paths_k: list length K of candidate-path tensors (multi family). REQUIRED when
        z_mode="candidate" (used to build the route-consistent Z); ignored otherwise.

        Inputs
        ---------------
        data.x : torch.Tensor
            Initial power tensor. Its shape depends on the communication problem:

            - problem == 'single':
                  [B, n, n]
            - problem == 'multicast':
                  [B, n, n]
            - problem in {'multi', 'converge', 'multiunicast'}:
                  [B, K, n, n]

            where:
                B = number of frequency bands,
                n = number of nodes,
                K = number of messages / commodities.

        data.tx : int | list[int]
            Transmitter index or indices, depending on the problem.

        data.rx : int | list[int]
            Receiver index or indices, depending on the problem.

        data.edge_index : torch.Tensor
            Graph connectivity tensor of shape [2, E], where E is the number of
            directed edges.

        data.edge_attr : torch.Tensor
            Edge feature tensor of shape [E, 2B].

        Returns
        -------
        list
            A list containing one output per GNN layer.
            - For problem in {'single', 'multicast'}:
                  each element is a tuple (P, Z), where
                      P : [B, n, n]
                      Z : [B, n, n] - all ones
            - For problem in {'multi', 'converge', 'multiunicast'}:
                  each element is a tuple (P, Z), where
                      P : [B, K, n, n]
                      Z : [B, K, n, n] - learned

        """
        P0, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        problem = self.problem
        K_eff = self.K

        # Shapes depend on problem
        if problem in {"multi", "converge", "multiunicast"}:
            assert P0.dim() == 4, f"For problem='{problem}', data.x must be [B,K,n,n]"
            B, K, n, _ = P0.shape
            assert K == K_eff, "data.K must match data.x.shape[1]"
        else:
            assert P0.dim() == 3, f"For problem='{problem}', data.x must be [B,n,n]"
            B, n, _ = P0.shape

        src, dst = edge_index

        # ---- Build roles ----
        tx_idx = getattr(data, 'tx', None)
        rx_idx = getattr(data, 'rx', None)

        x = self._build_node_features(
            P0 if problem not in {"multi", "converge", "multiunicast"} else P0.sum(dim=1),
            tx_idx=tx_idx,
            rx_idx=rx_idx,
            problem=problem,
            K=K_eff if problem != "single" else None
        )
        # For multi-message problems, pass a [B,n,n] proxy obtained by summing over
        # K so that x_out / x_in remain aggregate node-level traffic features.

        # ---- Noise conditioning: append a global SNR feature to every node ----
        # sigma is the noise std; -log10(sigma^2) ~ SNR (dB)/10 when channel var ~ 1, scaled by
        # /5 to sit ~[0,1] over 0-50 dB. Broadcast to all n nodes (noise is global). This is the
        # ONLY sigma-dependent input, so it lets the model produce SNR-adaptive allocations.
        if self.noise_conditioning:
            sigma = getattr(data, "sigma", None)
            if sigma is None:
                snr_val = x.new_zeros(())
            else:
                s = torch.as_tensor(sigma, device=x.device, dtype=x.dtype).reshape(-1)[0]
                snr_val = (-torch.log10(s * s + 1e-12) / 5.0)
            snr_col = snr_val.expand(x.shape[0], 1)
            x = torch.cat([x, snr_col], dim=-1)

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

            e_l = es_per_layer[l]
            dec_in = torch.cat([e_l, x_src, x_dst], dim=-1)  # [E, dec_in_dim]
            dec_in = self.dec_in_norm(dec_in)
            E = dec_in.size(0)

            if problem in {"single", "multicast"}:
                # ----- single / multicast: output [B,n,n] -----
                p_edge = self.p_act(self.p_head(dec_in))  # [E, B]
                p_full = P0.new_zeros((B, n, n))
                p_full[:, src, dst] = p_edge.t()  # [B,E] -> scatter
                z_full = torch.ones_like(p_full)

            else:
                # ----- multi / converge / multiunicast: output [B,K,n,n] -----
                p_edge_bk = self.p_act(self.p_head(dec_in)).view(E, B, K_eff)  # [E,B,K] amplitude A
                p_full = P0.new_zeros((B, K_eff, n, n))
                p_full[:, :, src, dst] = p_edge_bk.permute(1, 2, 0)  # [B,K,E] -> scatter

                if self.z_mode == "candidate":
                    # Route-consistent Z from candidate paths (uses this layer's edge embeddings).
                    if paths_k is None:
                        raise ValueError("z_mode='candidate' requires paths_k in forward().")
                    sources_k = None
                    route_hops = None
                    # Decentralized inference should be local AND hard. Decentralized/local
                    # TRAINING should usually be local BUT soft, so the route logits still
                    # receive gradients. hard_candidate_routing=None preserves the default:
                    # hard only for decentralized_inference, soft for local training.
                    use_local_candidates = bool(decentralized_inference or local_candidate_routing)
                    force_hard = bool(decentralized_inference) if hard_candidate_routing is None else bool(hard_candidate_routing)
                    if use_local_candidates:
                        sources_k = self._commodity_sources(tx_idx, rx_idx, K_eff)
                        # At layer l, the available message-passing radius is l+1.
                        # For the final layer this equals num_layers. If max_route_hops is
                        # passed, it acts as an additional cap.
                        layer_hops = l + 1
                        route_hops = layer_hops if max_route_hops is None else min(int(max_route_hops), layer_hops)

                    z_full = self.route_head(
                        e_l, edge_index, data.links_matrix, data.adj_matrix, paths_k,
                        B, n, K_eff, self.routing_tau, P0.device,
                        sources_k=sources_k,
                        max_hops=route_hops,
                        hard=force_hard,
                    )
                else:
                    # Per-edge softmax routing: [E,B*(K+1)] -> softmax over K+1 -> take first K
                    z_logits = self.z_head(dec_in).view(E, B, K_eff + 1)
                    z_probs = self.z_act(z_logits)
                    z_edge_bk = z_probs[..., :K_eff]  # [E,B,K]
                    z_full = P0.new_zeros((B, K_eff, n, n))
                    z_full[:, :, src, dst] = z_edge_bk.permute(1, 2, 0)

            outputs.append((p_full, z_full))

        return outputs



