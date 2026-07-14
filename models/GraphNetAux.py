import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.nn.utils import clip_grad_norm_
from utils.TensorUtils import normalize_power
from utils.MetricUtils import calc_sum_rate
from utils.PathUtils import find_all_paths, paths_to_tensor
from utils.EstimationUtils import lmmse_from_truth_masked, build_estimate_lookup
from Multicast.Objective import objective_multicast
from Multicast.SubGraphs import find_multicast_subgraphs
from Multicommodity.Objective import objective_multicommodity
#=======================================================================================================================
# Helpers
#=======================================================================================================================
def tau_linear(epoch: int, max_epochs: int, start: float = 2.0, end: float = 32.0) -> float:
    """Linearly increase τ from start → end."""
    t = min(max(epoch / max(1, max_epochs - 1), 0.0), 1.0)
    return start + t * (end - start)


def _compute_rates_per_layer(
    model,
    data,
    paths=None,
    subgraphs_per_band=None,
    paths_k=None,
    problem: str = "single",
    tau_min: float = 0.0,
    tau_max: float = 0.0,
    reduce: str = "fair",
    decentralized_inference: bool = False,
    max_route_hops: int | None = None,
    local_candidate_routing: bool = False,
    hard_candidate_routing: bool | None = None,
):
    """
    Forward pass -> per-layer power normalization -> objective value per layer.
    Args:
        model:
            GNN model; model.B is the number of bands; model.K may be > 1 for
            multi-message problems.
        data:
            Batch with at least:
                - adj_matrix [n, n]
                - links_matrix [B, n, n] (complex)
                - sigma (float or [B])
        paths:
            For problem='single':
            LongTensor of shape [num_paths, max_len], padded with -1.
        subgraphs_per_band:
            For problem='multicast':
            List of length B; each element is a list of subgraphs
            (e.g. edge lists or masks).
        paths_k:
            For problem in {'multi', 'converge', 'multiunicast'}:
            list of length K, where each item is a LongTensor of shape
            [num_paths_k, max_len], padded with -1.
        problem:
            One of:
                - 'single'
                - 'multicast'
                - 'multi'
                - 'converge'
                - 'multiunicast'
        tau_min:
            Soft-min temperature used in path/subgraph bottleneck aggregation.
        tau_max:
            Soft-max temperature used in path/subgraph aggregation.
            Kept for completeness.
        decentralized_inference:
            If True and model.z_mode == "candidate", the model is called in its
            source-local L-hop inference mode. Keep this False for centralized training.
        max_route_hops:
            Optional hard cap on candidate path length during decentralized inference.

    Returns:
        rates_per_layer:
            list[Tensor], each a scalar objective value for one layer.

        p_list:
            list[Tensor], each normalized power tensor:
                - [B, n, n] for {'single', 'multicast'}
                - [B, K, n, n] for {'multi', 'converge', 'multiunicast'}

        z_list:
            list[Tensor], each [B, K, n, n] for
            {'multi', 'converge', 'multiunicast'};
            empty list otherwise.
    """
    if problem == "single":
        if paths is None:
            raise ValueError("paths must be provided for problem='single'.")
    elif problem == "multicast":
        if subgraphs_per_band is None:
            raise ValueError("subgraphs_per_band must be provided for problem='multicast'.")
    elif problem in {"multi", "converge", "multiunicast"}:
        if paths_k is None:
            raise ValueError(f"paths_k must be provided for problem='{problem}'.")
    else:
        raise ValueError(
            "problem must be one of: 'single', 'multicast', 'multi', 'converge', 'multiunicast'."
        )

    try:
        outputs = model(
            data,
            paths_k=paths_k,
            decentralized_inference=decentralized_inference,
            max_route_hops=max_route_hops,
            local_candidate_routing=local_candidate_routing,
            hard_candidate_routing=hard_candidate_routing,
        )
    except TypeError:
        # Backward compatibility with older ChainedGNN.forward implementations.
        if decentralized_inference or max_route_hops is not None or local_candidate_routing or hard_candidate_routing is not None:
            raise
        outputs = model(data, paths_k=paths_k)
    rates_per_layer, p_list, z_list = [], [], []
    adj = data.adj_matrix
    h = data.links_matrix
    sig = data.sigma

    for out in outputs:
        P_raw, Z = out

        if problem in {"multi", "converge", "multiunicast"}:
            # Normalize per message / commodity: each slice P[:, k] is [B, n, n].
            # Objective.edge_rates_multicommodity uses p2z = p^2 * z.
            # Since we store/evaluate the scored allocation as (p=P_norm, z=ones),
            # the activation Z must be folded into the amplitude as sqrt(Z), so that
            # (P_raw * sqrt(Z))^2 equals P_raw^2 * Z before projection.
            # For hard candidate routing Z ∈ {0,1}, this is identical to multiplying by Z.
            Zc = Z.clamp(0, 1)
            P_eff = P_raw * torch.sqrt(Zc)
            P_norm = normalize_power(P_eff, adj)


            rate = objective_multicommodity(
                h=h.to(P_norm.device),
                p=P_norm,
                z=torch.ones_like(Z),
                sigma=sig.to(P_norm.device),
                adj=adj.to(P_norm.device),
                paths_k=paths_k,
                tau_min=tau_min,
                tau_max=tau_max,
                reduce=reduce,
                per_band=False,
            )
            # Store the *scored* allocation so downstream re-scoring (train/validate
            # on TRUE CSI, benchmarks) reproduces this exact objective. The scored
            # form is (p=P_norm, z=ones) — Z is already folded into P_norm via
            # sqrt(Z), and P_norm is projected onto the per-node power budget. Storing
            # the raw (P_eff, Z) instead made the multi training loss score an
            # un-normalized, Z^2-weighted allocation that violated the power
            # constraint and diverged from the benchmark objective (F3/F4/F5).
            p_list.append(P_norm)
            z_list.append(torch.ones_like(Z))

        else:
            P_norm = normalize_power(
                P_raw,
                adj=adj.to(P_raw.device),
                eps=1e-12
            )

            if problem == "single":
                rate = calc_sum_rate(
                    h_arr=h.to(P_norm.device),
                    p_arr=P_norm,
                    sigma=sig.to(P_norm.device),
                    paths_tensor=paths,
                    B=model.B,
                    tau_min=0.0,
                    tau_max=0.0,
                    eps=1e-12,
                    per_band=False,
                )
            else:  # multicast
                rate = objective_multicast(
                    h=h.to(P_norm.device),
                    p=P_norm,
                    sigma=sig.to(P_norm.device),
                    adj=adj.to(P_norm.device),
                    subgraphs_per_band=subgraphs_per_band,
                    eps=1e-12,
                    tau_min=tau_min,
                    tau_max=tau_max,
                    per_band=False,
                )
            p_list.append(P_norm)

        rates_per_layer.append(
            rate if torch.isfinite(rate)
            else torch.tensor(float("-inf"), device=h.device)
        )

    return rates_per_layer, p_list, z_list

#=======================================================================================================================
# ChainedNet
#=======================================================================================================================


def train_chained(
    model,
    loader,
    optimizer,
    epoch,
    *,
    batch_size: int = 1,
    mode: str = "single",          # "single" | "multicast" | "multi" | "converge" | "multiunicast"
    device=None,
    mono_weight: float = 0.0,
    sparsity_weight: float = 0.0,    # >0 adds a concentration penalty (see loss below)
    utilization_weight: float = 0.0, # >0 pushes ACTIVE tx nodes to saturate their power budget
    utilization_max_hops: int = 0,   # >0: only apply utilization when the route is <= this many hops
    struct_cache: dict = None,       # persistent {sample_id: structures} cache across epochs
    use_amp: bool = False,
    scaler=None,
    grad_clip=None,
    grad_accum_steps: int = 1,
    tau: float = 0.0,              # used as tau_min, tau_max if needed; hard max by default
    reduce: str = "fair",          # K>1 message reduction: "fair"=eq10 min | "sum"/"mean" throughput
    est_dataset=None,
    decentralized_inference: bool = False,  # Normally False during training; True mimics L-hop hard inference.
    max_route_hops: int | None = None,      # Optional cap for candidate paths in decentralized inference.
    local_candidate_routing: bool = False, # True during decentralized-compatible training: local paths but soft routing.
    hard_candidate_routing: bool | None = None, # None => hard only for decentralized_inference.
):
    """
    Unified trainer for chained MANET-GNN models under several communication modes.

    Args:
        model: torch.nn.Module
            ChainedGNN model producing a list of per-layer outputs.

        loader: torch.utils.data.DataLoader
            Yields graph batches with fields:
              - adj_matrix: [n, n] (bool/0-1)
              - links_matrix: [B, n, n] (complex64/complex128)
              - sigma: float or [B] (noise std per band)
              - tx: int or List[int]
              - rx: int or List[int]
              - optional sample_id: int (for estimated-CSI lookup)

        optimizer: torch.optim.Optimizer
            Optimizer over model parameters.

        epoch: int
            Current epoch index (for logging only).

    Keyword Args:
        mode: str = "single"
            Which communication objective to train:

              - "single":
                    One Tx -> one Rx.
                    Uses paths via find_all_paths + paths_to_tensor.
                    Objective: calc_sum_rate(...)

              - "multicast":
                    One Tx -> multiple Rx, same message.
                    Uses minimal multicast subgraphs via find_multicast_subgraphs.
                    Objective: objective_multicast(...)

              - "multi":
                    One Tx -> multiple Rx, different messages.
                    Builds one path set per receiver.
                    Objective: objective_multicommodity(...)

              - "converge":
                    Multiple Tx -> one Rx, different messages.
                    Builds one path set per transmitter.
                    Objective: objective_multicommodity(...)

              - "multiunicast":
                    Multiple Tx-Rx pairs, each carrying its own message.
                    Builds one path set per Tx-Rx pair.
                    Objective: objective_multicommodity(...)

        device: torch.device | None
            If None, inferred from model parameters.

        mono_weight: float = 0.0
            Weight of the monotonicity penalty across layers.
            Encourages rates[layer+1] >= rates[layer] + margin.

        use_amp: bool = False
            Enable PyTorch AMP autocast for the forward/backward pass.

        scaler: torch.cuda.amp.GradScaler | None
            Required if use_amp=True; ignored otherwise.

        grad_clip: float | None
            If set, clip gradient norm to this value before optimizer.step().

        grad_accum_steps: int = 1
            Gradient accumulation steps. Loss is divided internally.

        tau: float = 0.0
            Temperature parameter:
              - "single": forwarded to calc_sum_rate (soft-min over edges if > 0)
              - "multicast": used as tau_min in objective_multicast
              - {"multi", "converge", "multiunicast"}: used as tau_min in objective_multicommodity

        est_dataset: Optional[dict | EstimatedCSIDataset] = None
            If provided, forward pass uses estimated CSI (H_hat) looked up by sample_id,
            while the loss is computed under true CSI (H_true).

            - If dict: maps sample_id -> H_hat, shape [B, n, n] (complex)
            - If EstimatedCSIDataset: converted to a fast lookup via build_estimate_lookup()

    Returns:
        dict
            Training statistics with keys:
              - "loss": average loss over processed batches
              - "lr": current learning rate
    """
    assert mode in {"single", "multicast", "multi", "converge", "multiunicast"}

    device = next(model.parameters()).device if device is None else device
    model.train()
    optimizer.zero_grad(set_to_none=True)

    est_lookup = None
    if est_dataset is not None:
        est_lookup = est_dataset if isinstance(est_dataset, dict) else build_estimate_lookup(est_dataset)

    total_loss_val, num_batches = 0.0, 0

    for step, data in enumerate(loader):
        data = data.to(device)

        if batch_size == 1:
            if isinstance(data.rx, (list, tuple)) and len(data.rx) == 1 and isinstance(data.rx[0], (list, tuple)):
                data.rx = data.rx[0]
            if isinstance(data.tx, (list, tuple)) and len(data.tx) == 1 and isinstance(data.tx[0], (list, tuple)):
                data.tx = data.tx[0]

        # Per-sample routing structures are fixed across epochs (topology + tx/rx do
        # not change), so compute them once and reuse from struct_cache afterward.
        sid = int(getattr(data, "sample_id", step))
        _cached = struct_cache.get(sid) if struct_cache is not None else None

        min_hops = None  # shortest Tx->Rx hop count (single mode); gates hop-based penalties

        # ----- Build graph structures (cached per sample across epochs) -----
        if _cached is not None:
            paths, subgraphs_per_band, paths_k = _cached[:3]
            min_hops = _cached[3] if len(_cached) > 3 else None
        elif mode == "single":
            paths_list = find_all_paths(data.adj_matrix, data.tx, data.rx)
            if len(paths_list) == 0:
                continue
            paths = paths_to_tensor(paths_list, device)
            subgraphs_per_band, paths_k = None, None

            lengths = [len(pp) - 1 for pp in paths_list]
            min_hops = min(lengths)

        elif mode == "multicast":
            graph_list = find_multicast_subgraphs(data.adj_matrix, data.tx, data.rx)
            if len(graph_list) == 0:
                continue
            subgraphs_per_band = [list(graph_list) for _ in range(model.B)]
            paths, paths_k = None, None

        elif mode == "multi":
            # One Tx, multiple Rx, one path set per receiver/message
            tx = getattr(data, "tx", None)
            rx = getattr(data, "rx", [])

            paths_k = []
            for r in rx:
                plist = find_all_paths(data.adj_matrix, tx, int(r))
                if len(plist) == 0:
                    paths_k = None
                    break
                paths_k.append(paths_to_tensor(plist, device))

            if paths_k is None:
                continue
            subgraphs_per_band, paths = None, None

        elif mode == "converge":
            # Multiple Tx, one Rx, one path set per transmitter/message
            tx = getattr(data, "tx", [])
            rx = getattr(data, "rx", None)

            if not isinstance(tx, (list, tuple)):
                tx = [tx]

            if isinstance(rx, (list, tuple)):
                if len(rx) != 1:
                    raise ValueError("For mode='converge', rx must contain exactly one receiver.")
                rx = int(rx[0])

            paths_k = []
            for t in tx:
                plist = find_all_paths(data.adj_matrix, int(t), int(rx))
                if len(plist) == 0:
                    paths_k = None
                    break
                paths_k.append(paths_to_tensor(plist, device))

            if paths_k is None:
                continue
            subgraphs_per_band, paths = None, None

        else:  # mode == "multiunicast"
            # Multiple Tx-Rx pairs, one path set per pair/message
            tx = getattr(data, "tx", [])
            rx = getattr(data, "rx", [])

            if not isinstance(tx, (list, tuple)):
                tx = [tx]

            if not isinstance(rx, (list, tuple)):
                rx = [rx]

            if len(tx) != len(rx):
                raise ValueError(
                    f"For mode='multiunicast', len(tx) must equal len(rx), "
                    f"got {len(tx)} and {len(rx)}."
                )

            paths_k = []
            for t, r in zip(tx, rx):
                plist = find_all_paths(data.adj_matrix, int(t), int(r))
                if len(plist) == 0:
                    paths_k = None
                    break
                paths_k.append(paths_to_tensor(plist, device))

            if paths_k is None:
                continue
            subgraphs_per_band, paths = None, None

        # Store freshly-built structures for reuse in later epochs.
        if struct_cache is not None and _cached is None:
            struct_cache[sid] = (paths, subgraphs_per_band, paths_k, min_hops)

        # Save TRUE CSI
        H_true = data.links_matrix
        using_estimate = False

        try:
            # Optional: swap in estimated CSI for forward
            if est_lookup is not None:
                sid = int(getattr(data, "sample_id", step))
                H_hat = est_lookup.get(sid, None)
                if H_hat is not None:
                    data.links_matrix = H_hat.to(device)
                    using_estimate = True

            with autocast(device_type=device.type, enabled=use_amp):
                # Forward to get per-layer P (and Z for multicommodity-like modes)
                if mode == "single":
                    _, p_list, _ = _compute_rates_per_layer(
                        model,
                        data,
                        paths=paths,
                        problem="single",
                        tau_min=tau,
                        tau_max=tau,
                        decentralized_inference=decentralized_inference,
                        max_route_hops=max_route_hops,
                        local_candidate_routing=local_candidate_routing,
                        hard_candidate_routing=hard_candidate_routing,
                    )

                elif mode == "multicast":
                    _, p_list, _ = _compute_rates_per_layer(
                        model,
                        data,
                        subgraphs_per_band=subgraphs_per_band,
                        problem="multicast",
                        tau_min=tau,
                        tau_max=tau,
                        decentralized_inference=decentralized_inference,
                        max_route_hops=max_route_hops,
                        local_candidate_routing=local_candidate_routing,
                        hard_candidate_routing=hard_candidate_routing,
                    )

                else:  # {"multi", "converge", "multiunicast"}
                    _, p_list, z_list = _compute_rates_per_layer(
                        model,
                        data,
                        paths_k=paths_k,
                        problem=mode,
                        tau_min=tau,
                        tau_max=tau,
                        reduce=reduce,
                        decentralized_inference=decentralized_inference,
                        max_route_hops=max_route_hops,
                        local_candidate_routing=local_candidate_routing,
                        hard_candidate_routing=hard_candidate_routing,
                    )

                # Restore TRUE CSI for loss
                if using_estimate:
                    data.links_matrix = H_true

                # Compute per-layer TRUE-CSI objectives
                rates_true_list = []
                for idx, P in enumerate(p_list):
                    if mode == "single":
                        r = calc_sum_rate(
                            h_arr=H_true,
                            p_arr=P,
                            sigma=data.sigma,
                            paths_tensor=paths,
                            B=model.B,
                            tau_min=tau,
                            tau_max=tau,
                        )

                    elif mode == "multicast":
                        r = objective_multicast(
                            h=H_true,
                            p=P,
                            sigma=data.sigma,
                            adj=data.adj_matrix,
                            subgraphs_per_band=subgraphs_per_band,
                            eps=1e-12,
                            tau_min=tau,
                            tau_max=tau,
                            per_band=False,
                        )

                    else:  # {"multi", "converge", "multiunicast"}
                        Z = z_list[idx]
                        r = objective_multicommodity(
                            h=H_true,
                            p=P,
                            z=Z,
                            sigma=data.sigma,
                            adj=data.adj_matrix,
                            paths_k=paths_k,
                            tau_min=tau,
                            tau_max=tau,
                            reduce=reduce,
                            per_band=False,
                        )

                    rates_true_list.append(r)

                rates_true = torch.stack(rates_true_list)  # [L]
                rate_last_true = rates_true[-1]
                loss_unsup = -rate_last_true

                if mono_weight > 0.0 and rates_true.numel() >= 2:
                    margin = 0.05
                    deltas = rates_true[1:] - rates_true[:-1]
                    shortfall = F.relu(margin - deltas)
                    penalty = mono_weight * shortfall.mean()
                else:
                    penalty = torch.tensor(0.0, device=device)

                # Concentration penalty: push power onto FEW outgoing edges so the
                # allocation forms a clean route, cutting the self-/off-route
                # interference that caps the rate (the diagnostic showed the GNN
                # lighting up ~all edges — off_route~0.88, total_power~9.5 vs the
                # optimizer's ~1.7). We sum p^2 over BANDS first, so multi-band use of
                # a single route is NOT penalized (eq 9 rewards it) — only spreading
                # over multiple edges is. (L1/L2 - 1) is the Hoyer-style spread: 0 for a
                # single active edge, sqrt(E)-1 for E equal edges. It is scale-invariant
                # (no under-power bias). We scale it by the achieved rate (detached) so
                # the penalty stays a fixed FRACTION of the objective across the wide
                # training SNR range, instead of being swamped at high SNR and
                # over-penalizing at low SNR.
                if sparsity_weight > 0.0:
                    P_last = p_list[-1]
                    edge_amp = torch.sqrt((P_last ** 2).sum(dim=0) + 1e-12)  # [n,n] or [K,n,n]
                    if edge_amp.dim() == 3:
                        # Multi-message ([B,K,n,n] -> edge_amp [K,n,n]): measure the
                        # edge-concentration WITHIN each commodity, then average over
                        # commodities. A single global L1/L2 over the flattened [K,n,n]
                        # is minimized by putting ALL power in one (commodity,edge) cell,
                        # i.e. it rewards serving a SINGLE message and penalizes spreading
                        # power across commodities — which directly fights the max-min
                        # (eq 10) objective and starves the min-commodity. Per-commodity
                        # spread keeps each message on a clean route without penalizing
                        # multi-commodity use. Average only over commodities that actually
                        # carry power (a dead commodity is all noise-floor and would score
                        # a spurious large spread).
                        amp_k = edge_amp.reshape(edge_amp.shape[0], -1)   # [K, n*n]
                        active_k = amp_k.max(dim=1).values > 1e-3         # commodity has a real edge
                        l1_k = amp_k.sum(dim=1)                           # [K]
                        l2_k = amp_k.pow(2).sum(dim=1).sqrt().clamp_min(1e-12)
                        spread_k = (l1_k / l2_k) - 1.0                    # [K]
                        spread = (spread_k[active_k].mean()
                                  if active_k.any() else edge_amp.new_zeros(()))
                    else:
                        l1 = edge_amp.sum()
                        l2 = edge_amp.pow(2).sum().sqrt().clamp_min(1e-12)
                        spread = (l1 / l2) - 1.0
                    sparsity_pen = sparsity_weight * rate_last_true.detach().abs() * spread
                else:
                    sparsity_pen = torch.tensor(0.0, device=device)

                # Budget-utilization penalty: push ACTIVE transmitters to use their full
                # per-node power budget. normalize_power only scales DOWN (never up) and
                # the sparsity penalty is scale-invariant, so nothing else rewards
                # saturating the budget — the diagnostic showed the GNN transmitting at
                # ~60% of budget (tot_pwr ~1.3 vs greedy 2.22), under-powering even on
                # trivial 1-hop routes. node_pwr[i] = ||P[:, i, :]||^2 (sum over bands,
                # [commodities,] destinations) is <= 1 by construction (post-projection).
                # We average (1 - node_pwr) over ACTIVE nodes only, so silent off-route
                # nodes are NOT pushed on (that would fight sparsity / re-spread power) —
                # sparsity picks WHICH nodes transmit, utilization makes them transmit at
                # full power. Rate-scaled + detached like the sparsity term (SNR-robust).
                # Optionally restrict utilization to SHORT routes. Full-budget power is
                # near-optimal only when interference is mild: greedy's full-power 1-/2-hop
                # rate ~= the optimizer's, but at 3+ hops the optimizer wins by BACKING OFF
                # power (inter-hop interference), so saturating a long route hurts. Gating
                # to routes <= utilization_max_hops applies the nudge exactly where it helps
                # (where the GNN-vs-greedy gap actually lives) and never on long routes.
                util_gated = (
                    utilization_max_hops > 0
                    and min_hops is not None
                    and min_hops > utilization_max_hops
                )
                if utilization_weight > 0.0 and not util_gated:
                    P_last = p_list[-1]
                    node_dim = 1 if P_last.dim() == 3 else 2   # source-node axis
                    reduce_dims = tuple(d for d in range(P_last.dim()) if d != node_dim)
                    node_pwr = (P_last.real ** 2).sum(dim=reduce_dims)   # [n], each in [0,1]
                    # Only saturate nodes that are ALREADY substantially committed. A low
                    # threshold (e.g. 1e-4) grabs every marginal off-route node and pumps
                    # it to full budget, manufacturing off-route interference (observed:
                    # off_route 0.11 -> 0.36, tot_pwr 1.3 -> 2.8, rate DOWN). 0.25 leaves
                    # marginal nodes to the sparsity penalty to zero out. NOTE: full-budget
                    # saturation is only optimal in the interference-free (1-hop) case — the
                    # optimizer itself backs power off under interference — so keep the
                    # weight small if used at all.
                    active = node_pwr > 0.25
                    if active.any():
                        util = node_pwr[active].mean()
                        util_pen = utilization_weight * rate_last_true.detach().abs() * (1.0 - util)
                    else:
                        util_pen = torch.tensor(0.0, device=device)
                else:
                    util_pen = torch.tensor(0.0, device=device)

                loss = (loss_unsup + penalty + sparsity_pen + util_pen) / grad_accum_steps

        finally:
            data.links_matrix = H_true

        # Backward/step
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % grad_accum_steps == 0:
            use_scaler = scaler is not None and scaler.is_enabled()
            if use_scaler:
                scaler.unscale_(optimizer)
            # Non-finite guard: only clip / step when ALL grads are finite. A single inf/nan
            # grad (e.g. a high-SNR fp16 overflow) makes clip_grad_norm's total_norm NaN, which
            # multiplies EVERY grad by a NaN coeff and permanently corrupts the weights. Skip
            # such a step instead. (With AMP the enabled scaler also skips inf internally; this
            # additionally protects the fp32 path and stops clip from spreading the NaN.)
            grads_finite = all((p.grad is None) or torch.isfinite(p.grad).all()
                               for p in model.parameters())
            if grads_finite and grad_clip is not None:
                clip_grad_norm_(model.parameters(), grad_clip)

            if use_scaler:
                scaler.step(optimizer)   # internally skips on inf; clip above already guarded
                scaler.update()
            elif grads_finite:
                optimizer.step()         # fp32 path: guard the step ourselves

            optimizer.zero_grad(set_to_none=True)

        total_loss_val += float(loss.detach().cpu()) * grad_accum_steps
        num_batches += 1

    avg_loss = total_loss_val / max(num_batches, 1)
    stats = {"loss": avg_loss, "lr": optimizer.param_groups[0]["lr"]}
    print(f"[E{epoch:02d}][{mode}] loss={stats['loss']:.6f} | lr={stats['lr']:.2e}")
    return stats


@torch.no_grad()
def validate_chained(
    model,
    loader,
    *,
    batch_size: int = 1,
    mode: str = "single",          # "single" | "multicast" | "multi" | "converge" | "multiunicast"
    device=None,
    est_dataset=None,              # Optional EstimatedCSIDataset or dict(sample_id -> H_hat)
    tau: float = 0.0,              # soft-min temperature
    reduce: str = "fair",          # K>1 message reduction: "fair"=eq10 min | "sum"/"mean" throughput
    struct_cache: dict = None,     # persistent {sample_id: structures} cache across epochs
    verbose: bool = False,
    decentralized_inference: bool = False,  # True: candidate head scores only source-local L-hop paths.
    max_route_hops: int | None = None,      # Optional cap; if None, each layer uses its l+1 radius.
    local_candidate_routing: bool = False,
    hard_candidate_routing=None,
):
    """
    Validate a ChainedGNN on one of the supported communication problems:

      - single:
          One Tx -> one Rx. Best-path objective.

      - multicast:
          One Tx -> multiple Rx, same shared message.
          Objective is max over multicast subgraphs, then min over edges.

      - multi:
          One Tx -> multiple Rx, different messages.
          Objective is multicommodity: one path set per receiver/message.

      - converge:
          Multiple Tx -> one Rx, different messages.
          Objective is multicommodity: one path set per transmitter/message.

      - multiunicast:
          Multiple Tx-Rx pairs, each carrying a distinct message.
          Objective is multicommodity: one path set per Tx-Rx pair.

    Evaluation policy:
      - If `est_dataset` is provided:
          * Forward pass uses estimated CSI (H_hat) to predict powers.
          * Metric is computed under TRUE CSI (H_true).
      - If `est_dataset` is None:
          * Forward and metric both use TRUE CSI.
      - Metric per sample is the best over layers.
      - Final report is the average of per-sample best-layer TRUE-CSI rates.

    Args:
        model (torch.nn.Module):
            ChainedGNN model.

        loader (torch.utils.data.DataLoader):
            Yields graph batches with fields:
              - adj_matrix: [n, n] (bool/0-1)
              - links_matrix: [B, n, n] (complex)
              - sigma: float or [B] tensor
              - tx: int or list[int]
              - rx: int or list[int]
              - optional sample_id: int

        mode (str, optional):
            One of:
              - "single"
              - "multicast"
              - "multi"
              - "converge"
              - "multiunicast"

        device (torch.device | None, optional):
            If None, inferred from model parameters.

        est_dataset (dict | EstimatedCSIDataset | None, optional):
            If provided, maps sample_id -> H_hat, or is converted by
            build_estimate_lookup().

        tau (float, optional):
            Soft-min temperature. For "single", forwarded to calc_sum_rate.
            For "multicast", used as tau_min in objective_multicast.
            For {"multi", "converge", "multiunicast"}, used as tau_min in
            objective_multicommodity.

        verbose (bool, optional):
            If True, prints the averaged metric.

    Returns:
        dict:
          {
            "best_rate": float,   # average over samples of the best per-layer TRUE-CSI rate
          }
    """
    assert mode in {"single", "multicast", "multi", "converge", "multiunicast"}
    device = next(model.parameters()).device if device is None else device
    model.eval()

    est_lookup = None
    if est_dataset is not None:
        est_lookup = est_dataset if isinstance(est_dataset, dict) else build_estimate_lookup(est_dataset)

    total_best, count = 0.0, 0

    for step, data in enumerate(loader):
        data = data.to(device)
        if batch_size == 1:
            if isinstance(data.rx, (list, tuple)) and len(data.rx) == 1 and isinstance(data.rx[0], (list, tuple)):
                data.rx = data.rx[0]
            if isinstance(data.tx, (list, tuple)) and len(data.tx) == 1 and isinstance(data.tx[0], (list, tuple)):
                data.tx = data.tx[0]

        # Per-sample routing structures are fixed across epochs; compute once and
        # reuse from struct_cache afterward.
        sid = int(getattr(data, "sample_id", step))
        _cached = struct_cache.get(sid) if struct_cache is not None else None

        # ----- Build graph structures (cached per sample across epochs) -----
        if _cached is not None:
            paths, subgraphs_per_band, paths_k = _cached[:3]
        elif mode == "single":
            paths_list = find_all_paths(data.adj_matrix, data.tx, data.rx)
            if len(paths_list) == 0:
                continue
            paths = paths_to_tensor(paths_list, device)
            subgraphs_per_band, paths_k = None, None

        elif mode == "multicast":
            graph_list = find_multicast_subgraphs(data.adj_matrix, data.tx, data.rx)
            if len(graph_list) == 0:
                continue
            subgraphs_per_band = [list(graph_list) for _ in range(model.B)]
            paths, paths_k = None, None

        elif mode == "multi":
            # One Tx, multiple Rx, one path set per receiver/message
            tx = getattr(data, "tx", None)
            rx = getattr(data, "rx", [])

            if isinstance(tx, torch.Tensor):
                tx = tx.item() if tx.numel() == 1 else tx.view(-1).tolist()
            if isinstance(rx, torch.Tensor):
                rx = rx.view(-1).tolist()

            paths_k = []
            for r in rx:
                plist = find_all_paths(data.adj_matrix, tx, int(r))
                if len(plist) == 0:
                    paths_k = None
                    break
                paths_k.append(paths_to_tensor(plist, device))

            if paths_k is None:
                continue
            subgraphs_per_band, paths = None, None

        elif mode == "converge":
            # Multiple Tx, one Rx, one path set per transmitter/message
            tx = getattr(data, "tx", [])
            rx = getattr(data, "rx", None)

            if isinstance(tx, torch.Tensor):
                tx = tx.view(-1).tolist()
            elif not isinstance(tx, (list, tuple)):
                tx = [tx]

            if isinstance(rx, torch.Tensor):
                rx = rx.item() if rx.numel() == 1 else rx.view(-1).tolist()
            if isinstance(rx, (list, tuple)):
                if len(rx) != 1:
                    raise ValueError("For mode='converge', rx must contain exactly one receiver.")
                rx = int(rx[0])

            paths_k = []
            for t in tx:
                plist = find_all_paths(data.adj_matrix, int(t), int(rx))
                if len(plist) == 0:
                    paths_k = None
                    break
                paths_k.append(paths_to_tensor(plist, device))

            if paths_k is None:
                continue
            subgraphs_per_band, paths = None, None

        else:  # mode == "multiunicast"
            # Multiple Tx-Rx pairs, one path set per pair/message
            tx = getattr(data, "tx", [])
            rx = getattr(data, "rx", [])

            if isinstance(tx, torch.Tensor):
                tx = tx.view(-1).tolist()
            elif not isinstance(tx, (list, tuple)):
                tx = [tx]

            if isinstance(rx, torch.Tensor):
                rx = rx.view(-1).tolist()
            elif not isinstance(rx, (list, tuple)):
                rx = [rx]

            if len(tx) != len(rx):
                raise ValueError(
                    f"For mode='multiunicast', len(tx) must equal len(rx), "
                    f"got {len(tx)} and {len(rx)}."
                )

            paths_k = []
            for t, r in zip(tx, rx):
                plist = find_all_paths(data.adj_matrix, int(t), int(r))
                if len(plist) == 0:
                    paths_k = None
                    break
                paths_k.append(paths_to_tensor(plist, device))

            if paths_k is None:
                continue
            subgraphs_per_band, paths = None, None

        # Store freshly-built structures for reuse in later epochs.
        if struct_cache is not None and _cached is None:
            struct_cache[sid] = (paths, subgraphs_per_band, paths_k)

        # TRUE CSI
        H_true = data.links_matrix
        using_estimate = False

        try:
            if est_lookup is not None:
                sid = int(getattr(data, "sample_id", step))
                H_hat = est_lookup.get(sid, None)
                if H_hat is not None:
                    data.links_matrix = H_hat.to(device)
                    using_estimate = True

            with autocast(device_type=device.type, enabled=False):
                if mode == "single":
                    _, p_list, _ = _compute_rates_per_layer(
                        model, data,
                        paths=paths,
                        problem="single",
                        tau_min=0.0,
                        tau_max=0.0,
                        decentralized_inference=decentralized_inference,
                        max_route_hops=max_route_hops,
                        local_candidate_routing=local_candidate_routing,
                        hard_candidate_routing=hard_candidate_routing,
                    )
                elif mode == "multicast":
                    _, p_list, _ = _compute_rates_per_layer(
                        model, data,
                        subgraphs_per_band=subgraphs_per_band,
                        problem="multicast",
                        tau_min=0.0,
                        tau_max=0.0,
                        decentralized_inference=decentralized_inference,
                        max_route_hops=max_route_hops,
                        local_candidate_routing=local_candidate_routing,
                        hard_candidate_routing=hard_candidate_routing,
                    )
                else:  # {"multi", "converge", "multiunicast"}
                    _, p_list, z_list = _compute_rates_per_layer(
                        model, data,
                        paths_k=paths_k,
                        problem=mode,
                        tau_min=0.0,
                        tau_max=0.0,
                        reduce=reduce,
                        decentralized_inference=decentralized_inference,
                        max_route_hops=max_route_hops,
                        local_candidate_routing=local_candidate_routing,
                        hard_candidate_routing=hard_candidate_routing,
                    )

            if using_estimate:
                data.links_matrix = H_true

            rates_true_list = []
            for idx, P in enumerate(p_list):
                if mode == "single":
                    r = calc_sum_rate(
                        h_arr=H_true,
                        p_arr=P,
                        sigma=data.sigma,
                        paths_tensor=paths,
                        B=model.B,
                        tau_min=0,
                        tau_max=0,
                    )
                elif mode == "multicast":
                    r = objective_multicast(
                        h=H_true,
                        p=P,
                        sigma=data.sigma,
                        adj=data.adj_matrix,
                        subgraphs_per_band=subgraphs_per_band,
                        eps=1e-12,
                        tau_min=0,
                        tau_max=0.0,
                        per_band=False
                    )
                else:  # {"multi", "converge", "multiunicast"}
                    Z = z_list[idx]
                    r = objective_multicommodity(
                        h=H_true,
                        p=P,
                        z=Z,
                        sigma=data.sigma,
                        adj=data.adj_matrix,
                        paths_k=paths_k,
                        tau_min=0,
                        tau_max=0.0,
                        reduce=reduce,
                        per_band=False
                    )
                rates_true_list.append(r)

            finite_vals = [float(r.item()) if torch.isfinite(r) else float("-inf") for r in rates_true_list]
            if all(v == float("-inf") for v in finite_vals):
                continue

            best_rate_true = max(v for v in finite_vals if v != float("-inf"))
            total_best += best_rate_true
            count += 1

        finally:
            data.links_matrix = H_true

    avg_best = (total_best / count) if count > 0 else 0.0
    if verbose:
        print(f"[VAL][{mode}] best_rate={avg_best:.6f}")

    return {"best_rate": avg_best}