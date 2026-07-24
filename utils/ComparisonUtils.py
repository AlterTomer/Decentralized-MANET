# import numpy as np
# import torch
# from utils.CentralizedUtils import (evaluate_centralized_adam,
#                                     compute_centralized_best_single_channel_rate,
#                                     compute_decentralized_best_single_channel_rate,
#                                     compute_equal_power_bound,
#                                     compute_centralized_greedy_power_rate,
#                                     compute_decentralized_greedy_power_rate,
#                                     evaluate_ffn)
# from utils.PathUtils import find_all_paths, paths_to_tensor
# from utils.MetricUtils import calc_sum_rate
# from utils.DataUtils import mean_var_over_dataset
# from models.GraphNetAux import _compute_rates_per_layer
# from Multicast.SubGraphs import find_multicast_subgraphs
#
# # -------------------------------
# # Helpers
# # -------------------------------
#
# def evaluate_across_snr(
#     dataset,
#     model,
#     # ffn_dataset,
#     # ffn_model,
#     B,
#     snr_db_list,
#     *,
#     problem: str = "single",      # "single" | "multicast" | "multi" | "converge" | "multiunicast"
#     reduce: str = "fair",         # K>1 message reduction: "fair"=eq10 min | "sum"/"mean" throughput
# ):
#     """
#     Sequential evaluation across a list of SNR values.
#
#     Args:
#         dataset: iterable of graph data objects.
#         model:   trained GNN model.
#         ffn_dataset: iterable of MANET data objects for FFN.
#         ffn_model: trained FFN model.
#         B: number of bands.
#         snr_db_list: list of SNR values in dB.
#         problem: one of
#             - "single"
#             - "multicast"
#             - "multi"
#             - "converge"
#             - "multiunicast"
#         multi_mode: for multi-message problems in strongest-bottleneck:
#             - "global": single best (b,k) over all commodities.
#             - "per_commodity": unique band per commodity (if supported).
#
#     Returns:
#         dict: {
#             "gnn": {snr_db: mean_rate},
#             "centralized": {snr_db: mean_rate},
#             "strongest bottleneck": {snr_db: mean_rate},
#             "equal power": {snr_db: mean_rate},
#             "greedy maxlink": {snr_db: mean_rate}
#         }
#     """
#     device = next(model.parameters()).device
#     results = {
#         "gnn": {},
#         "ffn": {},
#         "centralized": {},
#         "strongest bottleneck": {},
#         "strongest bottleneck decentralized": {},
#         "equal power": {},
#         "greedy maxlink": {},
#         "greedy maxlink decentralized": {}
#     }
#
#     # --- compute mean channel variance for noise scaling ---
#     mean_channel_var = mean_var_over_dataset(dataset)
#
#     # Per-sample GNN routing structures (paths/subgraphs) are topology-only and thus
#     # identical across SNR, so build them once (on the first SNR) and reuse. Keyed by
#     # sample index in the fixed dataset iteration order.
#     gnn_struct_cache = {}
#     # Shared across the heuristic benchmarks (widest-path / greedy / equal-power): memoizes
#     # find_all_paths / find_multicast_subgraphs per (sample, endpoints) across the SNR sweep.
#     # Only the path ENUMERATION is cached — routing, random tie-breaks and scoring still run
#     # per SNR, so results are identical to the uncached path.
#     heuristic_paths_cache = {}
#
#     # Information horizon for the decentralized GREEDY heuristic: an L-layer GNN propagates
#     # node information L hops (each gated layer aggregates 1-hop neighbors), so a node's
#     # routing knowledge reaches L hops. Its distributed shortest-path routing is therefore
#     # bounded to L rounds (an L-hop horizon); routes longer than L hops cannot be discovered,
#     # while centralized greedy keeps exhaustive path enumeration (exponential, infeasible in
#     # real time) and finds the global shortest path. At n-1 (full convergence) distributed
#     # Bellman-Ford is EXACT and the two tie; the L bound is what makes centralized > decen-
#     # tralized greedy. NOTE: only greedy is bounded. The strongest-bottleneck (widest-path)
#     # is left EXACT/unbounded — bounding it inverts the ordering (shorter routes carry less
#     # self-interference, so a horizon-limited widest-path scores HIGHER), which is misleading.
#     # num_layers is the code's L (gated-layer count).
#     dec_rounds = int(getattr(model, "num_layers", 2))
#     # Toggle for the strongest-bottleneck (widest-path) benchmark: set to `dec_rounds` to
#     # apply the same L-hop horizon, or `None` to leave it EXACT/unbounded (original behavior,
#     # current default — bounding it inverts the ordering since shorter routes carry less
#     # self-interference; see note above). Kept wired so both options are one flip away.
#     bottleneck_rounds = None  # <- set to `dec_rounds` to bound the widest-path benchmark too
#     print(f"[decentralized greedy horizon] max routing rounds = L = {dec_rounds}"
#           f" | strongest-bottleneck horizon = {bottleneck_rounds if bottleneck_rounds is not None else 'exact'}")
#
#     for snr_db in snr_db_list:
#         snr = 10.0 ** (snr_db / 10.0)
#         sigma2 = mean_channel_var / snr
#         sigma = sigma2 ** 0.5
#         print(f"SNR: {snr_db} dB")
#
#         # =================================================
#         # 1) GNN mean rate
#         # =================================================
#         model.eval()
#         with torch.no_grad():
#             rates = []
#             for i, d in enumerate(dataset):
#                 # Set noise
#                 d.sigma = torch.tensor(sigma, device=device)
#
#                 adj = d.adj_matrix
#                 tx = d.tx
#                 rx = d.rx
#
#                 # normalize tx/rx into python lists when needed
#                 if isinstance(tx, torch.Tensor):
#                     tx = tx.view(-1).tolist() if tx.numel() > 1 else int(tx.item())
#                 if isinstance(rx, torch.Tensor):
#                     rx = rx.view(-1).tolist() if rx.numel() > 1 else int(rx.item())
#
#                 # Problem-specific structures for _compute_rates_per_layer.
#                 # Topology-only -> identical across SNR, so build once and cache per
#                 # sample. `skip=True` marks a sample with no valid path (scores 0).
#                 cached = gnn_struct_cache.get(i)
#                 if cached is not None:
#                     paths, subgraphs_per_band, paths_k, skip = cached
#                     if skip:
#                         rates.append(0.0)
#                         continue
#                 else:
#                     paths = None
#                     subgraphs_per_band = None
#                     paths_k = None
#                     skip = False
#
#                     if problem == "single":
#                         # Tx -> Rx paths
#                         raw_paths = find_all_paths(adj.cpu(), tx, rx)
#                         if not raw_paths:
#                             skip = True
#                         else:
#                             paths = paths_to_tensor(raw_paths, device)
#
#                     elif problem == "multicast":
#                         # one Tx, multiple Rx, shared message
#                         subgraphs = find_multicast_subgraphs(adj, tx, rx)
#                         if (subgraphs is None) or (len(subgraphs) == 0):
#                             skip = True
#                         else:
#                             subgraphs_per_band = [subgraphs for _ in range(B)]
#
#                     elif problem == "multi":
#                         # one Tx, multiple Rx, one path set per receiver/message
#                         rx_list = list(rx) if isinstance(rx, (list, tuple)) else [rx]
#
#                         paths_k = []
#                         has_any_path = False
#                         for rx_k in rx_list:
#                             raw_paths_k = find_all_paths(adj.cpu(), tx, int(rx_k))
#                             if raw_paths_k:
#                                 has_any_path = True
#                             paths_k.append(
#                                 paths_to_tensor(raw_paths_k, device)
#                                 if raw_paths_k else
#                                 torch.empty((0, 0), dtype=torch.long, device=device)
#                             )
#                         if not has_any_path:
#                             skip = True
#
#                     elif problem == "converge":
#                         # multiple Tx, one Rx, one path set per transmitter/message
#                         tx_list = list(tx) if isinstance(tx, (list, tuple)) else [tx]
#                         if isinstance(rx, (list, tuple)):
#                             if len(rx) != 1:
#                                 raise ValueError("For problem='converge', rx must contain exactly one receiver.")
#                             rx_scalar = int(rx[0])
#                         else:
#                             rx_scalar = int(rx)
#
#                         paths_k = []
#                         has_any_path = False
#                         for tx_k in tx_list:
#                             raw_paths_k = find_all_paths(adj.cpu(), int(tx_k), rx_scalar)
#                             if raw_paths_k:
#                                 has_any_path = True
#                             paths_k.append(
#                                 paths_to_tensor(raw_paths_k, device)
#                                 if raw_paths_k else
#                                 torch.empty((0, 0), dtype=torch.long, device=device)
#                             )
#                         if not has_any_path:
#                             skip = True
#
#                     elif problem == "multiunicast":
#                         # multiple Tx-Rx pairs, one path set per pair/message
#                         tx_list = list(tx) if isinstance(tx, (list, tuple)) else [tx]
#                         rx_list = list(rx) if isinstance(rx, (list, tuple)) else [rx]
#
#                         if len(tx_list) != len(rx_list):
#                             raise ValueError(
#                                 f"For problem='multiunicast', len(tx) must equal len(rx), "
#                                 f"got {len(tx_list)} and {len(rx_list)}."
#                             )
#
#                         paths_k = []
#                         has_any_path = False
#                         for tx_k, rx_k in zip(tx_list, rx_list):
#                             raw_paths_k = find_all_paths(adj.cpu(), int(tx_k), int(rx_k))
#                             if raw_paths_k:
#                                 has_any_path = True
#                             paths_k.append(
#                                 paths_to_tensor(raw_paths_k, device)
#                                 if raw_paths_k else
#                                 torch.empty((0, 0), dtype=torch.long, device=device)
#                             )
#                         if not has_any_path:
#                             skip = True
#
#                     else:
#                         raise ValueError(f"Unknown problem type: {problem}")
#
#                     gnn_struct_cache[i] = (paths, subgraphs_per_band, paths_k, skip)
#                     if skip:
#                         rates.append(0.0)
#                         continue
#
#                 # Tag problem (if model cares about it)
#                 setattr(d, "problem", problem)
#                 d = d.to(device)
#
#                 # Call helper
#                 rates_per_layer, _, _ = _compute_rates_per_layer(
#                     model,
#                     d,
#                     paths=paths,
#                     subgraphs_per_band=subgraphs_per_band,
#                     paths_k=paths_k,
#                     problem=problem,
#                     tau_min=0.0,
#                     tau_max=0.0,
#                     reduce=reduce,
#                 )
#
#                 # Take best layer
#                 layer_rates = torch.stack(rates_per_layer)  # [L]
#                 rate = layer_rates.max().item()
#                 rates.append(rate)
#
#             results["gnn"][snr_db] = float(np.mean(rates))
#
#         # =================================================
#         # 2) FFN mean rate
#         # =================================================
#         # ffn_model.eval()
#         # ffn_loader = torch.utils.data.DataLoader(ffn_dataset, batch_size=1, shuffle=False)
#         # with torch.no_grad():
#         #     ffn_rates, _ = evaluate_ffn(ffn_model, ffn_loader, sigma_noise=sigma, problem=problem, reduce=reduce)
#         # results["ffn"][snr_db] = float(np.mean(ffn_rates))
#         results["ffn"][snr_db] = 0
#
#         # ==============================================
#         # 3) Centralized ADAM benchmark
#         # ==============================================
#         adam_rates, _ = evaluate_centralized_adam(
#             dataset,
#             B,
#             noise_std=sigma,
#             num_iterations=500,
#             problem=problem,
#             reduce=reduce,
#         )
#         results["centralized"][snr_db] = float(np.mean(adam_rates))
#
#         # ==============================================
#         # 4) Strongest bottleneck lower bound
#         # ==============================================
#         bottleneck_rates, _ = compute_centralized_best_single_channel_rate(
#             dataset,
#             problem=problem,
#             sigma_noise=sigma,
#             paths_cache=heuristic_paths_cache,
#             reduce=reduce,
#         )
#         results["strongest bottleneck"][snr_db] = float(np.mean(bottleneck_rates))
#
#         # Strongest-bottleneck (widest-path): horizon controlled by `bottleneck_rounds`
#         # (default None = exact/unbounded, as before). Flip that toggle to bound it.
#         bottleneck_rates_decentralized, _, _ = compute_decentralized_best_single_channel_rate(
#             dataset,
#             problem=problem,
#             sigma_noise=sigma,
#             max_iters=bottleneck_rounds,
#             paths_cache=heuristic_paths_cache,
#             reduce=reduce,
#         )
#         results["strongest bottleneck decentralized"][snr_db] = float(np.mean(bottleneck_rates_decentralized))
#
#         # ==============================================
#         # 5) Equal-power heuristic
#         # ==============================================
#         rates_equal_power, _ = compute_equal_power_bound(
#             dataset,
#             sigma_noise=sigma,
#             problem=problem,
#             paths_cache=heuristic_paths_cache,
#             reduce=reduce,
#         )
#         results["equal power"][snr_db] = float(np.mean(rates_equal_power))
#
#         # ==============================================
#         # 6) Greedy max-link benchmark
#         # ==============================================
#         rates_greedy, _ = compute_centralized_greedy_power_rate(
#             dataset,
#             sigma_noise=sigma,
#             problem=problem,
#             paths_cache=heuristic_paths_cache,
#             reduce=reduce,
#         )
#         results["greedy maxlink"][snr_db] = float(np.mean(rates_greedy))
#
#         rates_greedy_decentralized, _ = compute_decentralized_greedy_power_rate(
#             dataset,
#             sigma_noise=sigma,
#             problem=problem,
#             max_iters=dec_rounds,
#             paths_cache=heuristic_paths_cache,
#             reduce=reduce,
#         )
#         results["greedy maxlink decentralized"][snr_db] = float(np.mean(rates_greedy_decentralized))
#
#     return results
#
#
#
# def time_model_compare(dataset, big_model, small_model, snr_db_list):
#     """
#     Sequential evaluation across a list of SNR values.
#     The goal is to test the scalability of ChainedGNN (evaluate data samples of large topology using a model that was trained on a smaller topology).
#
#     Args:
#         dataset: Dataset based on large topology (already on CPU or GPU as needed).
#         big_model: Trained GNN model on a 'large' topology.
#         small_model: Trained GNN model on a 'small' topology.
#         snr_db_list: List of SNR values in dB.
#
#     Returns:
#         dict: { "big": {snr_db: mean_rate}, "small": {snr_db: mean_rate}}
#     """
#     assert big_model.B == small_model.B , "models must have the same B attribute"
#     device = next(big_model.parameters()).device
#     results = {"big": {}, "small": {}}
#
#     # --- compute mean channel variance for noise scaling ---
#     mean_channel_var = mean_var_over_dataset(dataset)
#
#     for snr_db in snr_db_list:
#         snr = 10.0 ** (snr_db / 10.0)
#         sigma2 = mean_channel_var / snr
#         sigma = sigma2 ** 0.5
#         print(f'SNR: {snr_db} dB')
#
#         # GNN mean rate
#         big_model.eval()
#         small_model.eval()
#         with torch.no_grad():
#             big_rates = []
#             small_rates = []
#             for d in dataset:
#                 d.sigma = torch.tensor(sigma, device=device)
#                 d = d.to(device)
#
#                 paths = find_all_paths(d.adj_matrix, d.tx, d.rx)
#                 paths = paths_to_tensor(paths, device)
#
#                 big_gnn_rates, _ = _compute_rates_per_layer(big_model, d, paths)
#                 big_rate  = torch.stack(big_gnn_rates).max().item()
#                 big_rates.append(big_rate)
#
#                 small_gnn_rates, _ = _compute_rates_per_layer(small_model, d, paths)
#                 small_rate = torch.stack(small_gnn_rates).max().item()
#                 small_rates.append(small_rate)
#
#             results["big"][snr_db] = float(np.mean(big_rates))
#             results["small"][snr_db] = float(np.mean(small_rates))
#
#     return results
#
# @torch.inference_mode()
# def est_true_model_compare(true_dataset, est_dataset, true_model, est_model, snr_db_list):
#     """
#     Compare a model trained on true CSI vs a model trained on estimated CSI across SNR values.
#     Args:
#         true_dataset: Dataset based on true CSI (already on CPU or GPU as needed).
#         est_dataset: Dataset based on estimated CSI (already on CPU or GPU as needed).
#         true_model: Trained GNN model on true CSI..
#         est_model: Trained GNN model on estimated CSI.
#         snr_db_list: List of SNR values in dB.
#
#     Returns:
#         dict: { "true": {snr_db: mean_rate}, "est": {snr_db: mean_rate}}
#     """
#     assert true_model.B == est_model.B, "models must have the same B"
#     device = next(true_model.parameters()).device
#     true_model.eval()
#     est_model.eval()
#
#     # --- SNR normalization from TRUE dataset (fairness) ---
#     # mean_var is a scalar variance; convert to std for calc_sum_rate downstream.
#     true_mean_var = mean_var_over_dataset(true_dataset)  # scalar VAR
#     results = {"true": {}, "est": {}}
#
#     # sanity: we rely on index alignment across datasets
#     if len(true_dataset) != len(est_dataset):
#         raise ValueError("true_dataset and est_dataset length mismatch; cannot align by index.")
#
#     for snr_db in snr_db_list:
#         print(f'SNR: {snr_db} dB')
#         snr = 10.0 ** (snr_db / 10.0)
#
#         # σ^2 = mean_var / SNR  --> σ = sqrt(σ^2)
#         sigma2 = true_mean_var / snr
#         sigma = float(sigma2 ** 0.5)
#         sigma_t = torch.tensor(sigma, device=device)
#
#         # ----- TRUE track: powers from TRUE model on TRUE inputs; score on TRUE CSI -----
#         true_rates = []
#         for i in range(len(true_dataset)):
#             d_true = true_dataset[i].to(device)
#
#             # (re)compute paths as requested
#             paths = find_all_paths(d_true.adj_matrix, d_true.tx, d_true.rx)
#             if len(paths) == 0:
#                 continue
#             paths = paths_to_tensor(paths, device)
#
#             # get powers from the true-CSI-trained model (ignore internal rates)
#             _, P_list_true = _compute_rates_per_layer(true_model, d_true, paths)
#             P_true = P_list_true[-1]
#
#             r_true = calc_sum_rate(
#                 h_arr=d_true.links_matrix,  # TRUE CSI
#                 p_arr=P_true,
#                 sigma=sigma_t,  # std
#                 paths_tensor=paths,
#                 B=true_model.B,
#                 tau=0
#             )
#             true_rates.append(float(r_true.item()))
#         results["true"][snr_db] = float(np.mean(true_rates)) if true_rates else float("nan")
#
#         # ----- EST track: powers from EST model on EST inputs; score on TRUE CSI -----
#         est_rates = []
#         for i in range(len(est_dataset)):
#             d_est = est_dataset[i].to(device)
#             d_true = true_dataset[i].to(device)  # same graph/topology order
#
#             paths = find_all_paths(d_est.adj_matrix, d_est.tx, d_est.rx)
#             if len(paths) == 0:
#                 continue
#             paths = paths_to_tensor(paths, device)
#
#             _, P_list_est = _compute_rates_per_layer(est_model, d_est, paths)
#             P_est = P_list_est[-1]
#
#             r_est = calc_sum_rate(
#                 h_arr=d_true.links_matrix,  # score on TRUE CSI
#                 p_arr=P_est,
#                 sigma=sigma_t,  # std
#                 paths_tensor=paths,
#                 B=est_model.B,
#                 tau=0
#             )
#             est_rates.append(float(r_est.item()))
#         results["est"][snr_db] = float(np.mean(est_rates)) if est_rates else float("nan")
#
#     return results
#
# def evaluate_models_across_snr(
#     dataset,
#     models,
#     B,
#     snr_db_list,
#     *,
#     problem: str = "single",      # "single" | "multicast" | "multi"
#     multi_mode: str = "global",   # kept for API symmetry (not used inside GNN eval)
#     take_best_layer: bool = True, # matches your current behavior
# ):
#     """
#     Evaluate achieved mean rate vs SNR for multiple trained GNN models (ablation study).
#
#     Args:
#         dataset: iterable of graph data objects.
#         models: list of tuples [(name, model), ...] OR dict {name: model}.
#                 Each model must already be on the correct device.
#         B: number of bands.
#         snr_db_list: list of SNR values in dB.
#         problem: "single", "multicast", or "multi".
#         multi_mode: unused here; kept to match your existing signature style.
#         take_best_layer: if True, per sample take max over layer outputs (as in your code).
#                          if False, take last layer rate.
#
#     Returns:
#         dict: {
#             "models": {
#                 model_name: {snr_db: mean_rate, ...},
#                 ...
#             }
#         }
#     """
#     # Normalize models input
#     if isinstance(models, dict):
#         model_items = list(models.items())
#     else:
#         model_items = list(models)  # list of (name, model)
#
#     if len(model_items) == 0:
#         raise ValueError("models is empty.")
#
#     # Compute mean channel variance once (consistent noise scaling)
#     mean_channel_var = mean_var_over_dataset(dataset)
#
#     results = {"models": {name: {} for name, _ in model_items}}
#
#     for snr_db in snr_db_list:
#         snr = 10.0 ** (snr_db / 10.0)
#         sigma2 = mean_channel_var / snr
#         sigma = float(sigma2 ** 0.5)
#         print(f"SNR: {snr_db} dB")
#
#         for name, model in model_items:
#             device = next(model.parameters()).device
#             model.eval()
#
#             with torch.no_grad():
#                 rates = []
#                 for d in dataset:
#                     # Set noise (same behavior as your baseline function)
#                     d.sigma = torch.tensor(sigma, device=device)
#
#                     adj = d.adj_matrix
#                     tx = d.tx
#                     rx = d.rx
#
#                     paths = None
#                     subgraphs_per_band = None
#                     paths_k = None
#
#                     if problem == "single":
#                         raw_paths = find_all_paths(adj.cpu(), tx, rx)
#                         if not raw_paths:
#                             rates.append(0.0)
#                             continue
#                         paths = paths_to_tensor(raw_paths, device)
#
#                     elif problem == "multicast":
#                         # rx is list of receivers (kept for parity with your code)
#                         if isinstance(rx, (list, tuple)):
#                             rx_list = list(rx)
#                         else:
#                             rx_list = [rx]
#
#                         subgraphs = find_multicast_subgraphs(d.adj_matrix, d.tx, d.rx)
#                         if (subgraphs is None) or (len(subgraphs) == 0):
#                             rates.append(0.0)
#                             continue
#                         subgraphs_per_band = [subgraphs for _ in range(B)]
#
#                     elif problem == "multi":
#                         if isinstance(rx, (list, tuple)):
#                             rx_list = list(rx)
#                         else:
#                             rx_list = [rx]
#                         K = len(rx_list)
#
#                         paths_k = []
#                         has_any_path = False
#                         for rx_k in rx_list:
#                             raw_paths_k = find_all_paths(adj.cpu(), tx, rx_k)
#                             if raw_paths_k:
#                                 has_any_path = True
#                                 paths_k.append(paths_to_tensor(raw_paths_k, device))
#                             else:
#                                 paths_k.append(torch.empty((0, 0), dtype=torch.long, device=device))
#
#                         if not has_any_path:
#                             rates.append(0.0)
#                             continue
#
#                     else:
#                         raise ValueError(f"Unknown problem type: {problem}")
#
#                     setattr(d, "problem", problem)
#                     d = d.to(device)
#
#                     rates_per_layer, _, _ = _compute_rates_per_layer(
#                         model,
#                         d,
#                         paths=paths,
#                         subgraphs_per_band=subgraphs_per_band,
#                         paths_k=paths_k,
#                         problem=problem,
#                         tau_min=0.0,
#                         tau_max=0.0,
#                     )
#
#                     layer_rates = torch.stack(rates_per_layer)  # [L]
#                     if take_best_layer:
#                         rate = layer_rates.max().item()
#                     else:
#                         rate = layer_rates[-1].item()
#                     rates.append(rate)
#
#             results["models"][name][snr_db] = float(np.mean(rates))
#
#     return results
#
#
#
import numpy as np
import torch
import inspect
from utils.CentralizedUtils import (evaluate_centralized_adam,
                                    compute_centralized_best_single_channel_rate,
                                    compute_decentralized_best_single_channel_rate,
                                    compute_equal_power_bound,
                                    compute_centralized_greedy_power_rate,
                                    compute_decentralized_greedy_power_rate,
                                    evaluate_ffn)
from utils.PathUtils import find_all_paths, paths_to_tensor
from utils.MetricUtils import calc_sum_rate
from utils.DataUtils import mean_var_over_dataset
from models.GraphNetAux import _compute_rates_per_layer
from Multicast.SubGraphs import find_multicast_subgraphs
from Multicommodity.Objective import edge_rates_multicommodity, objective_multicommodity

# -------------------------------
# Helpers
# -------------------------------


# -------------------------------
# Diagnostics helpers
# -------------------------------

def _as_int_list(x):
    if isinstance(x, torch.Tensor):
        return x.view(-1).detach().cpu().tolist()
    if isinstance(x, (list, tuple)):
        if len(x) == 1 and isinstance(x[0], (list, tuple)):
            x = x[0]
        return [int(v) for v in x]
    return [int(x)]


def _tx_rx_lists_for_problem(tx, rx, problem):
    """Return aligned source/destination lists for multi-message diagnostics."""
    if isinstance(tx, torch.Tensor):
        tx = tx.view(-1).tolist() if tx.numel() > 1 else int(tx.item())
    if isinstance(rx, torch.Tensor):
        rx = rx.view(-1).tolist() if rx.numel() > 1 else int(rx.item())

    if problem == "multi":
        rx_list = _as_int_list(rx)
        tx_list = [int(tx)] * len(rx_list) if not isinstance(tx, (list, tuple)) else _as_int_list(tx)
        if len(tx_list) == 1 and len(rx_list) > 1:
            tx_list = tx_list * len(rx_list)
        return tx_list, rx_list
    if problem == "converge":
        tx_list = _as_int_list(tx)
        rx_list = _as_int_list(rx)
        if len(rx_list) != 1:
            raise ValueError("For problem='converge', rx must contain exactly one receiver.")
        return tx_list, [rx_list[0]] * len(tx_list)
    if problem == "multiunicast":
        tx_list = _as_int_list(tx)
        rx_list = _as_int_list(rx)
        if len(tx_list) != len(rx_list):
            raise ValueError("For problem='multiunicast', len(tx) must equal len(rx).")
        return tx_list, rx_list
    return _as_int_list(tx), _as_int_list(rx)


def _build_eval_structures(idx, data, B, problem, device, cache):
    """Build/cached paths/subgraphs for a single sample. Topology-only."""
    hit = cache.get(idx)
    if hit is not None:
        return hit

    adj = data.adj_matrix
    tx = data.tx
    rx = data.rx
    if isinstance(tx, torch.Tensor):
        tx = tx.view(-1).tolist() if tx.numel() > 1 else int(tx.item())
    if isinstance(rx, torch.Tensor):
        rx = rx.view(-1).tolist() if rx.numel() > 1 else int(rx.item())

    paths = None
    subgraphs_per_band = None
    paths_k = None
    skip = False

    if problem == "single":
        raw_paths = find_all_paths(adj.cpu(), tx, rx)
        if not raw_paths:
            skip = True
        else:
            paths = paths_to_tensor(raw_paths, device)

    elif problem == "multicast":
        subgraphs = find_multicast_subgraphs(adj, tx, rx)
        if (subgraphs is None) or (len(subgraphs) == 0):
            skip = True
        else:
            subgraphs_per_band = [subgraphs for _ in range(B)]

    elif problem in {"multi", "converge", "multiunicast"}:
        tx_list, rx_list = _tx_rx_lists_for_problem(tx, rx, problem)
        paths_k = []
        has_any_path = False
        for s, t in zip(tx_list, rx_list):
            raw_paths_k = find_all_paths(adj.cpu(), int(s), int(t))
            if raw_paths_k:
                has_any_path = True
                paths_k.append(paths_to_tensor(raw_paths_k, device))
            else:
                paths_k.append(torch.empty((0, 0), dtype=torch.long, device=device))
        if not has_any_path:
            skip = True
    else:
        raise ValueError(f"Unknown problem type: {problem}")

    out = (paths, subgraphs_per_band, paths_k, skip)
    cache[idx] = out
    return out


def _call_compute_rates_for_eval(
    model, d, paths, subgraphs_per_band, paths_k, problem, reduce, dec_rounds,
    decentralized_gnn_inference: bool,
    hard_candidate_gnn_inference: bool,
):
    """Call _compute_rates_per_layer with the requested candidate-routing inference mode."""
    kwargs = dict(
        paths=paths,
        subgraphs_per_band=subgraphs_per_band,
        paths_k=paths_k,
        problem=problem,
        tau_min=0.0,
        tau_max=0.0,
        reduce=reduce,
    )
    # New GraphNetAux supports these flags. Keep fallback for older files.
    if (
        decentralized_gnn_inference
        and getattr(model, "z_mode", "edge") == "candidate"
        and problem in {"multi", "converge", "multiunicast"}
    ):
        kwargs.update(
            decentralized_inference=True,
            max_route_hops=dec_rounds,
            local_candidate_routing=True,
            hard_candidate_routing=hard_candidate_gnn_inference,
        )
    try:
        return _compute_rates_per_layer(model, d, **kwargs)
    except TypeError as exc:
        if (
            decentralized_gnn_inference
            and getattr(model, "z_mode", "edge") == "candidate"
            and problem in {"multi", "converge", "multiunicast"}
        ):
            raise RuntimeError(
                "Decentralized candidate GNN inference was requested, but the imported "
                "models.GraphNetAux._compute_rates_per_layer does not accept the required "
                "routing flags. Replace models/GraphNetAux.py with the patched local-training "
                "version before trusting this benchmark. The old silent fallback would evaluate "
                "global candidate routing instead of L-hop routing."
            ) from exc
        return _compute_rates_per_layer(model, d, **kwargs)


def _safe_float(x, default=0.0):
    try:
        if isinstance(x, torch.Tensor):
            if x.numel() == 0:
                return default
            x = x.detach().float().cpu().item()
        if np.isfinite(x):
            return float(x)
    except Exception:
        pass
    return float(default)


def _r_bk_multicommodity(h, p, z, sigma, adj, paths_k):
    """Return end-to-end rates per [B,K] using the same hard path aggregation as eval."""
    R = edge_rates_multicommodity(h=h, p=p, z=z, sigma=sigma, adj=adj)
    B, K, _, _ = R.shape
    device = R.device
    r_bk = torch.zeros(B, K, device=device)
    for k in range(K):
        paths = paths_k[k]
        if paths is None or paths.numel() == 0:
            continue
        paths = paths.to(device)
        edge_start = paths[:, :-1]
        edge_end = paths[:, 1:]
        if edge_start.numel() == 0:
            continue
        valid_mask = (edge_start >= 0) & (edge_end >= 0)
        link_rates = R[:, k][:, edge_start, edge_end]  # [B,P,L-1]
        masked_link_rates = torch.where(
            valid_mask.unsqueeze(0),
            link_rates,
            torch.full_like(link_rates, float("inf")),
        )
        path_vals, _ = masked_link_rates.min(dim=2)  # [B,P]
        r_bk[:, k], _ = path_vals.max(dim=1)
    return r_bk


def _extract_selected_paths_from_power(P, Z, paths_k, max_route_hops=None, eps=1e-12,
                                       band_active_frac=0.05):
    """Infer selected path(s)/band(s) per commodity from power-on-path mass.

    Under Eq. (9) a commodity uses each band independently (per-band route selection), so it can
    legitimately spread power across several bands. This reports BOTH:
      - the single dominant band/path (``selected_bands``/``selected_path_length``/... ) kept for
        backward compatibility, and
      - true multi-band occupancy: for every band, that band's own best candidate route, how many
        bands are active per commodity, and the dominant band's share of the commodity power.
    ``off_route_fraction`` is measured against the UNION of every band's selected route (not just the
    dominant band), so multi-band power is no longer miscounted as leakage; for a route-consistent Z
    (candidate head) it is ~0, while for a dense Z (optimizer/greedy) it reports genuine off-route mass.
    A band counts as active for a commodity when it carries >= ``band_active_frac`` of that
    commodity's total power.
    """
    if P is None or P.dim() != 4 or paths_k is None:
        return {}
    device = P.device
    B, K, n, _ = P.shape
    if Z is None:
        Z = torch.ones_like(P)
    active_power = (P.float().pow(2) * Z.float().clamp(0, 1)).detach()

    selected_lengths, selected_bands, selected_path_power, within_hops = [], [], [], []
    valid_lhop_counts, selected_paths = [], []
    off_route_fracs = []
    n_active_bands_pc, dominant_band_frac_pc, per_band_route_len_pc = [], [], []

    for k in range(K):
        paths = paths_k[k]
        ap_k = active_power[:, k]                       # [B,n,n]
        total_k = ap_k.sum().clamp_min(eps)
        band_totals = ap_k.sum(dim=(1, 2))              # [B] total commodity power in each band

        # Per-band best candidate route: for every band pick the candidate path carrying the most
        # power in THAT band (bands may choose different routes, per Eq. 9).
        best_band_pow = torch.zeros(B, device=device)
        best_band_len = [0] * B
        best_band_edges = [[] for _ in range(B)]
        valid_lhop = 0
        if paths is not None and paths.numel() > 0:
            paths = paths.to(device)
            for row in range(paths.shape[0]):
                seq = [int(x) for x in paths[row].detach().cpu().tolist() if int(x) >= 0]
                if len(seq) < 2:
                    continue
                plen = len(seq) - 1
                if max_route_hops is None or plen <= max_route_hops:
                    valid_lhop += 1
                rows = torch.tensor(seq[:-1], dtype=torch.long, device=device)
                cols = torch.tensor(seq[1:], dtype=torch.long, device=device)
                vals = ap_k[:, rows, cols].sum(dim=1)   # [B] this path's power per band
                improved = vals > best_band_pow
                for b in torch.nonzero(improved, as_tuple=False).flatten().tolist():
                    best_band_pow[b] = vals[b]
                    best_band_len[b] = plen
                    best_band_edges[b] = list(zip(seq[:-1], seq[1:]))

        # Dominant band = the band whose best route carries the most power (matches the old field).
        if float(best_band_pow.max()) > 0:
            dom_b = int(best_band_pow.argmax().item())
            best_b = dom_b
            best_len = best_band_len[dom_b]
            best_edges = best_band_edges[dom_b]
            best_val = float(best_band_pow[dom_b])
        else:
            best_b, best_len, best_edges, best_val = None, 0, [], 0.0

        selected_bands.append(best_b)
        selected_lengths.append(best_len)
        selected_path_power.append(best_val if best_b is not None else 0.0)
        within_hops.append(bool(max_route_hops is not None and best_len > 0 and best_len <= max_route_hops))
        valid_lhop_counts.append(int(valid_lhop))
        selected_paths.append(best_edges)

        # True multi-band occupancy.
        n_active = int((band_totals >= band_active_frac * total_k).sum().item())
        n_active_bands_pc.append(n_active)
        dominant_band_frac_pc.append(_safe_float(band_totals.max() / total_k, 0.0)
                                     if float(total_k) > eps else 0.0)
        per_band_route_len_pc.append([int(l) for l in best_band_len])

        # Off-route fraction vs the union of every band's selected route.
        on_power = float(best_band_pow.sum())           # sum over bands of on-(that band's)route power
        off_route_fracs.append(_safe_float((float(total_k) - on_power) / (float(total_k) + eps), 0.0)
                               if on_power > 0 else (1.0 if float(total_k) > eps else 0.0))

    return dict(
        selected_path_lengths=selected_lengths,
        selected_bands=selected_bands,
        selected_path_power=selected_path_power,
        selected_paths=selected_paths,
        selected_path_within_horizon=within_hops,
        valid_lhop_candidate_counts=valid_lhop_counts,
        off_route_fraction_per_commodity=off_route_fracs,
        off_route_fraction=float(np.mean(off_route_fracs)) if off_route_fracs else 0.0,
        # New multi-band fields:
        n_active_bands_per_commodity=n_active_bands_pc,
        mean_active_bands=float(np.mean(n_active_bands_pc)) if n_active_bands_pc else 0.0,
        dominant_band_power_fraction_per_commodity=dominant_band_frac_pc,
        dominant_band_power_fraction=float(np.mean(dominant_band_frac_pc)) if dominant_band_frac_pc else 0.0,
        per_band_selected_route_lengths=per_band_route_len_pc,
    )


def _allocation_diagnostics(method, data, P, Z, paths_k, sigma, problem, reduce, max_route_hops=None, rate_override=None):
    """Build per-sample diagnostics for a P/Z allocation."""
    if P is None:
        return {"method": method, "rate": _safe_float(rate_override, 0.0)}
    if isinstance(P, (tuple, list)) and len(P) >= 2:
        P, Z = P[0], P[1]
    P = P.detach()
    if Z is not None:
        Z = Z.detach()

    out = {"method": method, "rate": _safe_float(rate_override, 0.0)}
    if problem not in {"multi", "converge", "multiunicast"} or P.dim() != 4:
        out["total_power"] = _safe_float((P.float() ** 2).sum())
        return out

    h = data.links_matrix.to(P.device)
    adj = data.adj_matrix.to(P.device)
    sigma_t = torch.as_tensor(sigma, device=P.device) if not isinstance(sigma, torch.Tensor) else sigma.to(P.device)
    if Z is None:
        Z = torch.ones_like(P)
    Z = Z.to(P.device).float().clamp(0, 1)
    P = P.float().clamp_min(0.0)

    try:
        r_bk = _r_bk_multicommodity(h, P, Z, sigma_t, adj, paths_k)
        R_k = r_bk.sum(dim=0)
        out.update({
            "fair_rate": _safe_float(R_k.min(), 0.0),
            "sum_rate": _safe_float(R_k.sum(), 0.0),
            "mean_rate": _safe_float(R_k.mean(), 0.0),
            "per_commodity_rates": [float(x) for x in R_k.detach().cpu().tolist()],
            "served_commodities": int((R_k > 1e-6).sum().item()),
            "min_commodity_index": int(R_k.argmin().item()) if R_k.numel() else -1,
        })
        out["rate"] = out.get(f"{reduce}_rate", out["fair_rate"] if reduce == "fair" else out["mean_rate"])
    except Exception as exc:
        out["diagnostic_error"] = repr(exc)

    p2z = (P.pow(2) * Z).detach()
    power_k = p2z.sum(dim=(0, 2, 3))
    out["total_power_per_commodity"] = [float(x) for x in power_k.cpu().tolist()]
    out["total_power"] = _safe_float(power_k.sum(), 0.0)
    out.update(_extract_selected_paths_from_power(P, Z, paths_k, max_route_hops=max_route_hops))
    return out


def _summarize_method_diagnostics(diags):
    """Compact mean diagnostics for one method/SNR."""
    if not diags:
        return {}
    def mean_field(name):
        vals = [d.get(name) for d in diags if isinstance(d.get(name), (int, float)) and np.isfinite(d.get(name))]
        return float(np.mean(vals)) if vals else 0.0
    out = {
        "mean_rate": mean_field("rate"),
        "mean_fair_rate": mean_field("fair_rate"),
        "mean_sum_rate": mean_field("sum_rate"),
        "mean_served_commodities": mean_field("served_commodities"),
        "mean_total_power": mean_field("total_power"),
        "mean_off_route_fraction": mean_field("off_route_fraction"),
        "mean_active_bands": mean_field("mean_active_bands"),
        "mean_dominant_band_power_fraction": mean_field("dominant_band_power_fraction"),
    }
    # Average path length over all selected positive-length commodity paths.
    lengths = []
    within = []
    valid_counts = []
    for d in diags:
        lengths.extend([x for x in d.get("selected_path_lengths", []) if x > 0])
        within.extend([bool(x) for x in d.get("selected_path_within_horizon", [])])
        valid_counts.extend(d.get("valid_lhop_candidate_counts", []))
    out["mean_selected_path_length"] = float(np.mean(lengths)) if lengths else 0.0
    out["frac_selected_paths_within_horizon"] = float(np.mean(within)) if within else 0.0
    out["mean_valid_lhop_candidates"] = float(np.mean(valid_counts)) if valid_counts else 0.0
    return out


def _diagnose_method_store(method, dataset, store, aux_store, sigma, problem, reduce, B, device, struct_cache, max_route_hops, rates=None):
    diags = []
    for i, d in enumerate(dataset):
        if i >= len(store):
            break
        paths, subgraphs_per_band, paths_k, skip = _build_eval_structures(i, d, B, problem, device, struct_cache)
        P = store[i]
        Z = None
        if isinstance(P, (tuple, list)) and len(P) >= 2:
            P, Z = P[0], P[1]
        elif aux_store is not None and i < len(aux_store):
            aux = aux_store[i]
            if isinstance(aux, dict):
                Z = aux.get("Z", None)
        rate_override = rates[i] if rates is not None and i < len(rates) else None
        if skip:
            diags.append({"method": method, "rate": 0.0, "skip": True})
            continue
        diags.append(_allocation_diagnostics(method, d, P.to(device), Z.to(device) if isinstance(Z, torch.Tensor) else Z,
                                             paths_k, sigma, problem, reduce, max_route_hops, rate_override))
    return diags


def evaluate_across_snr(
    dataset,
    model,
    B,
    snr_db_list,
    *,
    problem: str = "single",      # "single" | "multicast" | "multi" | "converge" | "multiunicast"
    reduce: str = "fair",         # K>1 message reduction: "fair"=eq10 min | "sum"/"mean" throughput
    collect_diagnostics: bool = True,
    diagnostics_snr=None,          # None -> all SNRs; otherwise iterable of SNRs to store raw diagnostics for.
    decentralized_gnn_inference=None,
    max_route_hops=None,
    hard_candidate_gnn_inference: bool = True,
    optimizer_lr_grid=None,       # Optional list of LRs for the centralized optimizer line search.
    ffn_model=None,               # Optional trained FFNPowerAllocator; None -> FFN curve stays 0.
    ffn_dataset=None,             # FFNDataset aligned with `dataset` (SAME CSI: true or estimated).
):
    """
    Sequential evaluation across SNR values.

    In addition to the old mean-rate curves, this diagnostic version can also store
    per-sample multi-message diagnostics under results["diagnostics"][snr_db]. The
    mean curves keep the original keys, so existing plotting code should continue to work.
    """
    device = next(model.parameters()).device
    method_keys = [
        "gnn",
        "ffn",
        "centralized",
        "strongest bottleneck",
        "strongest bottleneck decentralized",
        "equal power",
        "greedy maxlink",
        "greedy maxlink decentralized",
    ]
    results = {k: {} for k in method_keys}
    # Per-SNR standard error of the mean rate (std / sqrt(N)), same keys as the mean
    # curves, so plotting code can draw confidence intervals. `n_test` is the number of
    # test networks the means/SEMs are computed over (for the reviewer's stated N).
    results["sem"] = {k: {} for k in method_keys}
    results["n_test"] = len(dataset)

    def _record(method_key, snr_key, per_sample_rates):
        """Store the mean rate and its standard error for one method at one SNR."""
        arr = np.asarray(list(per_sample_rates), dtype=float)
        results[method_key][snr_key] = float(arr.mean()) if arr.size else 0.0
        # ddof=1 sample std; SEM = std / sqrt(N). Undefined for N<2 -> 0.
        if arr.size > 1:
            results["sem"][method_key][snr_key] = float(arr.std(ddof=1) / np.sqrt(arr.size))
        else:
            results["sem"][method_key][snr_key] = 0.0

    if collect_diagnostics:
        results["diagnostics"] = {}
        results["diagnostics_summary"] = {}

    diagnostics_snr_set = None if diagnostics_snr is None else {int(x) for x in diagnostics_snr}

    # FFN benchmark loader (optional). ffn_dataset already carries the SAME CSI as `dataset`
    # (true or estimated); evaluate_ffn forwards AND scores on it, at the per-SNR sigma below.
    ffn_loader = None
    if ffn_model is not None and ffn_dataset is not None:
        from torch.utils.data import DataLoader as _FFNLoader
        ffn_loader = _FFNLoader(ffn_dataset, batch_size=1, shuffle=False)
        ffn_model.eval()
        print(f"[FFN] benchmark enabled: {len(ffn_dataset)} samples.")

    mean_channel_var = mean_var_over_dataset(dataset)
    gnn_struct_cache = {}
    heuristic_paths_cache = {}
    diag_struct_cache = {}

    inferred_L = int(getattr(model, "num_layers", 2))
    dec_rounds = inferred_L if max_route_hops is None else int(max_route_hops)
    if decentralized_gnn_inference is None:
        decentralized_gnn_inference = (
            getattr(model, "z_mode", "edge") == "candidate"
            and problem in {"multi", "converge", "multiunicast"}
        )
    bottleneck_rounds = None
    print(f"[decentralized greedy horizon] max routing rounds = L = {inferred_L}"
          f" | strongest-bottleneck horizon = {bottleneck_rounds if bottleneck_rounds is not None else 'exact'}")
    print(
        "[GNN candidate inference] "
        f"z_mode={getattr(model, 'z_mode', 'edge')} | "
        f"decentralized_gnn_inference={bool(decentralized_gnn_inference)} | "
        f"max_route_hops={dec_rounds} | "
        f"hard_candidate_gnn_inference={bool(hard_candidate_gnn_inference)} | "
        f"effective_for_gnn={bool(decentralized_gnn_inference) and getattr(model, 'z_mode', 'edge') == 'candidate' and problem in {'multi', 'converge', 'multiunicast'}}"
    )

    for snr_db in snr_db_list:
        snr = 10.0 ** (snr_db / 10.0)
        sigma2 = mean_channel_var / snr
        sigma = sigma2 ** 0.5
        print(f"SNR: {snr_db} dB")
        keep_raw_diag = collect_diagnostics and (diagnostics_snr_set is None or int(snr_db) in diagnostics_snr_set)
        snr_diags = {}

        # =================================================
        # 1) GNN mean rate + diagnostics
        # =================================================
        model.eval()
        with torch.no_grad():
            rates = []
            gnn_p_store = []
            gnn_z_store = []
            gnn_rate_store = []
            for i, d in enumerate(dataset):
                d.sigma = torch.tensor(sigma, device=device)
                paths, subgraphs_per_band, paths_k, skip = _build_eval_structures(i, d, B, problem, device, gnn_struct_cache)
                if skip:
                    rates.append(0.0)
                    # Always store (aligned with dataset index) so the GNN allocation is
                    # available to warm-start the centralized optimizer at every SNR.
                    gnn_p_store.append(None); gnn_z_store.append(None); gnn_rate_store.append(0.0)
                    continue

                setattr(d, "problem", problem)
                d = d.to(device)
                rates_per_layer, p_list, z_list = _call_compute_rates_for_eval(
                    model, d, paths, subgraphs_per_band, paths_k, problem, reduce, dec_rounds,
                    decentralized_gnn_inference=bool(decentralized_gnn_inference),
                    hard_candidate_gnn_inference=bool(hard_candidate_gnn_inference),
                )
                layer_rates = torch.stack(rates_per_layer)
                best_idx = int(layer_rates.argmax().item())
                rate = float(layer_rates[best_idx].item())
                rates.append(rate)
                gnn_p_store.append(p_list[best_idx].detach().cpu())
                if z_list and best_idx < len(z_list):
                    gnn_z_store.append(z_list[best_idx].detach().cpu())
                else:
                    gnn_z_store.append(None)
                gnn_rate_store.append(rate)
            _record("gnn", snr_db, rates)
            if keep_raw_diag:
                gnn_diags = []
                for i, d in enumerate(dataset):
                    paths, subgraphs_per_band, paths_k, skip = _build_eval_structures(i, d, B, problem, device, diag_struct_cache)
                    if skip or i >= len(gnn_p_store) or gnn_p_store[i] is None:
                        gnn_diags.append({"method": "gnn", "rate": 0.0, "skip": True})
                        continue
                    P = gnn_p_store[i].to(device)
                    Z = gnn_z_store[i].to(device) if isinstance(gnn_z_store[i], torch.Tensor) else None
                    gnn_diags.append(_allocation_diagnostics("gnn", d.to(device), P, Z, paths_k, sigma, problem, reduce,
                                                              max_route_hops=dec_rounds, rate_override=gnn_rate_store[i]))
                snr_diags["gnn"] = gnn_diags

        # =================================================
        # 2) FFN mean rate
        # =================================================
        if ffn_loader is not None:
            ffn_rates, _ = evaluate_ffn(
                ffn_model, ffn_loader, sigma_noise=sigma, problem=problem, reduce=reduce,
            )
            _record("ffn", snr_db, ffn_rates if len(ffn_rates) else [0.0])
        else:
            results["ffn"][snr_db] = 0
            results["sem"]["ffn"][snr_db] = 0.0

        # ==============================================
        # 3) Centralized ADAM benchmark
        # ==============================================
        # Warm-start the multi-message optimizer from the GNN allocation (multi-init with
        # greedy). Guarantees optimizer >= max(greedy, GNN) via classic_opt's best-iterate
        # return, so the optimizer stays a valid upper reference instead of being beaten by
        # the GNN / collapsing at high SNR. Ignored internally for single/multicast.
        opt_warm_start = list(zip(gnn_p_store, gnn_z_store)) if problem in {"multi", "converge", "multiunicast"} else None
        adam_out = evaluate_centralized_adam(
            dataset,
            B,
            noise_std=sigma,
            num_iterations=1000,
            problem=problem,
            reduce=reduce,
            return_aux=keep_raw_diag,
            warm_start=opt_warm_start,
            lr_grid=optimizer_lr_grid,
        )
        if keep_raw_diag:
            adam_rates, adam_p, adam_aux = adam_out
        else:
            adam_rates, adam_p = adam_out
            adam_aux = None
        _record("centralized", snr_db, adam_rates)
        if keep_raw_diag:
            snr_diags["centralized"] = _diagnose_method_store(
                "centralized", dataset, adam_p, adam_aux, sigma, problem, reduce, B, device, diag_struct_cache,
                dec_rounds, rates=adam_rates)

        # ==============================================
        # 4) Strongest bottleneck lower bound
        # ==============================================
        bottleneck_rates, bottleneck_store = compute_centralized_best_single_channel_rate(
            dataset,
            problem=problem,
            sigma_noise=sigma,
            paths_cache=heuristic_paths_cache,
            reduce=reduce,
        )
        _record("strongest bottleneck", snr_db, bottleneck_rates)
        if keep_raw_diag:
            snr_diags["strongest bottleneck"] = _diagnose_method_store(
                "strongest bottleneck", dataset, bottleneck_store, None, sigma, problem, reduce, B, device,
                diag_struct_cache, dec_rounds, rates=bottleneck_rates)

        bottleneck_rates_decentralized, bottleneck_dec_store, bottleneck_dec_aux = compute_decentralized_best_single_channel_rate(
            dataset,
            problem=problem,
            sigma_noise=sigma,
            max_iters=bottleneck_rounds,
            paths_cache=heuristic_paths_cache,
            reduce=reduce,
        )
        _record("strongest bottleneck decentralized", snr_db, bottleneck_rates_decentralized)
        if keep_raw_diag:
            snr_diags["strongest bottleneck decentralized"] = _diagnose_method_store(
                "strongest bottleneck decentralized", dataset, bottleneck_dec_store, bottleneck_dec_aux, sigma, problem,
                reduce, B, device, diag_struct_cache, dec_rounds, rates=bottleneck_rates_decentralized)

        # ==============================================
        # 5) Equal-power heuristic
        # ==============================================
        rates_equal_power, equal_store = compute_equal_power_bound(
            dataset,
            sigma_noise=sigma,
            problem=problem,
            paths_cache=heuristic_paths_cache,
            reduce=reduce,
        )
        _record("equal power", snr_db, rates_equal_power)
        if keep_raw_diag:
            snr_diags["equal power"] = _diagnose_method_store(
                "equal power", dataset, equal_store, None, sigma, problem, reduce, B, device, diag_struct_cache,
                dec_rounds, rates=rates_equal_power)

        # ==============================================
        # 6) Greedy max-link benchmark
        # ==============================================
        rates_greedy, greedy_store = compute_centralized_greedy_power_rate(
            dataset,
            sigma_noise=sigma,
            problem=problem,
            paths_cache=heuristic_paths_cache,
            reduce=reduce,
        )
        _record("greedy maxlink", snr_db, rates_greedy)
        if keep_raw_diag:
            snr_diags["greedy maxlink"] = _diagnose_method_store(
                "greedy maxlink", dataset, greedy_store, None, sigma, problem, reduce, B, device, diag_struct_cache,
                dec_rounds, rates=rates_greedy)

        rates_greedy_decentralized, greedy_dec_store = compute_decentralized_greedy_power_rate(
            dataset,
            sigma_noise=sigma,
            problem=problem,
            max_iters=dec_rounds,
            paths_cache=heuristic_paths_cache,
            reduce=reduce,
        )
        _record("greedy maxlink decentralized", snr_db, rates_greedy_decentralized)
        if keep_raw_diag:
            snr_diags["greedy maxlink decentralized"] = _diagnose_method_store(
                "greedy maxlink decentralized", dataset, greedy_dec_store, None, sigma, problem, reduce, B, device,
                diag_struct_cache, dec_rounds, rates=rates_greedy_decentralized)

        if keep_raw_diag:
            results["diagnostics"][snr_db] = snr_diags
            results["diagnostics_summary"][snr_db] = {
                method: _summarize_method_diagnostics(diags)
                for method, diags in snr_diags.items()
            }
            print("[diag summary]", snr_db, results["diagnostics_summary"][snr_db].get("gnn", {}))

    return results


def time_model_compare(dataset, big_model, small_model, snr_db_list):
    """
    Sequential evaluation across a list of SNR values.
    The goal is to test the scalability of ChainedGNN (evaluate data samples of large topology using a model that was trained on a smaller topology).

    Args:
        dataset: Dataset based on large topology (already on CPU or GPU as needed).
        big_model: Trained GNN model on a 'large' topology.
        small_model: Trained GNN model on a 'small' topology.
        snr_db_list: List of SNR values in dB.

    Returns:
        dict: { "big": {snr_db: mean_rate}, "small": {snr_db: mean_rate}}
    """
    assert big_model.B == small_model.B , "models must have the same B attribute"
    device = next(big_model.parameters()).device
    results = {"big": {}, "small": {}}

    # --- compute mean channel variance for noise scaling ---
    mean_channel_var = mean_var_over_dataset(dataset)

    for snr_db in snr_db_list:
        snr = 10.0 ** (snr_db / 10.0)
        sigma2 = mean_channel_var / snr
        sigma = sigma2 ** 0.5
        print(f'SNR: {snr_db} dB')

        # GNN mean rate
        big_model.eval()
        small_model.eval()
        with torch.no_grad():
            big_rates = []
            small_rates = []
            for d in dataset:
                d.sigma = torch.tensor(sigma, device=device)
                d = d.to(device)

                paths = find_all_paths(d.adj_matrix, d.tx, d.rx)
                paths = paths_to_tensor(paths, device)

                big_gnn_rates, _ = _compute_rates_per_layer(big_model, d, paths)
                big_rate  = torch.stack(big_gnn_rates).max().item()
                big_rates.append(big_rate)

                small_gnn_rates, _ = _compute_rates_per_layer(small_model, d, paths)
                small_rate = torch.stack(small_gnn_rates).max().item()
                small_rates.append(small_rate)

            results["big"][snr_db] = float(np.mean(big_rates))
            results["small"][snr_db] = float(np.mean(small_rates))

    return results

@torch.inference_mode()
def est_true_model_compare(true_dataset, est_dataset, true_model, est_model, snr_db_list):
    """
    Compare a model trained on true CSI vs a model trained on estimated CSI across SNR values.
    Args:
        true_dataset: Dataset based on true CSI (already on CPU or GPU as needed).
        est_dataset: Dataset based on estimated CSI (already on CPU or GPU as needed).
        true_model: Trained GNN model on true CSI..
        est_model: Trained GNN model on estimated CSI.
        snr_db_list: List of SNR values in dB.

    Returns:
        dict: { "true": {snr_db: mean_rate}, "est": {snr_db: mean_rate}}
    """
    assert true_model.B == est_model.B, "models must have the same B"
    device = next(true_model.parameters()).device
    true_model.eval()
    est_model.eval()

    # --- SNR normalization from TRUE dataset (fairness) ---
    # mean_var is a scalar variance; convert to std for calc_sum_rate downstream.
    true_mean_var = mean_var_over_dataset(true_dataset)  # scalar VAR
    results = {"true": {}, "est": {}}

    # sanity: we rely on index alignment across datasets
    if len(true_dataset) != len(est_dataset):
        raise ValueError("true_dataset and est_dataset length mismatch; cannot align by index.")

    for snr_db in snr_db_list:
        print(f'SNR: {snr_db} dB')
        snr = 10.0 ** (snr_db / 10.0)

        # σ^2 = mean_var / SNR  --> σ = sqrt(σ^2)
        sigma2 = true_mean_var / snr
        sigma = float(sigma2 ** 0.5)
        sigma_t = torch.tensor(sigma, device=device)

        # ----- TRUE track: powers from TRUE model on TRUE inputs; score on TRUE CSI -----
        true_rates = []
        for i in range(len(true_dataset)):
            d_true = true_dataset[i].to(device)

            # (re)compute paths as requested
            paths = find_all_paths(d_true.adj_matrix, d_true.tx, d_true.rx)
            if len(paths) == 0:
                continue
            paths = paths_to_tensor(paths, device)

            # get powers from the true-CSI-trained model (ignore internal rates)
            _, P_list_true = _compute_rates_per_layer(true_model, d_true, paths)
            P_true = P_list_true[-1]

            r_true = calc_sum_rate(
                h_arr=d_true.links_matrix,  # TRUE CSI
                p_arr=P_true,
                sigma=sigma_t,  # std
                paths_tensor=paths,
                B=true_model.B,
                tau=0
            )
            true_rates.append(float(r_true.item()))
        results["true"][snr_db] = float(np.mean(true_rates)) if true_rates else float("nan")

        # ----- EST track: powers from EST model on EST inputs; score on TRUE CSI -----
        est_rates = []
        for i in range(len(est_dataset)):
            d_est = est_dataset[i].to(device)
            d_true = true_dataset[i].to(device)  # same graph/topology order

            paths = find_all_paths(d_est.adj_matrix, d_est.tx, d_est.rx)
            if len(paths) == 0:
                continue
            paths = paths_to_tensor(paths, device)

            _, P_list_est = _compute_rates_per_layer(est_model, d_est, paths)
            P_est = P_list_est[-1]

            r_est = calc_sum_rate(
                h_arr=d_true.links_matrix,  # score on TRUE CSI
                p_arr=P_est,
                sigma=sigma_t,  # std
                paths_tensor=paths,
                B=est_model.B,
                tau=0
            )
            est_rates.append(float(r_est.item()))
        results["est"][snr_db] = float(np.mean(est_rates)) if est_rates else float("nan")

    return results

def evaluate_models_across_snr(
    dataset,
    models,
    B,
    snr_db_list,
    *,
    problem: str = "single",      # "single" | "multicast" | "multi"
    multi_mode: str = "global",   # kept for API symmetry (not used inside GNN eval)
    take_best_layer: bool = True, # matches your current behavior
):
    """
    Evaluate achieved mean rate vs SNR for multiple trained GNN models (ablation study).

    Args:
        dataset: iterable of graph data objects.
        models: list of tuples [(name, model), ...] OR dict {name: model}.
                Each model must already be on the correct device.
        B: number of bands.
        snr_db_list: list of SNR values in dB.
        problem: "single", "multicast", or "multi".
        multi_mode: unused here; kept to match your existing signature style.
        take_best_layer: if True, per sample take max over layer outputs (as in your code).
                         if False, take last layer rate.

    Returns:
        dict: {
            "models": {
                model_name: {snr_db: mean_rate, ...},
                ...
            }
        }
    """
    # Normalize models input
    if isinstance(models, dict):
        model_items = list(models.items())
    else:
        model_items = list(models)  # list of (name, model)

    if len(model_items) == 0:
        raise ValueError("models is empty.")

    # Compute mean channel variance once (consistent noise scaling)
    mean_channel_var = mean_var_over_dataset(dataset)

    results = {"models": {name: {} for name, _ in model_items}}

    for snr_db in snr_db_list:
        snr = 10.0 ** (snr_db / 10.0)
        sigma2 = mean_channel_var / snr
        sigma = float(sigma2 ** 0.5)
        print(f"SNR: {snr_db} dB")

        for name, model in model_items:
            device = next(model.parameters()).device
            model.eval()

            with torch.no_grad():
                rates = []
                for d in dataset:
                    # Set noise (same behavior as your baseline function)
                    d.sigma = torch.tensor(sigma, device=device)

                    adj = d.adj_matrix
                    tx = d.tx
                    rx = d.rx

                    paths = None
                    subgraphs_per_band = None
                    paths_k = None

                    if problem == "single":
                        raw_paths = find_all_paths(adj.cpu(), tx, rx)
                        if not raw_paths:
                            rates.append(0.0)
                            continue
                        paths = paths_to_tensor(raw_paths, device)

                    elif problem == "multicast":
                        # rx is list of receivers (kept for parity with your code)
                        if isinstance(rx, (list, tuple)):
                            rx_list = list(rx)
                        else:
                            rx_list = [rx]

                        subgraphs = find_multicast_subgraphs(d.adj_matrix, d.tx, d.rx)
                        if (subgraphs is None) or (len(subgraphs) == 0):
                            rates.append(0.0)
                            continue
                        subgraphs_per_band = [subgraphs for _ in range(B)]

                    elif problem == "multi":
                        if isinstance(rx, (list, tuple)):
                            rx_list = list(rx)
                        else:
                            rx_list = [rx]
                        K = len(rx_list)

                        paths_k = []
                        has_any_path = False
                        for rx_k in rx_list:
                            raw_paths_k = find_all_paths(adj.cpu(), tx, rx_k)
                            if raw_paths_k:
                                has_any_path = True
                                paths_k.append(paths_to_tensor(raw_paths_k, device))
                            else:
                                paths_k.append(torch.empty((0, 0), dtype=torch.long, device=device))

                        if not has_any_path:
                            rates.append(0.0)
                            continue

                    else:
                        raise ValueError(f"Unknown problem type: {problem}")

                    setattr(d, "problem", problem)
                    d = d.to(device)

                    rates_per_layer, _, _ = _compute_rates_per_layer(
                        model,
                        d,
                        paths=paths,
                        subgraphs_per_band=subgraphs_per_band,
                        paths_k=paths_k,
                        problem=problem,
                        tau_min=0.0,
                        tau_max=0.0,
                    )

                    layer_rates = torch.stack(rates_per_layer)  # [L]
                    if take_best_layer:
                        rate = layer_rates.max().item()
                    else:
                        rate = layer_rates[-1].item()
                    rates.append(rate)

            results["models"][name][snr_db] = float(np.mean(rates))

    return results



