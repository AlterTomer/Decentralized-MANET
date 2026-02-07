import numpy as np
import torch
from utils.CentralizedUtils import evaluate_centralized_adam, compute_strongest_bottleneck_rate, compute_equal_power_bound, compute_greedy_power_rate
from utils.PathUtils import find_all_paths, paths_to_tensor
from utils.MetricUtils import calc_sum_rate
from utils.DataUtils import mean_var_over_dataset
from models.GraphNetAux import _compute_rates_per_layer
from Multicast.SubGraphs import find_multicast_subgraphs

# -------------------------------
# Helpers
# -------------------------------
def evaluate_across_snr(
    dataset,
    model,
    B,
    snr_db_list,
    *,
    problem: str = "single",      # "single" | "multicast" | "multi"
    multi_mode: str = "global",   # only used for problem=="multi" in bottleneck
):
    """
    Sequential evaluation across a list of SNR values.

    Args:
        dataset: iterable of graph data objects.
        model:   trained GNN model.
        B: number of bands.
        snr_db_list: list of SNR values in dB.
        problem: "single", "multicast", or "multi".
        multi_mode: for problem=="multi" in strongest-bottleneck:
            - "global": single best (b,k) over all commodities.
            - "per_commodity": unique band per commodity (if you use that mode).

    Returns:
        dict: {
            "gnn": {snr_db: mean_rate},
            "centralized": {snr_db: mean_rate},
            "strongest bottleneck": {snr_db: mean_rate},
            "equal power": {snr_db: mean_rate},
            "greedy maxlink": {snr_db: mean_rate}
        }
    """
    device = next(model.parameters()).device
    results = {
        "gnn": {},
        "centralized": {},
        "strongest bottleneck": {},
        "equal power": {},
        "greedy maxlink": {}
    }

    # --- compute mean channel variance for noise scaling ---
    mean_channel_var = mean_var_over_dataset(dataset)

    for snr_db in snr_db_list:
        snr = 10.0 ** (snr_db / 10.0)
        sigma2 = mean_channel_var / snr
        sigma = sigma2 ** 0.5
        print(f"SNR: {snr_db} dB")

        # =================================================
        # 1) GNN mean rate
        # =================================================
        model.eval()
        with torch.no_grad():
            rates = []
            for d in dataset:
                # Set noise
                d.sigma = torch.tensor(sigma, device=device)

                adj = d.adj_matrix
                tx = d.tx
                rx = d.rx

                # Problem-specific structures for _compute_rates_per_layer
                paths = None
                subgraphs_per_band = None
                paths_k = None

                if problem == "single":
                    # Tx → Rx paths
                    raw_paths = find_all_paths(adj.cpu(), tx, rx)
                    if not raw_paths:
                        # no connectivity → rate 0 for this sample
                        rates.append(0.0)
                        continue
                    paths = paths_to_tensor(raw_paths, device)

                elif problem == "multicast":
                    # rx is list of receivers
                    if isinstance(rx, (list, tuple)):
                        rx_list = list(rx)
                    else:
                        rx_list = [rx]

                    # One multicast subgraph set, replicated across bands
                    subgraphs = find_multicast_subgraphs(d.adj_matrix, d.tx, d.rx)
                    if (subgraphs is None) or (len(subgraphs) == 0):
                        rates.append(0.0)
                        continue
                    subgraphs_per_band = [subgraphs for _ in range(B)]

                elif problem == "multi":
                    # multicommodity: Tx→rx_k for each k
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
                        paths_k.append(paths_to_tensor(raw_paths_k, device) if raw_paths_k else
                                       torch.empty((0, 0), dtype=torch.long, device=device))
                    if not has_any_path:
                        rates.append(0.0)
                        continue

                else:
                    raise ValueError(f"Unknown problem type: {problem}")

                # Tag problem (if model cares about it)
                setattr(d, "problem", problem)
                d = d.to(device)

                # Call the new helper
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

                # Take best layer (as before)
                layer_rates = torch.stack(rates_per_layer)  # [L]
                rate = layer_rates.max().item()
                rates.append(rate)

            results["gnn"][snr_db] = float(np.mean(rates))

        # ==============================================
        # 2) Centralized ADAM benchmark (problem-aware)
        # ==============================================
        adam_rates, _ = evaluate_centralized_adam(
            dataset,
            B,
            noise_std=sigma,
            num_iterations=50,
            problem=problem,
        )
        results["centralized"][snr_db] = float(np.mean(adam_rates))

        # ==============================================
        # 3) Strongest bottleneck lower bound
        # ==============================================
        bottleneck_rates, _ = compute_strongest_bottleneck_rate(
            dataset,
            problem=problem,
            sigma_noise=sigma,
            multi_mode=multi_mode if problem == "multi" else "global",
        )
        # For multi+per_commodity you may want a different aggregation;
        # here we assume scalar-per-graph (global) or we just mean over all entries.
        results["strongest bottleneck"][snr_db] = float(np.mean(bottleneck_rates))

        # ==============================================
        # 4) Equal-power heuristic (make it problem-aware too)
        # ==============================================
        rates_equal_power, _ = compute_equal_power_bound(
            dataset,
            sigma_noise=sigma,
            problem=problem,
        )
        results["equal power"][snr_db] = float(np.mean(rates_equal_power))

        # ==============================================
        # 5) Greedy max-link benchmark
        # ==============================================
        rates_greedy, _ = compute_greedy_power_rate(
            dataset,
            sigma_noise=sigma,
            problem=problem,
        )
        results["greedy maxlink"][snr_db] = float(np.mean(rates_greedy))

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



