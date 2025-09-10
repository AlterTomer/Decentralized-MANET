import numpy as np
import torch
from utils.CentralizedUtils import evaluate_centralized_adam, compute_strongest_bottleneck_rate, compute_equal_power_bound
from utils.PathUtils import find_all_paths, paths_to_tensor
from utils.MetricUtils import calc_sum_rate
from utils.DataUtils import mean_var_over_dataset
from models.GraphNetAux import _compute_rates_per_layer


# -------------------------------
# Helpers
# -------------------------------
def evaluate_across_snr(dataset, model, B, snr_db_list):
    """
    Sequential evaluation across a list of SNR values.

    Args:
        dataset: Your dataset (already on CPU or GPU as needed).
        model: Trained GNN model.
        B: Number of bands.
        snr_db_list: List of SNR values in dB.

    Returns:
        dict: { "gnn": {snr_db: mean_rate}, "adam": {...}, "lower": {...} }
    """
    device = next(model.parameters()).device
    results = {"gnn": {} ,"centralized": {}, "strongest bottleneck": {}, "equal power": {}}

    # --- compute mean channel variance for noise scaling ---
    vals = mean_var_over_dataset(dataset)
    out = torch.stack(vals).mean()
    mean_channel_var = out.item()

    for snr_db in snr_db_list:
        snr = 10.0 ** (snr_db / 10.0)
        sigma2 = mean_channel_var / snr
        sigma = sigma2 ** 0.5
        print(f'sigma: {sigma}')

        # GNN mean rate
        model.eval()
        with torch.no_grad():
            rates = []
            for d in dataset:
                d.sigma = torch.tensor(sigma, device=device)
                d = d.to(device)

                paths = find_all_paths(d.adj_matrix, d.tx, d.rx)
                paths = paths_to_tensor(paths, device)
                gnn_rates, _ = _compute_rates_per_layer(model, d, paths)
                rate = torch.stack(gnn_rates).max().item()
                rates.append(rate)


            results["gnn"][snr_db] = float(np.mean(rates))
        # Centralized optimization
        adam_rates = evaluate_centralized_adam(dataset, B, noise_std=sigma, num_iterations=50)[0]
        results["centralized"][snr_db] = float(np.mean(adam_rates))

        # Strongest bottleneck
        bottleneck_rates, _ = compute_strongest_bottleneck_rate(dataset ,sigma_noise=sigma)
        results["strongest bottleneck"][snr_db] = float(np.mean(bottleneck_rates))

        rates_equal_power, _ = compute_equal_power_bound(dataset, sigma_noise=sigma)
        results["equal power"][snr_db] = float(np.mean(rates_equal_power))
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
    vals = mean_var_over_dataset(dataset)
    out = torch.stack(vals).mean()
    mean_channel_var = out.item()

    for snr_db in snr_db_list:
        snr = 10.0 ** (snr_db / 10.0)
        sigma2 = mean_channel_var / snr
        sigma = sigma2 ** 0.5
        print(f'sigma: {sigma}')

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