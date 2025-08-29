import numpy as np
import torch
from utils.CentralizedUtils import evaluate_centralized_adam, compute_strongest_bottleneck_rate, compute_equal_power_bound
from utils.PathUtils import find_all_paths, paths_to_tensor
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
    vals = []
    for d in dataset:
        H = d.links_matrix  # [B,n,n], complex
        A = d.adj_matrix  # [n,n], 0/1

        mask = A.bool()
        E = int(mask.sum())
        if E == 0:
            continue  # or append 0.0, your call

        # Mask across edges for each band -> [B, E]
        Hr = H.real[:, mask]
        Hi = H.imag[:, mask]

        # Var per band over edges, then sum Re+Im
        var_r = Hr.var(dim=1, unbiased=False)
        var_i = Hi.var(dim=1, unbiased=False)
        per_band_var = var_r + var_i  # [B]

        vals.append(per_band_var.mean())
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
            # swa_rates = []
            for d in dataset:
                d.sigma = torch.tensor(sigma, device=device)
                d = d.to(device)

                paths = find_all_paths(d.adj_matrix, d.tx, d.rx)
                paths = paths_to_tensor(paths, device)
                gnn_rates, _ = _compute_rates_per_layer(model, d, paths)
                rate = max(gnn_rates)
                rates.append(rate.item())


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



