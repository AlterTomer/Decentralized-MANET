import numpy as np
import torch
from CentralizedUtils import evaluate_centralized_adam, compute_lower_bound_rate
from PathUtils import find_all_paths, paths_to_tensor
from GraphNetAux import _compute_rates_per_layer


# -------------------------------
# Helpers
# -------------------------------
def evaluate_across_snr(dataset, model, swa_model, B, snr_db_list):
    """
    Sequential evaluation across a list of SNR values.

    Args:
        dataset: Your dataset (already on CPU or GPU as needed).
        model: Trained GNN model.
        swa_model: SWA model (optional; pass None if not used).
        B: Number of bands.
        snr_db_list: List of SNR values in dB.

    Returns:
        dict: { "gnn": {snr_db: mean_rate}, "adam": {...}, "lower": {...} }
    """
    device = next(model.parameters()).device
    results = {"gnn": {} ,"adam": {}, "lower": {}}

    # --- compute mean channel variance for noise scaling ---
    # channel_vars = []
    # for d in dataset:
    #     H = d.links_matrix.detach().cpu().numpy()
    #     channel_vars.append(np.var(H.real) + np.var(H.imag))
    # mean_channel_var = float(np.mean(channel_vars))

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

                # out = model(d)[-1]  # last layer output
                # out = normalize_power(out, d.adj_matrix)
                paths = find_all_paths(d.adj_matrix, d.tx, d.rx)
                paths = paths_to_tensor(paths, device)
                # rate = calc_sum_rate(d.links_matrix, out, torch.tensor(sigma, device=device), paths, B)

                gnn_rates, _ = _compute_rates_per_layer(model, d, paths)
                rate = max(gnn_rates)
                rates.append(rate.item())

                # swa_out = swa_model(d)[-1]
                # swa_out = normalize_power(swa_out, d.adj_matrix)
                # swa_rate = calc_sum_rate(d.links_matrix, swa_out, torch.tensor(sigma, device=device), paths, B)
                # swa_rates.append(float(swa_rate))

            results["gnn"][snr_db] = float(np.mean(rates))
            # results["swa"][snr_db] = float(np.mean(swa_rates))

        # ADAM centralized optimization
        adam_rates = evaluate_centralized_adam(dataset, B, noise_std=sigma, num_iterations=50)[0]
        results["adam"][snr_db] = float(np.mean(adam_rates))

        # Lower bound
        lower_rates, _ = compute_lower_bound_rate(dataset ,sigma_noise=sigma)
        results["lower"][snr_db] = float(np.mean(lower_rates))

    return results



