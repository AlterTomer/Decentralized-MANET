import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PathUtils import find_all_paths, paths_to_tensor
from TensorUtils import normalize_power, create_normalized_tensor, init_equal_power
from MetricUtils import calc_sum_rate

#=======================================================================================================================
# Centralized Optimization
#=======================================================================================================================
def classic_opt(num_iterations, optimizer, adj_mat, links_mat, p_arr, sigma_noise, paths, B):
    """
    Runs centralized ADAM optimization for power allocation.

    Args:
        num_iterations (int): Number of optimization steps.
        optimizer (torch.optim.Optimizer): ADAM optimizer.
        adj_mat (torch.Tensor): [n, n] adjacency matrix.
        links_mat (torch.Tensor): [B, n, n] channel matrices.
        p_arr (torch.nn.Parameter): Power allocation variable.
        sigma_noise (float): Noise power.
        paths (Tensor): Tensor of all paths.
        B (int): Number of frequency bands.

    Returns:
        torch.Tensor: Optimized power allocation tensor.
    """
    a = 0

    with torch.no_grad():
        p_arr.copy_(normalize_power(p_arr, adj_mat))
    for i in range(num_iterations):
        optimizer.zero_grad()
        loss = -calc_sum_rate(links_mat, p_arr, sigma_noise, paths, B)
        try:
            loss.backward()
            optimizer.step()
        except RuntimeError:
            a = 1
            p0 = p_arr
            p0 = normalize_power(p0, adj_mat)
        with torch.no_grad():
            p_arr.copy_(normalize_power(p_arr, adj_mat))
        # if i == num_iterations - 1:
        #     print(f"Iteration {i + 1}: Loss = {loss.item()}")
    return p_arr if a == 0 else p0


def evaluate_centralized_adam(loader, B, noise_std=None, lr=0.1, num_iterations=30):
    """
    Evaluates centralized ADAM optimization over a dataset.

    Args:
        loader (DataLoader): Graph dataset loader.
        B (int): Number of frequency bands.
        noise_std (float or array): Noise std.
        lr (float): Learning rate for ADAM.
        num_iterations (int): Number of optimization steps.

    Returns:
        Tuple[List[float], List[torch.Tensor]]: List of achieved rates and power allocations.
    """
    rate_results = []
    p_opt_results = []
    sigma_arr = None
    if isinstance(noise_std, list):
        sigma_arr = noise_std

    for k, data in enumerate(loader):
        adj = data.adj_matrix.to(data.links_matrix.device)
        links = data.links_matrix
        if sigma_arr:
            sigma = sigma_arr[k]
        elif isinstance(noise_std, float):
            sigma = noise_std
        else:
            sigma = data.sigma
        paths = find_all_paths(adj.cpu(), data.tx, data.rx)
        paths = paths_to_tensor(paths, data.links_matrix.device)
        p_arr = torch.stack([create_normalized_tensor(adj.shape[0], adj.shape[1], mask=adj, device=links.device) for _ in range(B)])
        p_arr = nn.Parameter(p_arr, requires_grad=True)
        optimizer = optim.AdamW([p_arr], lr=lr)
        p_opt = classic_opt(num_iterations, optimizer, adj, links, p_arr, sigma, paths, B)
        p_opt_results.append(p_opt)
        rate = calc_sum_rate(links, p_opt, sigma, paths, B).item()
        rate_results.append(rate)

    return rate_results, p_opt_results


def evaluate_centralized_adam_single(data, B, lr=0.1, num_iterations=100):
    """
    Evaluate the centralized ADAM optimization strategy for a set of graphs.

    This function performs ADAM-based power allocation optimization for each
    graph in the loader, aiming to maximize the sum rate. For each graph:
    - It initializes a learnable power allocation tensor `p_arr` of shape [B, n, n].
    - Optimizes it over `num_iterations` steps using the given learning rate.
    - Stores the final optimized power allocation and the corresponding rate.

    Args:
        loader (DataLoader or Iterable): A loader or iterable yielding graph data objects.
            Each data object is expected to contain:
                - data.adj_matrix: [n, n] binary adjacency matrix
                - data.links_matrix: [B, n, n] complex link gains
                - data.sigma: scalar noise standard deviation
                - data.tx: transmitter node index (int)
                - data.rx: receiver node index (int)
        B (int): Number of frequency bands
        lr (float): Learning rate for the ADAM optimizer (default 0.1)
        num_iterations (int): Number of optimization steps per graph (default 100)

    Returns:
        rate_results (List[float]): Achieved sum rate for each graph after optimization
        p_opt_results (List[Tensor]): Optimized power allocation tensors of shape [B, n, n] for each graph
    """

    adj = data.adj_matrix                # [n, n] binary adjacency matrix
    links = data.links_matrix            # [B, n, n] complex channel gains
    sigma = data.sigma                   # noise std
    tx = data.tx                         # transmitter index
    rx = data.rx                         # receiver index
    paths = find_all_paths(adj, tx, rx)  # list of paths from Tx to Rx
    paths = paths_to_tensor(paths, data.links_matrix.device)

    # Initialize power allocation tensor [B, n, n] with masked entries
    p_arr = torch.stack([
        create_normalized_tensor(adj.shape[0], adj.shape[1], mask=adj) for _ in range(B)
    ])
    p_arr = nn.Parameter(p_arr, requires_grad=True)

    # ADAM optimizer on p_arr
    optimizer = optim.AdamW([p_arr], lr=lr)

    # Optimize with respect to sum rate (maximize via negative loss)
    p_opt = classic_opt(num_iterations, optimizer, adj, links, p_arr, sigma, paths, B)

    # Evaluate and store the achieved rate
    rate = calc_sum_rate(links, p_opt, sigma, paths, B)
    # print(f'lower bound = {compute_lower_bound_single(data, sigma)}')

    return rate, p_opt



def compute_lower_bound_rate(dataset, sigma_noise=False):
    """
    Compute the lower bound rate for each graph in the dataset.
    The lower bound is determined by the strongest weakest link among all paths
    and all frequency bands between Tx and Rx.

    Args:
        dataset: List of graph data objects, each with attributes:
            - adj_matrix: adjacency matrix [n, n]
            - links_matrix: complex channel links [B, n, n]
            - sigma: noise std
            - B: number of frequency bands
            - tx: transmitter node index
            - rx: receiver node index

    Returns:
        np.ndarray: lower bound rates for each graph
    """
    lower_bounds = []
    p_min = []
    sigma = sigma_noise

    for data in dataset:
        adj = data.adj_matrix
        links = data.links_matrix  # shape: [B, n, n]
        if not sigma_noise:
            sigma = data.sigma
        B = data.B
        tx = data.tx
        rx = data.rx
        p = torch.zeros_like(links)
        paths = find_all_paths(adj, tx, rx)
        if not paths:
            lower_bounds.append(0.0)
            continue

        h_data = []
        h_idx = []
        for b in range(B):
            band_min_links = []
            band_min_idx = []
            for path in paths:
                b_links = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
                rows, cols = zip(*b_links)
                h_path_min = torch.min(torch.abs(links[b][rows, cols]) ** 2).item()
                band_mask = torch.abs(links[b][rows, cols]) ** 2 == h_path_min
                idx = list(b_links[band_mask.nonzero(as_tuple=True)[0]])
                idx.insert(0, b)
                band_min_idx.append(idx)
                band_min_links.append(h_path_min)

            h_band_min = min(band_min_links)
            h_data.append(h_band_min)
            h_idx.append(band_min_idx[band_min_links.index(h_band_min)])


        snr = max(h_data) / (sigma ** 2)
        rate = np.log2(1 + snr)
        lower_bounds.append(rate)
        b, i, j = h_idx[h_data.index(max(h_data))]
        p[b, i, j] = 1
        p_min.append(p)

    return np.array(lower_bounds), p_min


def compute_lower_bound_rate_single(sigma, adj, links, B, tx, rx):
    """
    Compute the lower bound rate for each graph in the dataset.
    The lower bound is determined by the strongest weakest link among all paths
    and all frequency bands between Tx and Rx.

    Args:
            - adj: adjacency matrix [n, n]
            - links: complex channel links [B, n, n]
            - sigma: noise std
            - B: number of frequency bands
            - tx: transmitter node index
            - rx: receiver node index

    Returns:
        np.ndarray: lower bound rates for each graph
    """
    n, _ = adj.shape
    p = torch.zeros((B, n, n))
    paths = find_all_paths(adj, tx, rx)
    if len(paths) == 0:
        p = torch.stack([create_normalized_tensor(adj.shape[0], adj.shape[1], mask=adj, device=links.device) for _ in range(B)])
        p = normalize_power(p, adj)
        rate = 0
        print('no paths found')
        return rate, p

    h_data = []
    h_idx = []
    for b in range(B):
        band_min_links = []
        band_min_idx = []
        for path in paths:
            b_links = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
            rows, cols = zip(*b_links)
            h_path_min = torch.min(torch.abs(links[b][rows, cols]) ** 2).item()
            band_mask = torch.abs(links[b][rows, cols]) ** 2 == h_path_min
            idx = list(b_links[band_mask.nonzero(as_tuple=True)[0]])
            idx.insert(0, b)
            band_min_idx.append(idx)
            band_min_links.append(h_path_min)

        h_band_min = min(band_min_links)
        h_data.append(h_band_min)
        h_idx.append(band_min_idx[band_min_links.index(h_band_min)])


    snr = max(h_data) / (sigma ** 2)
    rate = np.log2(1 + snr)
    b, i, j = h_idx[h_data.index(max(h_data))]
    p[b, i, j] = 1

    return rate, p

def compute_equal_power_bound(dataset, sigma_noise=False):
    """
    Compute the rate using equal power variables across all links and frequency bands.

    Args:
        dataset: List of graph data objects, each with attributes:
            - adj_matrix: adjacency matrix [n, n]
            - links_matrix: complex channel links [B, n, n]
            - sigma: noise std
            - B: number of frequency bands
            - tx: transmitter node index
            - rx: receiver node index

    Returns:
        np.ndarray: lower bound rates for each graph
    """
    rate_bounds = []
    p_arr = []
    sigma = sigma_noise

    for data in dataset:
        device = data.x.device
        adj = data.adj_matrix
        n = adj.shape[0]
        links = data.links_matrix  # shape: [B, n, n]
        if not sigma_noise:
            sigma = data.sigma
        B = data.B
        tx = data.tx
        rx = data.rx
        p = torch.zeros(B, n, n)
        p = init_equal_power(p, adj)
        p = normalize_power(p, adj)
        paths = find_all_paths(adj, tx, rx)
        if not paths:
            rate_bounds.append(0.0)
            continue
        paths = paths_to_tensor(paths, device)

        rate = calc_sum_rate(links, p, torch.tensor(sigma, device=device), paths, B)
        rate_bounds.append(rate)

    return np.array(rate_bounds), p_arr