import torch
import numpy as np
import mat73

from datasets.GraphDataSet import GraphNetDataset

#=======================================================================================================================
# Graph and Dataset Generation
#=======================================================================================================================
def generate_valid_adj(n, directed=False, device='cpu', min_connections=1):
    """
    Generates a random adjacency matrix with no isolated nodes.

    Args:
        n (int): Number of nodes.
        directed (bool): Whether the graph is directed.
        device (str): Device for the tensor.
        min_connections (int): Minimum connections to add if node is isolated.

    Returns:
        torch.Tensor: [n, n] adjacency matrix.
    """
    adj_matrix = torch.randint(0, 2, (n, n), device=device)

    if not directed:
        adj_matrix = torch.triu(adj_matrix, diagonal=1)
        adj_matrix = adj_matrix + adj_matrix.T

    adj_matrix.fill_diagonal_(0)  # No self-loops

    # Fix isolated nodes
    for i in range(n):
        if adj_matrix[i].sum() == 0 and adj_matrix[:, i].sum() == 0:
            # Choose random target nodes, excluding self
            candidates = [j for j in range(n) if j != i]
            targets = torch.tensor(
                np.random.choice(candidates, size=min_connections, replace=False),
                device=device
            )
            adj_matrix[i, targets] = 1
            if not directed:
                adj_matrix[targets, i] = 1

    return adj_matrix

def generate_graph_and_channel_matrices(n, B, directed=False, mu: float = 0.0, sigma=1.0, seed=1, device=None):
    """
    Generates an adjacency matrix and a wireless channel matrix for a graph with n vertices.

    Args:
        n (int): Number of nodes in the graph.
        B (int): Number of frequency bands.
        directed (bool): Whether the graph is directed. Defaults to False.
        mu (float): Mean for channel matrix normalization.
        sigma (float): Standard deviation for channel matrix normalization.
        seed (int): Random seed for reproducibility.
        device (torch.device): Device to place the tensors on. If None, uses CUDA if available.

    Returns:
        adj_matrix (torch.Tensor): [n, n] adjacency matrix.
        channel_matrix_arr (torch.Tensor): [B, n, n] stacked channel matrices.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adj_matrix = generate_valid_adj(n, directed=directed, device=device, min_connections=1)
    channel_matrix_arr = []

    for b in range(B):
        torch.manual_seed(seed + b)
        real_part = torch.randn(size=(n, n), device=device)
        torch.manual_seed(seed + 1 + b)
        imag_part = torch.randn(size=(n, n), device=device)
        channel_matrix = torch.complex(real_part, imag_part)

        if not directed:
            channel_matrix = (torch.triu(channel_matrix, diagonal=1) + torch.triu(channel_matrix, diagonal=1).T) / 2

        mask = adj_matrix.bool()
        channel_matrix *= mask

        non_zero_vals = channel_matrix[mask]
        if non_zero_vals.numel() > 1:
            mean = torch.mean(non_zero_vals)
            std = torch.std(non_zero_vals)
            normalized_vals = (non_zero_vals - mean) / std
            scaled_vals = normalized_vals * sigma + mu
            channel_matrix[mask] = scaled_vals

        channel_matrix_arr.append(channel_matrix)

    return adj_matrix, torch.stack(channel_matrix_arr)


def generate_graph_data(n_list, tx_list, rx_list, sigma_list, B, seed=1000, channel_path=None, channel_key='channels', device='cpu'):
    """
    Generates a dataset of graphs and associated channel matrices.

    Args:
        n_list (List[int]): List of node counts for each sample.
        tx_list (List[int]): List of transmitter node indices for each sample.
        rx_list (List[int]): List of receiver node indices for each sample.
        sigma_list (List[float]): Noise variances per graph.
        B (int): Number of frequency bands.
        seed (int): Starter seed for data generation
        channel_path (str): Optional path to .mat file containing precomputed channels.
        channel_key (str): Key used to load channels from .mat.
        device : Target device (cpu or cuda).

    Returns:
        GraphNetDataset: A dataset of graph data objects.
    """
    adj_list, links_list = [], []
    num_samples = len(n_list)

    if channel_path:
        mat = mat73.loadmat(channel_path)
        raw_channels = mat['H_all']
        print(f"Loaded channel matrix from {channel_path} with shape {raw_channels.shape}")
        if raw_channels.shape[0] != num_samples:
            raise ValueError(f"Mismatch: {num_samples} graphs expected, but .mat contains {raw_channels.shape[0]} samples.")

        for i in range(num_samples):
            n = n_list[i]
            links = torch.as_tensor(raw_channels[i], dtype=torch.cfloat, device=device)
            adj = (links.abs().sum(dim=0) > 0).float()
            adj.fill_diagonal_(0)
            adj_list.append(adj)
            links_list.append(links)
    else:
        for i in range(num_samples):
            n = n_list[i]
            adj, links = generate_graph_and_channel_matrices(n, B, seed=i+seed, device=device)
            adj_list.append(adj)
            links_list.append(links)

    dataset = GraphNetDataset(adj_list, links_list, tx_list, rx_list, sigma_list, B, device=device)
    return dataset
