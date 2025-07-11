import math
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np
from GraphDataSet import GraphNetDataset
from scipy.io import loadmat

def generate_graph_and_channel_matrices(n: int, B: int, directed: bool = False, mu: float = 0.0, sigma: float = 1.0, seed: int = 1):
    """Generates an adjacency matrix and a wireless channel matrix for a graph with n vertices.

    Args:
        n (int): Number of vertices.
        B (int): Number of frequency bands
        directed (bool): If True, generates a directed graph. If False, generates an undirected graph.
        mu (float): Mean of the Gaussian distribution for channel gains.
        sigma (float): Standard deviation of the Gaussian distribution for channel gains.
        seed (int): Seed for generating random channel samples

    Returns:
        tuple: (adjacency_matrix, channel_matrix), both as torch.Tensor.
    """
    # Generate adjacency matrix
    adj_matrix = torch.randint(0, 2, (n, n))  # Random binary adjacency matrix

    if not directed:
        adj_matrix = torch.triu(adj_matrix, diagonal=1)
        adj_matrix = adj_matrix + adj_matrix.T

    adj_matrix.fill_diagonal_(0)  # No self-loops

    channel_matrix_arr = []

    for b in range(B):
        torch.manual_seed(seed + b)
        real_part = torch.randn(size=(n, n))
        torch.manual_seed(seed + 1 + b)
        imag_part = torch.randn(size=(n, n))
        channel_matrix = torch.complex(real_part, imag_part)

        if not directed:
            channel_matrix = (torch.triu(channel_matrix, diagonal=1) + torch.triu(channel_matrix, diagonal=1).T) / 2

        # Apply adjacency mask
        mask = adj_matrix.bool()
        channel_matrix *= mask  # Zero out nonexistent links

        # Normalize only non-zero entries to N(0,1)
        non_zero_vals = channel_matrix[mask]
        if non_zero_vals.numel() > 1:
            mean = torch.mean(non_zero_vals)
            std = torch.std(non_zero_vals)
            normalized_vals = (non_zero_vals - mean) / std
            # Rescale to N(mu, sigma^2)
            scaled_vals = normalized_vals * sigma + mu
            channel_matrix[mask] = scaled_vals  # Update only masked entries

        channel_matrix_arr.append(channel_matrix)

    return adj_matrix, torch.stack(channel_matrix_arr)



def find_all_paths(adj_matrix, start, destination, path=None, visited=None):
    """
     Finds all paths from a starting node to a destination node in an undirected graph
     represented by a PyTorch tensor adjacency matrix.

     Parameters:
     -----------
     adj_matrix : torch.Tensor
         A square adjacency matrix (shape: [N, N]) where N is the number of nodes.
         - adj_matrix[i, j] > 0 indicates an edge between nodes i and j.
         - adj_matrix[i, j] = 0 indicates no edge.

     start : int
         The index of the starting node.

     destination : int
         The index of the destination node.

     path : list, optional
         A list storing the current traversal path (default is None, initialized as [start]).

     visited : set, optional
         A set to keep track of visited nodes to avoid cycles (default is None, initialized as empty set).

     Returns:
     --------
     paths : list of lists
         A list containing all possible paths from `start` to `destination`.
         Each path is represented as a list of node indices.
         """
    if path is None:
        path = [start]
    if visited is None:
        visited = set()

    if start == destination:
        return [path]  # Found a valid path

    visited.add(start)  # Mark node as visited
    paths = []

    for neighbor in range(adj_matrix.shape[0]):  # Iterate over all nodes
        if adj_matrix[start, neighbor].item() > 0 and neighbor not in visited:  # Convert tensor to scalar
            new_paths = find_all_paths(adj_matrix, neighbor, destination, path + [neighbor], visited.copy())
            paths.extend(new_paths)

    return paths


def link_rate(h, p, sigma):
    """
    This function calculates the max rate on a given link h
    :param h: Link (complex float)
    :param p: Power allocation (float)
    :param sigma: Noise std (float)
    :return: Link's rate (torch float)
    """
    snr = ((abs(h) * p) ** 2) / (sigma ** 2)
    return torch.log2(1 + snr)


def calc_sum_rate(h_arr, p_arr, sigma, paths, B):
    """
    Calculates the sum rate using either node-based or link-based power allocations.

    :param h_arr: B x n x n complex tensor of channel links
    :param p_arr: B x n x n or B x n tensor of power allocations
    :param sigma: Noise std per band (float or 1D tensor of shape [B])
    :param paths: List of paths (each a list of node indices)
    :param B: Number of frequency bands
    :return: Total sum rate (torch scalar tensor)
    """
    sum_rate = torch.tensor(0.0, dtype=torch.float32)

    for b in range(B):
        max_path_rate = torch.tensor(float('-inf'), dtype=torch.float32)
        band_links = h_arr[b]
        band_power = p_arr[b]

        for path in paths:
            edge_indices = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
            links = [band_links[i, j] for i, j in edge_indices]

            # Detect if power is node-based or full matrix
            if band_power.dim() == 1 or (
                band_power.dim() == 2 and torch.allclose(band_power, torch.diag(torch.diagonal(band_power)))
            ):
                power_allocation = [band_power[i] for i, _ in edge_indices]
            else:
                power_allocation = [band_power[i, j] for i, j in edge_indices]

            # Compute rate of weakest link
            min_link_rate = torch.tensor(float('inf'), dtype=torch.float32)
            for h, p in zip(links, power_allocation):
                if abs(h) == 0 or torch.norm(p) == 0:
                    continue
                s = sigma.item() if isinstance(sigma, torch.Tensor) else sigma
                r = link_rate(h, p.max() if p.dim() > 0 else p, s)
                r = r.squeeze()  # ensure scalar
                min_link_rate = torch.minimum(min_link_rate, r)

            max_path_rate = torch.maximum(max_path_rate, min_link_rate)

        sum_rate += max_path_rate.squeeze()

    return sum_rate / B

def create_normalized_tensor(m, n, mask=None):
    """
    This function generates a matrix where each row has a unit norm
    :param m: Number of rows (int)
    :param n: Number of columns (int)
    :param mask: If tensor, multiply the generated tensor by the mask (tensor)
    :return: Normalized mXn tensor
    """
    A = torch.rand(m, n)
    if isinstance(mask, torch.Tensor):
        A *= mask
    norm_A = torch.norm(A, dim=1, keepdim=True)
    normalized_A = A / norm_A
    return normalized_A


def find_best_paths_per_band(h_arr, p_arr, sigma, paths, B):
    """
    Find the best path in each frequency band based on optimized power allocation.

    :param h_arr: Wireless channel matrix (BxNxN tensor)
    :param p_arr: Optimized power allocation matrix (BxNxN tensor)
    :param sigma: Noise std (float)
    :param paths: List of all possible paths between TX and RX
    :param B: Number of frequency bands (int)
    :return: Dictionary {band_index: best_path}
    """
    best_paths = {}

    for b in range(B):
        max_rate = float('-inf')
        best_path = None

        # Extract the power and channel gains for this frequency band
        band_power = p_arr[b, :, :]
        band_links = h_arr[b, :, :]
        if b == 2:
            a = 0
        for path in paths:
            edge_indices = [(path[i], path[i+1]) for i in range(len(path) - 1)]  # Edges in the path

            # Get the channel gains and power allocations for the current path
            links = torch.tensor([band_links[row, col] for row, col in edge_indices])
            power_allocations = torch.tensor([band_power[row, col] for row, col in edge_indices])

            if len(links) == 0 or len(power_allocations) == 0:
                continue  # Skip invalid paths

            # Compute rates for all links in this path
            rates = torch.tensor([link_rate(h, p, sigma) for h, p in zip(links, power_allocations)])

            # The path rate is determined by the weakest link (min rate along the path)
            path_rate = torch.min(rates)

            # Update the best path if this one has a higher rate
            if path_rate > max_rate:
                max_rate = path_rate
                best_path = path

        best_paths[b] = best_path

    return best_paths


def visualize_best_paths(adj_matrix, best_paths, links_mat, p_arr, sigma, title="Best Paths in MANET"):
    """
    Visualizes the MANET graph and highlights the best paths per frequency band with a legend.

    :param adj_matrix: Adjacency matrix (NxN tensor)
    :param best_paths: Dictionary {band_index: best_path} with paths as lists of node indices
    :param links_mat: Channel matrix (BxNxN tensor) representing link strengths
    :param p_arr: Power allocation matrix (BxNxN tensor) optimized
    :param sigma: Noise std
    :param title: Title for the plot
    """
    n = adj_matrix.shape[0]  # Number of nodes
    B = p_arr.shape[0]  # Number of frequency bands
    G = nx.Graph()

    # Add nodes
    for i in range(n):
        G.add_node(i)

    # Add edges (from adjacency matrix)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):  # Avoid double counting edges
            if adj_matrix[i, j] > 0:
                G.add_edge(i, j)
                edges.append((i, j))

    pos = nx.spring_layout(G, seed=42)  # Compute node positions
    plt.figure(figsize=(10, 8))

    # Draw base graph with light gray edges
    nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="gray", width=0.5, node_size=500, font_size=10)

    # Colors for different frequency bands
    colors = [mcolors.to_hex(np.random.rand(3,)) for _ in range(B)]

    # Legend entries
    legend_patches = []

    # Draw best paths in different colors
    for b, path in best_paths.items():
        if path is None:
            continue  # Skip if no valid path

        path_edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color=colors[b % len(colors)], width=2.5)

        # Compute the rate for the best path
        path_links = [links_mat[b, path[i], path[i + 1]] for i in range(len(path) - 1)]
        path_power = [p_arr[b, path[i], path[i + 1]] for i in range(len(path) - 1)]

        rates = torch.tensor([link_rate(h, p, sigma) for h, p in zip(path_links, path_power)])
        path_rate = torch.min(rates).item()  # Min-rate on the path


        # Add entry to the legend
        legend_patches.append(mpatches.Patch(color=colors[b % len(colors)], label=f"Band {b}: {path_rate:.2f} bps/Hz, Path: {path}"))

    # Add legend
    plt.legend(handles=legend_patches, title="Best Path Rates", loc="upper right", fontsize=10)

    plt.title(title)
    plt.show()



def find_band_min_rate(h_arr, paths, sigma, B):
    """
    Find the naive lower limit on each band's rate.

    :param h_arr: Wireless channel matrix (BxNxN tensor)
    :param sigma: Noise std (float)
    :param paths: List of all possible paths between TX and RX
    :param B: Number of frequency bands (int)
    :return: Lower bounds of each band's rate (List)
    """
    bands_bounds = []
    for b in range(B):

        # Extract the power and channel gains for this frequency band
        band_links = h_arr[b, :, :]
        snr_path = math.inf
        for path in paths:
            edge_indices = [(path[i], path[i+1]) for i in range(len(path) - 1)]  # Edges in the path

            # Get the channel gains and power allocations for the current path
            links = torch.tensor([band_links[row, col] for row, col in edge_indices])
            h = torch.min(torch.abs(links ** 2))
            snr = h / (sigma ** 2)
            if snr < snr_path:
                snr_path = snr

        r = torch.log2(1 + snr_path)
        bands_bounds.append(r)

    return bands_bounds


def classic_opt(num_iterations, optimizer, links_mat, p_arr, sigma_noise, paths, B):
    """

    Args:
        num_iterations: Number of gradient-based optimization algorithm iterations (int)
        optimizer: Optimizer object
        links_mat: Channel links tensor (torch tensor)
        p_arr: Power allocation tensor for all users (torch tensor)
        sigma_noise: Noise std (float)
        paths: All paths between TX and RX (list)
        B: Number of frequency bands (int)

    Returns: Optimized power allocation tensor (torch tensor)

    """
    for i in range(num_iterations):
        optimizer.zero_grad()  # Reset gradients
        loss = -calc_sum_rate(links_mat, p_arr, sigma_noise, paths, B)  # Compute loss
        loss.backward()  # Compute gradients
        optimizer.step()  # Update p

        # Apply constraints
        with torch.no_grad():
            # Compute the norm of each row along dimension 2 (each row in an nxn matrix)
            row_norms = torch.norm(p_arr, dim=2, keepdim=True)  # Shape: (B, n, 1)

            # Normalize rows that exceed a norm of 1
            p_arr.data = torch.where(row_norms > 1, p_arr / row_norms, p_arr)

            # Enforce positivity
            p_arr.data = torch.clamp(p_arr, min=0)

        # Print progress every 10 iterations
        # if i % 10 == 0:
        #     print(f"Iteration {i}: Loss = {loss.item()}")

    return p_arr



def expand_power_allocation(power_allocation: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
    """
    Expand the node-level power allocation (n, B) to edge-level (B, n, n),
    based on the adjacency matrix. Ensures gradient flow is preserved.

    Args:
        power_allocation (torch.Tensor): Tensor of shape (n, B)
            - Each row corresponds to a node's power vector over B frequency bands.
        adjacency (torch.Tensor): Tensor of shape (n, n)
            - Binary adjacency matrix (0 or 1), where adj[i, j] = 1 means a link exists.

    Returns:
        torch.Tensor: Tensor of shape (B, n, n)
            - power[b, i, j] = power_allocation[i, b] if i→j exists, else 0.
    """

    n, B = power_allocation.shape

    # Step 1: Expand (n, B) → (n, B, n)
    # Each node i's power vector (B,) is broadcast to all its outgoing links (j)
    power_expanded = power_allocation.unsqueeze(2).expand(n, B, n)  # shape: [n, B, n]

    # Step 2: Transpose to match expected shape (B, n, n)
    power_expanded = power_expanded.transpose(0, 1)  # shape: [B, n, n]

    # Step 3: Create a float-valued adjacency mask (shape [B, n, n])
    # This ensures that the output retains gradients (float vs. bool)
    adj_mask = adjacency.float().unsqueeze(0).expand(B, n, n)

    # Step 4: Apply the mask to zero out non-connected links
    output = power_expanded * adj_mask

    return output


def evaluate_centralized_adam(loader, B, lr=0.1, num_iterations=100):
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
    rate_results = []
    p_opt_results = []

    for data in loader:
        adj = data.adj_matrix                # [n, n] binary adjacency matrix
        links = data.links_matrix            # [B, n, n] complex channel gains
        sigma = data.sigma                   # noise std
        tx = data.tx                         # transmitter index
        rx = data.rx                         # receiver index
        paths = find_all_paths(adj, tx, rx)  # list of paths from Tx to Rx

        # Initialize power allocation tensor [B, n, n] with masked entries
        p_arr = torch.stack([
            create_normalized_tensor(adj.shape[0], adj.shape[1], mask=adj) for _ in range(B)
        ])
        p_arr = nn.Parameter(p_arr, requires_grad=True)

        # ADAM optimizer on p_arr
        optimizer = optim.Adam([p_arr], lr=lr)

        # Optimize with respect to sum rate (maximize via negative loss)
        p_opt = classic_opt(num_iterations, optimizer, links, p_arr, sigma, paths, B)
        p_opt_results.append(p_opt)

        # Evaluate and store the achieved rate
        rate = calc_sum_rate(links, p_opt, sigma, paths, B).item()
        rate_results.append(rate)

    return rate_results, p_opt_results



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
    sigma = sigma_noise

    for data in dataset:
        adj = data.adj_matrix
        links = data.links_matrix  # shape: [B, n, n]
        if not sigma_noise:
            sigma = data.sigma
        B = data.B
        tx = data.tx
        rx = data.rx

        paths = find_all_paths(adj, tx, rx)
        if not paths:
            lower_bounds.append(0.0)
            continue

        h_data = []
        for b in range(B):
            band_min_links = []

            for path in paths:
                b_links = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
                rows, cols = zip(*b_links)
                h_path_min = torch.min(torch.abs(links[b][rows, cols]) ** 2).item()
                band_min_links.append(h_path_min)

            h_band_min = min(band_min_links)
            h_data.append(h_band_min)


        snr = max(h_data) / (sigma ** 2)
        rate = np.log2(1 + snr)
        lower_bounds.append(rate)

    return np.array(lower_bounds)



def generate_graph_data(n_list, tx_list, rx_list, sigma_list, B, channel_path=None, channel_key='channels'):
    """
    Dataset generator for GraphNet or ChainedGNN model.

    Args:
        n_list: List of number of devices per network (length = num_samples)
        tx_list: List of transmitter indices per graph
        rx_list: List of receiver indices per graph
        sigma_list: List of noise std per graph
        B: Number of frequency bands
        channel_path (str, optional): Path to a .mat file with pre-generated channels.
        channel_key (str): Name of the variable in the .mat file holding the channel data

    Returns:
        dataset: a GraphNetDataset with graphs and corresponding channel matrices
    """
    adj_list, links_list = [], []
    num_samples = len(n_list)

    if channel_path:
        # Load precomputed channels from .mat
        mat = loadmat(channel_path)
        raw_channels = mat[channel_key]  # expected shape: [num_samples, B, n, n] or similar
        print(f"Loaded channel matrix from {channel_path} with shape {raw_channels.shape}")

        # Make sure axes are in correct order: (num_samples, B, n, n)
        if raw_channels.shape[0] != num_samples:
            raise ValueError(f"Mismatch: {num_samples} graphs expected, but .mat contains {raw_channels.shape[0]} samples.")

        for i in range(num_samples):
            n = n_list[i]
            # Grab real channel matrix
            links = torch.as_tensor(raw_channels[i], dtype=torch.cfloat)  # shape [B, n, n]
            adj = (links.abs().sum(dim=0) > 0).float()  # [n, n] adjacency mask from non-zero links
            adj.fill_diagonal_(0)
            adj_list.append(adj)
            links_list.append(links)

    else:
        for i in range(num_samples):
            n = n_list[i]
            adj, links = generate_graph_and_channel_matrices(n, B)
            adj = torch.as_tensor(adj)
            links = torch.as_tensor(links)
            adj_list.append(adj)
            links_list.append(links)

    dataset = GraphNetDataset(adj_list, links_list, tx_list, rx_list, sigma_list, B)
    return dataset
