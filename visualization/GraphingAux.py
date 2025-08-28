import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import networkx as nx
import numpy as np
import os
import torch
from MetricUtils import link_rate

def plot_train_valid_loss(train_loss, valid_rate, filename=False):
    """
    Plot a loss curve vs epochs
    Args:
        train_loss: Training loss array
        valid_rate: Validation rate array
        filename: If not False, save the plot
    """
    epochs = np.arange(len(train_loss))

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))
    # --- Training Loss ---
    axes[0].plot(epochs, train_loss)
    axes[0].set_xlabel('Epoch', fontsize=14)
    axes[0].set_ylabel('Loss', fontsize=14)
    axes[0].set_title('Train Loss', fontsize=14)
    axes[0].grid(True)

    # --- Validation Rate ---
    axes[1].plot(epochs, valid_rate)
    axes[1].set_xlabel('Epoch', fontsize=14)
    axes[1].set_ylabel('Rate', fontsize=14)
    axes[1].set_title('Valid Rate', fontsize=14)
    axes[1].grid(True)


    plt.tight_layout()
    if filename:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()


def plot_mean_rate_vs_snr(snr_db, results, save_path=None):
    adam = list(results["adam"].values())
    gnn = list(results["gnn"].values())
    lb = list(results["lower"].values())
    if 'swa' in results.keys():
        swa = list(results["swa"].values())

    plt.figure(figsize=(10, 8))
    plt.plot(snr_db, adam, marker="o", label="Centralized Optimization")
    plt.plot(snr_db, gnn,  marker="s", label="Decentralized Optimization")
    plt.plot(snr_db, lb,   marker="^", label="Strongest Bottleneck")
    # if 'swa' in results.keys():
    #     plt.plot(snr_db, swa, marker="+", label="SWA")

    plt.yscale("log")
    plt.xlabel("SNR (dB)", fontsize=14)
    plt.ylabel("Mean Rate", fontsize=14)
    plt.title("Mean Sum-Rate vs SNR", fontsize=14)
    plt.grid(True, which="both")
    plt.legend(fontsize=14)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()

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