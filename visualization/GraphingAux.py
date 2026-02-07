import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import networkx as nx
import numpy as np
import os
import torch
from utils.MetricUtils import link_rate

def plot_train_valid_loss(train_loss, valid_rate, filename=False):
    """
    Plot a loss curve vs. epochs
    Args:
        train_loss: Training loss array
        valid_rate: Validation rate array
        filename: If not False, save the plot
    """
    epochs = np.arange(len(train_loss))

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))
    # --- Training Loss ---
    axes[0].plot(epochs, train_loss)
    axes[0].set_xlabel('Epoch', fontsize=30)
    axes[0].set_ylabel('Loss', fontsize=30)
    axes[0].set_title('Train Loss', fontsize=30)
    axes[0].grid(True)

    # --- Validation Rate ---
    axes[1].plot(epochs, valid_rate)
    axes[1].set_xlabel('Epoch', fontsize=30)
    axes[1].set_ylabel('Rate', fontsize=30)
    axes[1].set_title('Valid Rate', fontsize=30)
    axes[1].grid(True)


    plt.tight_layout()
    if filename:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()


def plot_mean_rate_vs_snr(snr_db, results, save_path=None):
    """
    Benchmark plots of rate vs. snr (centralized optimization, decentralized optimization, brute search, equal power)

    Args:
        snr_db: List of SNR values in dB.
        results: Results dict of rates for each snr value.
        save_path: Save path for saving plots, if None just show the plot.

    """
    adam = list(results["centralized"].values())
    gnn = list(results["gnn"].values())
    sbn = list(results["strongest bottleneck"].values())
    ep = list(results["equal power"].values())
    gp = list(results["greedy maxlink"].values())

    plt.figure(figsize=(16, 12))
    plt.plot(snr_db, adam, marker="o", label="Centralized Optimizer", markersize=12)
    plt.plot(snr_db, gnn,  marker="s", label="MANET-GNN", linestyle="dashed", markersize=12)
    plt.plot(snr_db, sbn,   marker="^", label="Best Single Channel", linestyle="dotted", markersize=12)
    plt.plot(snr_db, ep, marker="+", label="Equal-Split", linestyle="dashdot", markersize=12)
    plt.plot(snr_db, gp, marker="*", label="Greedy-Split", linestyle='-', markersize=12)


    plt.yscale("log")
    plt.xlabel("SNR (dB)", fontsize=30)
    plt.ylabel("Mean Rate", fontsize=30)
    plt.grid(True, which="both")
    plt.legend(fontsize=30)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=450)
        plt.close()
    else:
        plt.show()


def time_varying_model_compare_plot(snr_db, results, n_big, n_small, save_path=None):
    """
    Time varying topologies comparison plots of rate vs. snr (models were trained on different topologies, the data is based on one of the topologies)

    Args:
        snr_db: List of SNR values in dB.
        results: Results dict of rates for each snr value.
        n_big: Number of nodes in large topology.
        n_small: Number of nodes in small topology.
        save_path: Save path for saving plots, if None just show the plot.

    """
    big_rates = [results["big"][s] for s in snr_db]
    small_rates = [results["small"][s] for s in snr_db]


    plt.figure(figsize=(16, 12))
    plt.plot(snr_db, big_rates, marker="o", label=r"Train $|\mathcal{V}|=$" + f'{n_big} →' + r'Test $|\mathcal{V}|=$' + f'{n_big}', markersize=12)
    plt.plot(snr_db, small_rates,  marker="s", label=r"Train $|\mathcal{V}|=$" + f'{n_small} →' + r'Test $|\mathcal{V}|=$' + f'{n_big}', markersize=12, linestyle="dashed")

    plt.yscale("log")
    plt.xlabel("SNR (dB)", fontsize=30)
    plt.ylabel("Mean Rate", fontsize=30)
    plt.grid(True, which="both")
    plt.legend(fontsize=30)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=450)
        plt.close()
    else:
        plt.show()

def est_true_model_compare_plot(snr_db, results, save_path=None):
    """
    True CSI model compared with estimated CSI model plots of rate vs. snr (models were trained on different topologies, the data is based on one of the topologies)

    Args:
        snr_db: List of SNR values in dB.
        results: Results dict of rates for each snr value.
        save_path: Save path for saving plots, if None just show the plot.

    """
    eps = 1e-9
    true_rates = [max(results["true"][s], eps) for s in snr_db]
    est_rates = [max(results["est"][s], eps) for s in snr_db]


    plt.figure(figsize=(16, 12))
    plt.plot(snr_db, true_rates, marker="o", label="True CSI Model")
    plt.plot(snr_db, est_rates,  marker="s", label="Estimated CSI Model")

    plt.yscale("log")
    plt.xlabel("SNR (dB)", fontsize=30)
    plt.ylabel("Mean Rate", fontsize=30)
    plt.grid(True, which="both")
    plt.legend(fontsize=30)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=450)
        plt.close()
    else:
        plt.show()


def plot_models_mean_rate_vs_snr(snr_db, results, save_path=None):
    """
    Plot mean rate vs SNR for multiple trained models (one curve per model).

    Args:
        snr_db: list of SNR values in dB.
        results: dict returned by evaluate_models_across_snr:
                 {"models": {name: {snr_db: mean_rate}}}
        save_path: if provided, save figure; else show.
    """
    model_dict = results.get("models", {})
    if not model_dict:
        raise ValueError("results['models'] is empty or missing.")

    plt.figure(figsize=(16, 12))

    # Keep a consistent x ordering
    snr_db = list(snr_db)

    # Plot each model as one curve
    # (Markers cycle automatically; you can also hardcode if you want reproducibility)
    for name, snr_to_rate in model_dict.items():
        y = [snr_to_rate[s] for s in snr_db]
        plt.plot(snr_db, y, marker="o", label=str(name), markersize=12)

    plt.yscale("log")
    plt.xlabel("SNR (dB)", fontsize=30)
    plt.ylabel("Mean Rate", fontsize=30)
    plt.grid(True, which="both")
    plt.legend(fontsize=24)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=450)
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
    plt.figure(figsize=(16, 12))

    # Draw base graph with light gray edges
    nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="gray", width=0.5, node_size=500, font_size=10)

    # Colors for different frequency bands
    colors = [mcolors.to_hex(np.random.rand(3,)) for _ in range(B)]

    # Legend entries
    legend_patches = []

    # Draw the best paths in different colors
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