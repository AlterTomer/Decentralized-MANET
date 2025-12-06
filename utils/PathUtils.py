import torch
from typing import List

def find_all_paths(adj_matrix: torch.Tensor, start: int, destination: int) -> List[List[int]]:
    """
    Finds all paths between two nodes using iterative DFS and returns
    a padded tensor of node indices and path lengths.

    Args:
        adj_matrix (torch.Tensor): [n, n] adjacency matrix (on any device).
        start (int): start node index.
        destination (int): destination node index.

    Returns:
        all_paths (List[List[int]]): List of all paths between start and destination.
    """
    stack = [(int(start), [int(start)])]
    all_paths = []

    while stack:
        node, path = stack.pop()
        if node == destination:
            all_paths.append(path)
            continue

        # Find neighbors: j such that adj_matrix[node, j] != 0
        neighbors = (adj_matrix[node] != 0).nonzero(as_tuple=True)[0].tolist()
        for neighbor in reversed(neighbors):  # reverse to match recursive DFS order
            if neighbor not in path:  # avoid cycles
                stack.append((neighbor, path + [neighbor]))

    return all_paths

def paths_to_tensor(paths, device):
    """
    Converts a list of paths (each a list of node indices)
    into a padded tensor for vectorized calc_sum_rate.

    Args:
        paths (List[List[int]]): list of paths, each a list of node indices.
        device: torch.device for the resulting tensor.

    Returns:
        torch.LongTensor: [num_paths, max_path_len] padded with -1.
    """
    if len(paths) == 0:
        return torch.empty((0, 0), dtype=torch.long, device=device)

    max_len = max(len(p) for p in paths)
    num_paths = len(paths)

    paths_tensor = torch.full((num_paths, max_len), -1, dtype=torch.long, device=device)

    for i, path in enumerate(paths):
        paths_tensor[i, :len(path)] = torch.tensor(path, dtype=torch.long, device=device)

    return paths_tensor