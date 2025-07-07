import torch
import torch.nn.functional as F
from utils import find_all_paths, calc_sum_rate, expand_power_allocation

def train_GraphNet(model, loader, optimizer, device):
    """
    Training function for 1 epoch GraphNet training
    Args:
        model: GraphNet model
        loader: Train data loader
        optimizer: Optimizer object (SGD, ADAM, ...)
        device: cpu/cuda

    Returns: Train loss

    """
    model.train()
    total_loss = 0
    for data in loader:

        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)

        n, B = out.size()
        p_arr = expand_power_allocation(out, data.adj_matrix)

        paths = find_all_paths(data.adj_matrix, data.tx.item(), data.rx.item())
        if len(paths) < 1:
            continue
        loss = -calc_sum_rate(data.links_matrix, p_arr, data.sigma, paths, B)
        try:
            loss.backward()
            optimizer.step()
        except AttributeError:
            continue

        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def validate_GraphNet(model, loader, device, deep_supervision=False):
    """
    Validation function for 1 epoch GraphNet training
    Args:
        model: GraphNet model
        loader: Valid data loader
        device: cpu/cuda
        deep_supervision: If True, calculates loss after each layer (deep supervision).
                  If False, calculates loss only on the final output.

    Returns: Validation loss

    """
    model.eval()
    total_rate = 0
    for data in loader:
        data = data.to(device)
        out = model(data)
        if deep_supervision:
            out = out[-1]
        n, B = out.size()
        p_arr = expand_power_allocation(out, data.adj_matrix)

        paths = find_all_paths(data.adj_matrix, data.tx.item(), data.rx.item())
        rate = calc_sum_rate(data.links_matrix, p_arr, data.sigma, paths, B)
        total_rate += rate.item()
    return total_rate / len(loader)


# ------------------------------------------- Chained Net --------------------------------------------------------------
def train_chained(model, loader, optimizers, device):
    """
    Trains a ChainedGNN model with layer-wise supervision, where each layer is trained independently
    using its own optimizer and receives gradients only from its own output.

    This function assumes that the model returns a list of layers (as nn.ModuleList),
    and each layer behaves like a GNN block (GCN, NNConv, GAT, etc.).
    Each layer receives the detached output of the previous one as its input,
    mimicking the generator-discriminator training pattern (e.g., in GANs).

    Args:
        model (nn.Module): A ChainedGNN model composed of independently trainable layers.
        loader (DataLoader): PyTorch DataLoader providing batches of PyG Data objects.
        optimizers (List[torch.optim.Optimizer]): A list of optimizers, one per GNN layer in the model.
        device (torch.device): The device ('cpu' or 'cuda') to run the training on.

    Returns:
        float: The average training loss (sum-rate loss) across all layers and all samples.
    """
    model.train()
    total_loss = 0
    count = 0

    for data in loader:
        data = data.to(device)
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # Find all paths from tx to rx
        paths = find_all_paths(data.adj_matrix, data.tx.item(), data.rx.item())
        if len(paths) < 1:
            continue  # skip if no valid communication path

        # Layer-wise training
        for i, (layer, optimizer) in enumerate(zip(model.layers, optimizers)):
            optimizer.zero_grad()

            # Forward pass through one layer
            x = layer(x, edge_index, edge_attr)

            # Normalize power allocation for this layer's output
            norm_out = F.softmax(x, dim=1)
            norm_out = torch.sqrt(norm_out + 1e-8)
            norm_out = norm_out / norm_out.norm(p=2, dim=1, keepdim=True)

            # Expand to [B, n, n] using adjacency
            p_arr = expand_power_allocation(norm_out, data.adj_matrix)

            # Compute negative sum-rate loss
            loss = -calc_sum_rate(data.links_matrix, p_arr, data.sigma, paths, model.B)

            # Backprop and update only this layer
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            count += 1

            # Detach output to prevent graph retention
            x = x.detach()

    return total_loss / count if count > 0 else 0.0

@torch.no_grad()
def validate_chained(model, loader, device):
    """
    Evaluates the ChainedGNN model using the final output (after all layers).
    No gradient updates are performed.

    Args:
        model: ChainedGNN model
        loader: DataLoader providing test/validation data
        device: CPU or CUDA device

    Returns:
        float: Mean validation sum-rate
    """
    model.eval()
    total_rate = 0
    count = 0

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

            paths = find_all_paths(data.adj_matrix, data.tx.item(), data.rx.item())
            if len(paths) < 1:
                continue

            # Forward through all layers
            for layer in model.layers:
                x = layer(x, edge_index, edge_attr)

            # Normalize output for power allocation
            norm_out = F.softmax(x, dim=1)
            norm_out = torch.sqrt(norm_out + 1e-8)
            norm_out = norm_out / norm_out.norm(p=2, dim=1, keepdim=True)

            # Expand and evaluate rate
            p_arr = expand_power_allocation(norm_out, data.adj_matrix)
            rate = calc_sum_rate(data.links_matrix, p_arr, data.sigma, paths, model.B)

            total_rate += rate.item()
            count += 1

    return total_rate / count if count > 0 else 0.0

