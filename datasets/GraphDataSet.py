from torch_geometric.data import Data, Dataset
from torch_geometric.utils import dense_to_sparse
import torch
import copy
from utils.TensorUtils import create_normalized_tensor, normalize_power


class GraphNetDataset(Dataset):
    def __init__(self, adj_list, links_list, tx_list, rx_list, sigma_list, B, device=None):
        super().__init__()
        assert len(adj_list) == len(links_list) == len(tx_list) == len(rx_list) == len(sigma_list), "All lists must match"
        self.adj_list = adj_list
        self.links_list = links_list
        self.tx_list = tx_list
        self.rx_list = rx_list
        self.sigma_list = sigma_list
        self.B = B
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def len(self):
        return len(self.adj_list)

    def __getitem__(self, idx):
        adj = self.adj_list[idx].to(self.device) if isinstance(self.adj_list[idx], torch.Tensor) else torch.tensor(self.adj_list[idx], device=self.device)
        links = self.links_list[idx].to(self.device) if isinstance(self.links_list[idx], torch.Tensor) else torch.tensor(self.links_list[idx], device=self.device)  # shape: [B, n, n]
        edge_index, _ = dense_to_sparse(adj)
        n = adj.shape[0]
        # === Physical Edge Attributes ===
        edge_attr = []
        for i, j in edge_index.t():
            h_complex = [links[b, i, j] for b in range(self.B)]
            real_parts = torch.tensor([h.real for h in h_complex], device=self.device)
            imag_parts = torch.tensor([h.imag for h in h_complex], device=self.device)
            h_vec = torch.cat([real_parts, imag_parts])  # shape: [2*B]
            edge_attr.append(h_vec)

        edge_attr = torch.stack(edge_attr).to(self.device)  # shape: [E, 2B]

        node_feats = torch.stack([create_normalized_tensor(adj.shape[0], adj.shape[1], mask=adj, device=links.device) for _ in range(self.B)])
        node_feats = normalize_power(node_feats, adj, eps=1e-12)

        data = Data(x=node_feats, edge_index=edge_index.to(self.device), edge_attr=edge_attr)
        data.sample_id = torch.tensor(idx)
        data.adj_matrix = adj
        data.links_matrix = links
        data.sigma = torch.tensor(self.sigma_list[idx], device=self.device)
        data.B = self.B
        data.tx = self.tx_list[idx]
        data.rx = self.rx_list[idx]
        return data


class SupervisedGraphNetDataset(Dataset):
    """
    Extends an existing GraphNetDataset by attaching ground-truth
    ADAM power tensors  (B,n,n)  and scalar rates  r_opt.
    """
    def __init__(self, triplet_list, B, device=None):
        super().__init__()
        self.triplet_list = triplet_list  # list of (data, p_opt, r_opt)
        self.B = B
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def len(self):
        return len(self.triplet_list)

    def __getitem__(self, idx):
        data, p_opt, r_opt = self.triplet_list[idx]

        # Move data and targets to the correct device
        data = data.to(self.device)
        p_opt = p_opt.to(self.device) if isinstance(p_opt, torch.Tensor) else torch.tensor(p_opt, dtype=torch.float32,
                                                                                           device=self.device)
        r_opt = torch.tensor(r_opt, dtype=torch.float32, device=self.device)

        # Attach supervised labels to the Data object
        data.p_opt = p_opt
        data.r_opt = r_opt

        return data


class EstimatedCSIDataset(Dataset):
    """
    Wraps a base dataset and stores precomputed H_hat per sample.
    __getitem__ returns a shallow copy with links_matrix := H_hat and also
    exposes .links_true (the original H).
    """
    def __init__(self, base_dataset, H_hats, sample_ids):
        super().__init__()
        assert len(H_hats) == len(base_dataset) == len(sample_ids)
        self.base = base_dataset
        self.H_hats = H_hats
        self.sample_ids = sample_ids

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        data = self.base[idx]
        # Make a shallow copy of the Data object (PyG Data has clone(); fallback to constructing a new one if needed)
        if hasattr(data, "clone"):
            out = data.clone()
        else:
            # If not PyG, assume a simple namespace-like object; adapt as needed.
            out = copy.copy(data)
        out.links_true = data.links_matrix
        out.links_matrix = self.H_hats[idx]
        out.sample_id = self.sample_ids[idx]
        return out
