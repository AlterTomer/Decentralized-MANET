from torch_geometric.data import Data, Dataset
from torch_geometric.utils import dense_to_sparse
import torch
import copy
from utils.TensorUtils import create_normalized_tensor, normalize_power


class GraphNetDataset(Dataset):
    def __init__(self, adj_list: list, links_list: list, tx_list: list, rx_list: list, sigma_list: list, B: int,
        problem: str = "single",          # "single" | "multicast" | "multi"
        K: int = 1,  device=None):
        super().__init__()
        assert len(adj_list) == len(links_list) == len(tx_list) == len(rx_list) == len(sigma_list), "All lists must match"
        self.adj_list = adj_list
        self.links_list = links_list
        self.tx_list = tx_list
        self.rx_list = rx_list
        self.sigma_list = sigma_list
        self.B = B
        self.problem = str(problem).lower()
        self.K = K
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ---- validate problem/K coherence ----
        valid_modes = {"single", "multicast", "multi"}
        if self.problem not in valid_modes:
            raise ValueError(f"problem must be one of {valid_modes}, got {self.problem}")

        if self.problem == "single":
            if self.K is not None and self.K != 1:
                raise ValueError("For problem='single', K must be 1 (or None).")
            self.K = 1

        elif self.problem == "multi":
            if self.K is None or self.K < 1:
                raise ValueError("For problem='multi', you must provide a positive K.")

    def len(self):
        return len(self.adj_list)

    def __getitem__(self, idx):
        adj = self.adj_list[idx].to(self.device) if isinstance(self.adj_list[idx], torch.Tensor) else torch.tensor(
            self.adj_list[idx], device=self.device)
        links = self.links_list[idx].to(self.device) if isinstance(self.links_list[idx],
                                                                   torch.Tensor) else torch.tensor(self.links_list[idx],
                                                                                                   device=self.device)  # [B, n, n]
        edge_index, _ = dense_to_sparse(adj)
        n = adj.shape[0]

        # ---- edge attributes [E, 2B] from complex H ----
        edge_attr = []
        for i, j in edge_index.t():
            h_complex = [links[b, i, j] for b in range(self.B)]
            real_parts = torch.tensor([h.real for h in h_complex], device=self.device)
            imag_parts = torch.tensor([h.imag for h in h_complex], device=self.device)
            h_vec = torch.cat([real_parts, imag_parts])  # [2B]
            edge_attr.append(h_vec)
        edge_attr = torch.stack(edge_attr).to(self.device)  # [E, 2B]

        # ---- determine K for this sample ----
        rx_i = self.rx_list[idx]
        if isinstance(rx_i, (list, tuple)):
            K_sample = len(rx_i)
        else:
            K_sample = 1

        if self.problem == "single":
            K_eff = 1
        elif self.problem == "multi":
            K_eff = int(self.K)  # user-specified, fixed across dataset
        else:  # multicast
            K_eff = int(self.K) if (self.K is not None) else int(K_sample)

        # ---- initialize P0
        if self.problem == "multi":
            # x: [B, K, n, n]
            x4 = []
            for _b in range(self.B):
                x4_k = torch.stack([
                    create_normalized_tensor(n, n, mask=adj, device=links.device)
                    for _k in range(K_eff)
                ])  # [K, n, n]
                x4.append(x4_k)
            x = torch.stack(x4, dim=0)  # [B, K, n, n]
            x = normalize_power(x, adj, eps=1e-12)  # 4D projection
        else:
            # x: [B, n, n]  (single or multicast shared-message case)
            x = torch.stack([
                create_normalized_tensor(n, n, mask=adj, device=links.device)
                for _ in range(self.B)
            ])  # [B, n, n]
            x = normalize_power(x, adj, eps=1e-12)  # 3D projection

        # ---- build Data object
        data = Data(
            x=x,  # [B,n,n] or [B,K,n,n]
            edge_index=edge_index.to(self.device),
            edge_attr=edge_attr,
        )
        data.sample_id = torch.tensor(idx)
        data.adj_matrix = adj
        data.links_matrix = links  # [B, n, n]
        data.sigma = torch.tensor(self.sigma_list[idx], device=self.device)
        data.B = self.B
        data.tx = self.tx_list[idx]
        data.rx = rx_i  # int or list[int]
        data.problem = self.problem
        data.K = int(K_eff)  # <-- ALWAYS set, even multicast
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
        if hasattr(data, "clone"):
            out = data.clone()
        else:
            out = copy.copy(data)
        out.adj_matrix = data.adj_matrix
        out.links_true = data.links_matrix
        out.links_matrix = self.H_hats[idx]
        out.sample_id = self.sample_ids[idx]
        return out
