from torch_geometric.data import Data, Dataset
from torch_geometric.utils import dense_to_sparse
import torch


class GraphNetDataset(Dataset):
    def __init__(self, adj_list, links_list, tx_list, rx_list, sigma_list, B):
        super().__init__()
        assert len(adj_list) == len(links_list) == len(tx_list) == len(rx_list) == len(sigma_list), "All lists must match"
        self.adj_list = adj_list
        self.links_list = links_list
        self.tx_list = tx_list
        self.rx_list = rx_list
        self.sigma_list = sigma_list
        self.B = B

    def len(self):
        return len(self.adj_list)

    def __getitem__(self, idx):
        adj = self.adj_list[idx]
        links = self.links_list[idx]  # shape: [B, n, n], complex tensor
        edge_index, _ = dense_to_sparse(adj)
        edge_attr = []

        for i, j in edge_index.t():
            h_complex = [links[b, i, j] for b in range(self.B)]
            real_parts = torch.tensor([h.real for h in h_complex])
            imag_parts = torch.tensor([h.imag for h in h_complex])
            h_vec = torch.cat([real_parts, imag_parts])  # shape: [2*B]
            edge_attr.append(h_vec)

        edge_attr = torch.stack(edge_attr)  # shape: [E, 2B]

        n = adj.shape[0]
        node_feats = torch.full((n, self.B), 1.0 / (self.B ** 0.5))

        data = Data(x=node_feats, edge_index=edge_index, edge_attr=edge_attr)
        data.adj_matrix = adj
        data.links_matrix = links
        data.sigma = self.sigma_list[idx]
        data.B = self.B
        data.tx = self.tx_list[idx]
        data.rx = self.rx_list[idx]
        return data