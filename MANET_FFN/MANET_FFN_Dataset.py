import torch
from torch.utils.data import Dataset

class FFNDataset(Dataset):
    """
    Plain tensor dataset for the FFN benchmark.

    Parameters
    ----------
    adj_list : list
        List of adjacency matrices. Each element has shape [n, n].
    links_list : list
        List of complex CSI tensors. Each element has shape [B, n, n].
    tx_list : list
        Transmitter definitions. Format depends on the problem type.
    rx_list : list
        Receiver definitions. Format depends on the problem type.
    sigma_list : list
        Noise standard deviations, one per graph/sample.
    B : int
        Number of frequency bands.
    problem : str
        One of {"single", "multicast", "multi", "converge", "multiunicast"}.
        "multiunicast" is the many-to-many case.
    K : int
        Number of messages/commodities for multi-message cases.
    device : torch.device or str, optional
        Optional target device. Usually left as None and moved in the training loop.

    Returns
    -------
    dict
        Sample dictionary containing links_matrix, adj_matrix, sigma, tx, rx, B,
        K, problem, and sample_id.
    """

    def __init__(
        self,
        adj_list,
        links_list,
        tx_list,
        rx_list,
        sigma_list,
        B: int,
        problem: str = "single",
        K: int = 1,
        device=None,
    ):
        assert len(adj_list) == len(links_list) == len(tx_list) == len(rx_list) == len(sigma_list), \
            "All input lists must have the same length."

        self.adj_list = adj_list
        self.links_list = links_list
        self.tx_list = tx_list
        self.rx_list = rx_list
        self.sigma_list = sigma_list
        self.B = int(B)
        self.problem = problem.lower()
        self.K = int(K)
        self.device = device

        valid_modes = {"single", "multicast", "multi", "converge", "multiunicast"}
        if self.problem not in valid_modes:
            raise ValueError(f"Unknown problem type: {self.problem}. Expected one of {valid_modes}.")

        if self.problem == "single":
            self.K = 1

    @staticmethod
    def _to_list(x):
        """
        Convert tensors/tuples to Python lists while leaving scalar values unchanged.

        Parameters
        ----------
        x : object
            Tensor, list, tuple, or scalar.

        Returns
        -------
        object
            List if x is tensor/list/tuple, otherwise x itself.
        """
        if isinstance(x, torch.Tensor):
            return x.view(-1).tolist()
        if isinstance(x, (list, tuple)):
            return list(x)
        return x

    def __len__(self):
        """
        Returns
        -------
        int
            Number of graph samples in the dataset.
        """
        return len(self.adj_list)

    def __getitem__(self, idx):
        """
        Fetch one graph sample.

        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        dict
            Tensor sample and metadata used by the FFN training loop.
        """
        adj = self.adj_list[idx]
        links = self.links_list[idx]

        if not isinstance(adj, torch.Tensor):
            adj = torch.tensor(adj, dtype=torch.float32)
        else:
            adj = adj.float()

        if not isinstance(links, torch.Tensor):
            links = torch.tensor(links, dtype=torch.cfloat)
        elif not torch.is_complex(links):
            links = links.to(torch.cfloat)

        sigma = torch.tensor(self.sigma_list[idx], dtype=torch.float32)

        tx_i = self.tx_list[idx]
        rx_i = self.rx_list[idx]

        if self.problem == "single":
            tx = int(tx_i)
            rx = int(rx_i)
            K_eff = 1

        elif self.problem == "multicast":
            tx = int(tx_i)
            rx = self._to_list(rx_i)
            if not isinstance(rx, list):
                rx = [int(rx)]
            K_eff = len(rx)

        elif self.problem == "multi":
            tx = int(tx_i)
            rx = self._to_list(rx_i)[:self.K]
            K_eff = self.K

        elif self.problem == "converge":
            tx = self._to_list(tx_i)[:self.K]
            rx = int(rx_i)
            K_eff = self.K

        else:  # multiunicast / many-to-many
            tx = self._to_list(tx_i)[:self.K]
            rx = self._to_list(rx_i)[:self.K]
            K_eff = self.K

        sample = {
            "links_matrix": links,   # [B, n, n] complex CSI tensor
            "adj_matrix": adj,       # [n, n] directed adjacency matrix
            "sigma": sigma,          # scalar noise standard deviation
            "tx": tx,                # source node(s)
            "rx": rx,                # destination node(s)
            "B": self.B,
            "K": K_eff,
            "problem": self.problem,
            "sample_id": idx,
        }

        if self.device is not None:
            sample["links_matrix"] = sample["links_matrix"].to(self.device)
            sample["adj_matrix"] = sample["adj_matrix"].to(self.device)
            sample["sigma"] = sample["sigma"].to(self.device)

        return sample
