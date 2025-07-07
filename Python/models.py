import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch_geometric.nn import GCNConv, NNConv, MessagePassing
from torch_geometric.utils import softmax

# ---------------------------------------------- GATConv ---------------------------------------------------------------
class EdgeGATConv(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_dim, heads=1, dropout=0.0):
        super().__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout

        self.lin_node = nn.Linear(in_channels, heads * out_channels)
        self.lin_edge = nn.Linear(edge_dim, heads * out_channels)
        self.attn = nn.Parameter(torch.Tensor(1, heads, out_channels))
        nn.init.xavier_uniform_(self.attn)

    def forward(self, x, edge_index, edge_attr):
        x = self.lin_node(x)  # [N, H * F]
        e = self.lin_edge(edge_attr)  # [E, H * F]

        return self.propagate(edge_index, x=x, edge_attr=e)  # [N, H * F] in, out

    def message(self, x_j, edge_attr, index):
        num_heads = self.heads
        feat_dim = self.out_channels
        x_j = x_j.view(-1, num_heads, feat_dim)
        edge_attr = edge_attr.view(-1, num_heads, feat_dim)

    # # Debug: edge_attr statistics
    #     print("edge_attr stats:",
    #           "mean =", edge_attr.mean().item(),
    #           "std =", edge_attr.std().item(),
    #           "max =", edge_attr.max().item(),
    #           "min =", edge_attr.min().item())
        alpha = (x_j + edge_attr) * self.attn  # shape: [E, heads, feat_dim]

    # # Debug: alpha before activation
    #     print("alpha raw stats:",
    #           "mean =", alpha.mean().item(),
    #           "std =", alpha.std().item())
        alpha = F.leaky_relu(alpha.sum(dim=-1))  # shape: [E, heads]

    # # Debug: alpha after leaky_relu
    #     print("alpha after leaky_relu:",
    #           "mean =", alpha.mean().item(),
    #           "std =", alpha.std().item())
        # alpha = F.softmax(alpha, dim=1)
        alpha = softmax(alpha, index, dim=0)

    # # Debug: alpha after softmax
    #     print("alpha after softmax:",
    #           "mean =", alpha.mean().item(),
    #           "std =", alpha.std().item())

        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return (x_j * alpha.unsqueeze(-1)).reshape(x_j.size(0), -1)

# ---------------------------------------------- GatedGCN --------------------------------------------------------------
class GatedGCNLayer(MessagePassing):
    def __init__(self, in_dim, out_dim, B):
        super().__init__(aggr='add')
        self.W1 = nn.Linear(in_dim, out_dim)
        self.W2 = nn.Linear(in_dim, out_dim)
        self.edge_gate_net = GatedConvEdgeNet(B, out_dim)
        self.gate_act = nn.Softplus()
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x, edge_index, edge_attr):
        self.x_i = x  # Save for update
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        W2pj = self.W2(x_j)
        e_ij = self.edge_gate_net(edge_attr)         # [E, out_dim]
        gate = self.gate_act(e_ij)                   # element-wise gate
        return gate * W2pj

    def update(self, aggr_out):
        p_i = self.x_i
        h = self.W1(p_i) + aggr_out
        return F.relu(self.norm(h))


# ---------------------------------------------- GatedConvEdgeNet ---------------------------------------------------------------
class GatedConvEdgeNet(nn.Module):
    def __init__(self, B, out_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(2, 8, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(8, 1, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1)
        )
        self.fc = nn.Linear(B, out_dim)

        # Better weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, edge_attr):
        x = edge_attr.view(-1, 2, self.fc.in_features)  # [E, 2, B]
        x = self.conv(x).squeeze(1)  # [E, B]
        return self.fc(x)  # [E, out_dim]

# ---------------------------------------------- NNConvEdgeNet ---------------------------------------------------------------
class NNConvEdgeNet(nn.Module):
    def __init__(self, B, in_dim, out_dim):
        super().__init__()
        self.B = B
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=8, kernel_size=3, padding=1),  # (Re, Im) as channels
            nn.ReLU(),
            nn.Conv1d(in_channels=8, out_channels=1, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.fc = nn.Linear(B, in_dim * out_dim)

    def forward(self, edge_attr):
        x = edge_attr.view(-1, 2, self.B)  # [E, 2, B]
        x = self.conv(x).squeeze(1)       # [E, B]
        x = self.fc(x)                    # [E, in_dim * out_dim]
        return x

# ---------------------------------------------- GraphNet --------------------------------------------------------------
class GraphNet(nn.Module):
    def __init__(self, layer_sizes, layer_types, B, dropout=0.5):
        super(GraphNet, self).__init__()
        self.layers = nn.ModuleList()
        self.edge_nns = nn.ModuleList()
        self.dropout = dropout
        self.B = B

        for i, (in_dim, out_dim) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            if layer_types[i] == 'nn':
                edge_net = NNConvEdgeNet(B, in_dim, out_dim)
                self.edge_nns.append(edge_net)
                self.layers.append(NNConv(in_dim, out_dim, edge_net))
            elif layer_types[i] == 'conv':
                self.layers.append(GCNConv(in_dim, out_dim))
                self.edge_nns.append(None)
            elif layer_types[i] == 'gat':
                self.layers.append(EdgeGATConv(in_dim, out_dim, edge_dim=2*B, heads=1, dropout=dropout))
                self.edge_nns.append(None)
            elif layer_types[i] == 'gated':
                self.layers.append(GatedGCNLayer(in_dim, out_dim, B))
                self.edge_nns.append(None)
            else:
                raise ValueError(f"Unknown layer type: {layer_types[i]}")

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        for i, layer in enumerate(self.layers):
            if isinstance(layer, NNConv):
                x = layer(x, edge_index, edge_attr)
            elif isinstance(layer, GCNConv):
                edge_weight = edge_attr[:, :self.B].pow(2).sum(dim=1).sqrt()
                x = layer(x, edge_index, edge_weight=edge_weight)
            elif isinstance(layer, (EdgeGATConv, GatedGCNLayer)):
                x = layer(x, edge_index, edge_attr)
            else:
                x = layer(x, edge_index)

            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        x = F.softmax(x, dim=1)
        x = torch.sqrt(x + 1e-8)
        x = x / x.norm(p=2, dim=1, keepdim=True)
        return x


# ------------------------------------------- Single Layer -------------------------------------------------------------
class SingleGNNLayer(nn.Module):
    def __init__(self, layer_type, in_dim, out_dim, B, dropout=0.5):
        super().__init__()
        self.B = B
        self.layer_type = layer_type
        self.dropout = dropout

        if layer_type == 'nn':
            self.edge_net = NNConvEdgeNet(B, in_dim, out_dim)
            self.layer = NNConv(in_dim, out_dim, self.edge_net)
        elif layer_type == 'conv':
            self.layer = GCNConv(in_dim, out_dim)
            self.edge_net = None
        elif layer_type == 'gat':
            self.layer = EdgeGATConv(in_dim, out_dim, edge_dim=2 * B, heads=1, dropout=dropout)
            self.edge_net = None
        elif layer_type == 'gated':
            self.layer = GatedGCNLayer(in_dim, out_dim, B)
            self.edge_net = None
        else:
            raise ValueError(f"Unsupported layer type: {layer_type}")

    def forward(self, x, edge_index, edge_attr):
        if self.layer_type == 'nn':
            x = self.layer(x, edge_index, edge_attr)
        elif self.layer_type == 'conv':
            edge_weight = edge_attr[:, :self.B].pow(2).sum(dim=1).sqrt()
            x = self.layer(x, edge_index, edge_weight=edge_weight)
        elif self.layer_type in ['gat', 'gated']:
            x = self.layer(x, edge_index, edge_attr)
        else:
            raise RuntimeError("Unknown layer type.")

        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x

# ------------------------------------------- Chained Net --------------------------------------------------------------
class ChainedGNN(nn.Module):
    def __init__(self, layer_sizes, layer_types, B, dropout=0.5):
        super().__init__()
        self.B = B
        self.layers = nn.ModuleList([
            SingleGNNLayer(layer_type, in_dim, out_dim, B, dropout)
            for in_dim, out_dim, layer_type in zip(layer_sizes[:-1], layer_sizes[1:], layer_types)
        ])

    def forward(self, x, edge_index, edge_attr):
        outputs = []
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
            outputs.append(x)
        return outputs  # list of outputs from each layer
