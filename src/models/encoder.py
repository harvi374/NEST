"""
SAGEGATEncoder — the core graph encoder used across all experiments.

Architecture: SAGEConv ×2 (with skip connections) → GATv2Conv (query-dependent attention).
Supports ordered-dropout (FjORD) via forward_od() and attention saliency
extraction via forward_with_attention().
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, SAGEConv

from src.config import ExperimentConfig


class SAGEGATEncoder(nn.Module):
    """
    Encoder: SAGEConv ×2 (skip connections) → GATv2Conv (query-dependent attention).
    forward_with_attention() returns GAT attention weights for saliency logging
    at zero extra compute cost.
    """

    def __init__(self, in_dim: int, cfg: ExperimentConfig, edge_dim: int = None):
        super().__init__()
        h, e, dr = cfg.hidden, cfg.emb_dim, cfg.dropout
        self.dropout = dr
        self.conv1 = SAGEConv(in_dim, h)
        self.conv2 = SAGEConv(h, h)
        self.conv3 = GATv2Conv(h, e, heads=1, dropout=dr, concat=False,
                               edge_dim=edge_dim)
        self.skip1 = nn.Linear(in_dim, h, bias=False)
        self.skip2 = nn.Linear(h, h, bias=False)
        self.bn1   = nn.BatchNorm1d(h)
        self.bn2   = nn.BatchNorm1d(h)
        self.proj_head = nn.Sequential(
            nn.Linear(e, e * 2), nn.ReLU(), nn.Linear(e * 2, e)
        )
        self.raw_projector = nn.Linear(in_dim, e, bias=False)

    def _sage_layers(self, x, edge_index):
        h1 = self.conv1(x, edge_index) + self.skip1(x)
        h1 = F.relu(self.bn1(h1))
        h1 = F.dropout(h1, p=self.dropout, training=self.training)
        h2 = self.conv2(h1, edge_index) + self.skip2(h1)
        h2 = F.relu(self.bn2(h2))
        h2 = F.dropout(h2, p=self.dropout, training=self.training)
        return h2

    def forward(self, x, edge_index, edge_attr=None):
        h2 = self._sage_layers(x, edge_index)
        return self.conv3(h2, edge_index, edge_attr=edge_attr)

    def forward_with_attention(self, x, edge_index):
        h2 = self._sage_layers(x, edge_index)
        emb, (att_edge_index, att_weights) = self.conv3(
            h2, edge_index, return_attention_weights=True
        )
        return emb, att_edge_index, att_weights

    def encode_with_proj(self, x, edge_index, edge_attr=None):
        z = self.forward(x, edge_index, edge_attr=edge_attr)
        return z, self.proj_head(z)

    def forward_od(self, x, edge_index, width_ratio: float):
        """Ordered Dropout (FjORD) forward pass. Zeroes the tail channels of
        every hidden/output layer beyond width_ratio * layer_width, in a fixed
        channel order, so a low-budget client's active sub-network is always a
        strict nested prefix of every larger client's sub-network."""
        def _od_mask(h):
            keep = max(1, round(h.shape[1] * width_ratio))
            mask = torch.zeros_like(h)
            mask[:, :keep] = 1.0
            return h * mask

        h1 = self.conv1(x, edge_index) + self.skip1(x)
        h1 = F.relu(self.bn1(h1))
        h1 = _od_mask(h1)
        h1 = F.dropout(h1, p=self.dropout, training=self.training)
        h2 = self.conv2(h1, edge_index) + self.skip2(h1)
        h2 = F.relu(self.bn2(h2))
        h2 = _od_mask(h2)
        h2 = F.dropout(h2, p=self.dropout, training=self.training)
        emb = self.conv3(h2, edge_index)
        emb = _od_mask(emb)
        return emb
