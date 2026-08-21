"""SAGEModel — pure GraphSAGE baseline for comparison (same dims as FullGAT)."""

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

from src.config import ExperimentConfig


class SAGEModel(nn.Module):
    """SAGE baseline — same dims as PGFCL for fair comparison."""

    def __init__(self, in_dim: int, cfg: ExperimentConfig, n_classes: int = 2):
        super().__init__()
        h, e = cfg.hidden, cfg.emb_dim
        self.conv1   = SAGEConv(in_dim, h)
        self.conv2   = SAGEConv(h, h)
        self.conv3   = SAGEConv(h, e)
        self.skip1   = nn.Linear(in_dim, h, bias=False)
        self.skip2   = nn.Linear(h, h, bias=False)
        self.bn1     = nn.BatchNorm1d(h)
        self.bn2     = nn.BatchNorm1d(h)
        self.head    = nn.Sequential(
            nn.Linear(e, e * 2), nn.ReLU(), nn.Linear(e * 2, n_classes)
        )
        self.dropout = cfg.dropout

    def forward(self, x, edge_index):
        h1 = F.relu(self.bn1(self.conv1(x, edge_index) + self.skip1(x)))
        h1 = F.dropout(h1, p=self.dropout, training=self.training)
        h2 = F.relu(self.bn2(self.conv2(h1, edge_index) + self.skip2(h1)))
        h2 = F.dropout(h2, p=self.dropout, training=self.training)
        emb = self.conv3(h2, edge_index)
        return self.head(emb), emb
