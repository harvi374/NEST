"""FullGAT — composes SAGEGATEncoder + ClassHead into the main model."""

import torch
import torch.nn as nn

from src.config import ExperimentConfig
from src.models.encoder import SAGEGATEncoder
from src.models.head import ClassHead


class FullGAT(nn.Module):
    def __init__(self, in_dim: int, cfg: ExperimentConfig, edge_dim: int = None):
        super().__init__()
        self.encoder = SAGEGATEncoder(in_dim, cfg, edge_dim=edge_dim)
        self.head    = ClassHead(cfg.emb_dim, cfg)

    def forward(self, x, edge_index, edge_attr=None, trunc_dim: int = None):
        """trunc_dim: EP-FedProto v3 fixed-tier baseline. When set,
        zeroes embedding dims >= trunc_dim BEFORE the classification head.
        None = full emb_dim, unchanged v10 behaviour."""
        emb = self.encoder(x, edge_index, edge_attr=edge_attr)
        if trunc_dim is not None:
            mask = torch.zeros_like(emb)
            mask[:, :trunc_dim] = 1.0
            emb = emb * mask
        return self.head(emb), emb
