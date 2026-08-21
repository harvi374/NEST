"""ClassHead — 2-layer classification head with batch norm and dropout."""

import torch.nn as nn

from src.config import ExperimentConfig


class ClassHead(nn.Module):
    def __init__(self, in_dim: int, cfg: ExperimentConfig, n_classes: int = 2):
        super().__init__()
        h1 = max(128, in_dim * 2)
        h2 = h1 // 2
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(),
            nn.Dropout(cfg.head_dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(cfg.head_dropout / 2),
            nn.Linear(h2, n_classes),
        )

    def forward(self, x):
        return self.net(x)
