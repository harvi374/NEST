"""
Data split utilities for federated learning.
"""

import torch
import numpy as np
from torch_geometric.data import Data

from src.config import ExperimentConfig


def make_mask(n: int, idx: np.ndarray) -> torch.Tensor:
    m = torch.zeros(n, dtype=torch.bool)
    if len(idx):
        m[torch.tensor(np.array(idx), dtype=torch.long)] = True
    return m


def temporal_federated_split(data: Data, cfg: ExperimentConfig):
    labels    = data.y.numpy()
    timesteps = data.timestep.numpy()
    valid_idx  = np.where(labels >= 0)[0]
    sorted_idx = valid_idx[np.argsort(timesteps[valid_idx])]
    splits     = np.array_split(sorted_idx, cfg.n_clients)

    def strat_split(arr):
        if len(arr) == 0:
            return arr, arr
        n_te = max(1, int(len(arr) * cfg.test_ratio))
        return arr[:-n_te], arr[-n_te:]

    clients = []
    for i, split in enumerate(splits):
        illicit        = split[labels[split] == 1]
        licit          = split[labels[split] == 0]
        ill_tr, ill_te = strat_split(illicit)
        lic_tr, lic_te = strat_split(licit)
        tr = np.concatenate([ill_tr, lic_tr])
        te = np.concatenate([ill_te, lic_te])
        clients.append({
            'id':         i,
            'train_mask': make_mask(data.num_nodes, tr),
            'test_mask':  make_mask(data.num_nodes, te),
            'n_train':    len(tr),
            'n_test':     len(te),
        })
        print(f'  Client {i}: train={len(tr):5d} test={len(te):4d} '
              f'ill_train={len(ill_tr):4d} ill_test={len(ill_te):3d}')
    return clients
