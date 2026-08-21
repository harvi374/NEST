"""
Graph utilities (subgraph extraction).
"""

import torch
from torch_geometric.data import Data


def get_local_edge_index(data: Data, mask: torch.Tensor,
                          device: torch.device) -> torch.Tensor:
    """Edges where BOTH endpoints are inside mask."""
    ei   = data.edge_index.to(device)
    m    = mask.to(device)
    keep = m[ei[0]] & m[ei[1]]
    return ei[:, keep]


def get_inductive_edge_index(data, train_mask, test_mask, device):
    """Inductive test edges — at least one endpoint in test_mask."""
    ei    = data.edge_index.to(device)
    tr    = train_mask.to(device)
    te    = test_mask.to(device)
    all_m = tr | te
    keep  = (te[ei[1]] & all_m[ei[0]]) | (te[ei[0]] & te[ei[1]])
    return ei[:, keep]
