"""
ContribAgg and Rank-Aware ContribAgg — prototype quality weighted aggregation.
"""

import torch
import torch.nn.functional as F

from src.config import ExperimentConfig


def compute_client_contrib_weights(client_emb_ps, sizes, prev_global_protos,
                                    cfg: ExperimentConfig):
    """Standard ContribAgg — cosine similarity to consensus."""
    if prev_global_protos is None:
        total = sum(sizes)
        return [s / total for s in sizes]
    weights = []
    for i, c_protos in enumerate(client_emb_ps):
        p_licit = c_protos[0].float()
        g_licit = prev_global_protos[0].float()
        if p_licit.norm() < 1e-8 or g_licit.norm() < 1e-8:
            quality = 1.0
        else:
            cos_sim = F.cosine_similarity(p_licit.unsqueeze(0), g_licit.unsqueeze(0)).item()
            quality = cfg.contrib_floor + (1.0 - cfg.contrib_floor) * (cos_sim + 1.0) / 2.0
        weights.append((sizes[i] ** 0.5) * quality)
    total_w = sum(weights) + 1e-8
    return [w / total_w for w in weights]


def rank_aware_contrib_agg(client_full_protos: list, client_tier_dims: list,
                            client_sizes: list, prev_global_full_protos,
                            cfg: ExperimentConfig, tier_dims=(8, 16, 32, 64)) -> dict:
    """Rank-aware aggregator, dimension-wise."""
    boundaries = [0] + list(tier_dims)
    n = len(client_full_protos)
    seg_out = {0: [], 1: []}

    for seg_lo, seg_hi in zip(boundaries[:-1], boundaries[1:]):
        eligible = [i for i in range(n) if client_tier_dims[i] >= seg_hi]
        if not eligible:
            max_dim = max(client_tier_dims)
            eligible = [i for i in range(n) if client_tier_dims[i] == max_dim]

        weights = []
        for i in eligible:
            sz = client_sizes[i]
            if prev_global_full_protos is None:
                weights.append(sz ** 0.5)
                continue
            local_seg  = client_full_protos[i][1][seg_lo:seg_hi]
            global_seg = prev_global_full_protos[1][seg_lo:seg_hi]
            if local_seg.norm() < 1e-8 or global_seg.norm() < 1e-8:
                quality = 1.0
            else:
                cos_sim = F.cosine_similarity(local_seg.unsqueeze(0),
                                               global_seg.unsqueeze(0)).item()
                quality = cfg.contrib_floor + (1.0 - cfg.contrib_floor) * (cos_sim + 1.0) / 2.0
            weights.append((sz ** 0.5) * quality)
        total_w = sum(weights) + 1e-8

        for c in [0, 1]:
            acc = torch.zeros(seg_hi - seg_lo, device=client_full_protos[0][c].device)
            for i, w in zip(eligible, weights):
                acc += (w / total_w) * client_full_protos[i][c][seg_lo:seg_hi]
            seg_out[c].append(acc)

    return {c: torch.cat(seg_out[c], dim=0) for c in [0, 1]}


def uniform_dimwise_agg(client_full_protos: list, client_tier_dims: list,
                         client_sizes: list, tier_dims=(8, 16, 32, 64)) -> dict:
    """Ablation arm 'Uniform-Agg' (NB3)."""
    boundaries = [0] + list(tier_dims)
    n = len(client_full_protos)
    seg_out = {0: [], 1: []}
    for seg_lo, seg_hi in zip(boundaries[:-1], boundaries[1:]):
        eligible = [i for i in range(n) if client_tier_dims[i] >= seg_hi]
        if not eligible:
            max_dim = max(client_tier_dims)
            eligible = [i for i in range(n) if client_tier_dims[i] == max_dim]
        total_sz = sum(client_sizes[i] for i in eligible)
        for c in [0, 1]:
            acc = torch.zeros(seg_hi - seg_lo, device=client_full_protos[0][c].device)
            for i in eligible:
                acc += (client_sizes[i] / total_sz) * client_full_protos[i][c][seg_lo:seg_hi]
            seg_out[c].append(acc)
    return {c: torch.cat(seg_out[c], dim=0) for c in [0, 1]}
