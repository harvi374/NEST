"""
GradGuard — Gradient anomaly detection.
"""

import torch
from src.config import ExperimentConfig


def compute_update_directions(global_model, local_models):
    """Compute parameter updates for each client."""
    updates = []
    for lm in local_models:
        u = []
        for gp, lp in zip(global_model.parameters(), lm.parameters()):
            u.append((lp.data - gp.data).view(-1))
        updates.append(torch.cat(u))
    return updates


def grad_guard_weights(global_model, local_models, sizes, cfg: ExperimentConfig):
    """GradGuard: compute cosine similarities between client updates and the
    size-weighted average update. Clients below anomaly_thresh are excluded."""
    updates = compute_update_directions(global_model, local_models)
    total = sum(sizes)
    weights = [s / total for s in sizes]

    avg_u = torch.zeros_like(updates[0])
    for w, u in zip(weights, updates):
        avg_u += w * u

    if avg_u.norm() < 1e-8:
        return weights, []

    sims = []
    for u in updates:
        if u.norm() < 1e-8:
            sims.append(1.0)
        else:
            sim = torch.cosine_similarity(u.unsqueeze(0), avg_u.unsqueeze(0)).item()
            sims.append(sim)

    safe = [s >= cfg.anomaly_thresh for s in sims]
    if not any(safe):
        return weights, sims

    safe_sizes = [sz if is_safe else 0 for sz, is_safe in zip(sizes, safe)]
    safe_total = sum(safe_sizes)
    safe_weights = [sz / safe_total for sz in safe_sizes]

    return safe_weights, sims
