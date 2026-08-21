"""
Loss functions used across all experiments.

Includes classification losses (weighted CE, focal), contrastive losses (InfoNCE,
label-aware SupCon), and prototype-based losses (prototype SupCon, multi-budget).
"""

import torch
import torch.nn.functional as F
import numpy as np

from src.config import ExperimentConfig


# ── Classification Losses ─────────────────────────────────────────────

def weighted_ce(logits, labels, device):
    n_classes = logits.shape[1]
    counts    = torch.bincount(labels, minlength=n_classes).float().clamp(min=1.0)
    weight    = labels.shape[0] / (n_classes * counts)
    weight    = weight.clamp(0.1, 10.0)
    return F.cross_entropy(logits, labels, weight=weight.to(device))


def focal_loss(logits, labels, device, gamma=1.0):
    """Focal loss (Lin et al. 2017) with inverse-frequency class weights."""
    n_classes = logits.shape[1]
    counts    = torch.bincount(labels, minlength=n_classes).float().clamp(1.0)
    alpha_cls = (labels.shape[0] / (n_classes * counts)).clamp(0.1, 10.0)
    alpha_t   = alpha_cls[labels]
    ce  = F.cross_entropy(logits, labels, reduction='none')
    pt  = torch.exp(-ce).clamp(1e-7, 1.0 - 1e-7)
    fl  = alpha_t * (1.0 - pt) ** gamma * ce
    return fl.mean()


def supervised_loss(logits, labels, device, cfg=None, gamma=1.0):
    """Dispatch to focal or weighted CE based on config."""
    if cfg is not None:
        if cfg.use_focal_loss:
            return focal_loss(logits, labels, device, gamma=cfg.focal_gamma)
        return weighted_ce(logits, labels, device)
    # Fallback: use focal loss directly with provided gamma
    return focal_loss(logits, labels, device, gamma=gamma)


# ── Contrastive Losses ────────────────────────────────────────────────

def infonce_loss(z_proj, edge_index, local_mask, device, cfg: ExperimentConfig):
    z_norm    = F.normalize(z_proj, dim=1)
    feat_mask = torch.rand(z_proj.shape[1], device=device) > cfg.feat_drop
    z_aug     = F.normalize(z_norm * feat_mask.float(), dim=1)
    src, dst = edge_index
    keep     = torch.rand(src.shape[0], device=device) > cfg.edge_drop
    src_k, dst_k = src[keep], dst[keep]
    if src_k.numel() == 0:
        return torch.tensor(0.0, device=device)
    pos_sim = (z_norm[src_k] * z_aug[dst_k]).sum(1) / cfg.tau
    anchor_nodes = torch.cat([src_k, dst_k]).unique()
    local_nodes  = torch.where(local_mask.to(device))[0]
    neg_pool     = local_nodes[~torch.isin(local_nodes, anchor_nodes)]
    if neg_pool.numel() < 4:
        perm     = torch.randperm(src_k.shape[0], device=device)
        neg_pool = src_k[perm]
    n_neg   = min(256, neg_pool.shape[0])
    neg_idx = neg_pool[torch.randperm(neg_pool.shape[0], device=device)[:n_neg]]
    neg_sim = torch.mm(z_norm[src_k], z_norm[neg_idx].T) / cfg.tau
    logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
    target = torch.zeros(logits.shape[0], dtype=torch.long, device=device)
    return F.cross_entropy(logits, target)


def label_aware_edge_supcon(z_proj, data, local_mask, device, cfg):
    """Label-aware SupCon: illicit anchors, illicit positives, licit negatives."""
    labels = data.y.to(device)
    lm = local_mask.to(device)
    ill_mask = lm & (labels == 1)
    lic_mask = lm & (labels == 0)
    if ill_mask.sum() < 2 or lic_mask.sum() < 1:
        return torch.tensor(0.0, device=device)
    z_norm = F.normalize(z_proj, dim=1)
    feat_mask = torch.rand(z_proj.shape[1], device=device) > cfg.feat_drop
    z_aug = F.normalize(z_norm * feat_mask.float(), dim=1)
    ill_idx = torch.where(ill_mask)[0]
    lic_idx = torch.where(lic_mask)[0]
    losses = []
    n_anchors = min(64, ill_idx.shape[0])
    perm = torch.randperm(ill_idx.shape[0], device=device)[:n_anchors]
    anchors = ill_idx[perm]
    for a in anchors:
        pos_pool = ill_idx[ill_idx != a]
        if pos_pool.shape[0] == 0:
            continue
        n_pos = min(8, pos_pool.shape[0])
        pos = pos_pool[torch.randperm(pos_pool.shape[0], device=device)[:n_pos]]
        n_neg = min(32, lic_idx.shape[0])
        neg = lic_idx[torch.randperm(lic_idx.shape[0], device=device)[:n_neg]]
        pos_sim = (z_norm[a] * z_aug[pos]).sum(1) / cfg.tau
        neg_sim = (z_norm[a] * z_aug[neg]).sum(1) / cfg.tau
        logits = torch.cat([pos_sim, neg_sim])
        target = torch.zeros(logits.shape[0], device=device)
        target[:pos_sim.shape[0]] = 1.0 / pos_sim.shape[0]
        log_probs = F.log_softmax(logits, dim=0)
        loss = -(target * log_probs).sum()
        losses.append(loss)
    return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=device)


# ── Prototype Losses ──────────────────────────────────────────────────

def prototype_supcon_loss(z, labels, mask, global_protos, device,
                           cfg: ExperimentConfig) -> torch.Tensor:
    labeled = mask.to(device) & (labels.to(device) >= 0)
    if not labeled.any() or global_protos is None:
        return torch.tensor(0.0, device=device)
    z_lab = F.normalize(z[labeled], dim=1)
    y_lab = labels.to(device)[labeled]
    p0 = F.normalize(global_protos[0].detach().unsqueeze(0), dim=1)
    p1 = F.normalize(global_protos[1].detach().unsqueeze(0), dim=1)
    protos_cat   = torch.cat([p0, p1], dim=0)
    proto_logits = torch.mm(z_lab, protos_cat.T) / cfg.tau
    return F.cross_entropy(proto_logits, y_lab)


def multi_budget_proto_loss(z, labels, mask, global_protos_full, device, cfg,
                             budgets=(8, 16, 32, 64), weights=None):
    """Matryoshka-style multi-budget prototype loss (single-pass slicing)."""
    if global_protos_full is None:
        return torch.tensor(0.0, device=device)
    if weights is None:
        weights = [1.0 / len(budgets)] * len(budgets)
    assert len(weights) == len(budgets)
    total = torch.tensor(0.0, device=device)
    for w, d in zip(weights, budgets):
        protos_d = {c: v[:d] for c, v in global_protos_full.items()}
        total = total + w * prototype_supcon_loss(
            z[:, :d], labels, mask, protos_d, device, cfg
        )
    return total
