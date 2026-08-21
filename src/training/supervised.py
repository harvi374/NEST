"""
Supervised training functions — per-client local training and FedPer head fine-tuning.
"""

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from src.config import ExperimentConfig
from src.models.full_gat import FullGAT
from src.training.losses import supervised_loss, prototype_supcon_loss
from src.data.graph_utils import get_local_edge_index
from src.evaluation.metrics import evaluate_tuned, avg_metrics


def compute_embedding_prototypes(embeddings, data, mask, device, cfg):
    """Compute per-class embedding prototypes for a client."""
    labels  = data.y.to(device)
    labeled = mask.to(device) & (labels >= 0)
    if cfg.use_degree_weighting:
        ei  = data.edge_index.to(device)
        deg = torch.zeros(data.num_nodes, dtype=torch.float, device=device)
        deg.scatter_add_(0, ei[0], torch.ones(ei.size(1), device=device))
        deg = deg + 1.0
    protos, counts = {}, {}
    for c in [0, 1]:
        cm = labeled & (labels == c)
        counts[c] = int(cm.sum())
        if counts[c] == 0:
            protos[c] = torch.zeros(embeddings.shape[1], device=device)
            continue
        emb_c = embeddings[cm]
        if cfg.use_degree_weighting:
            w = deg[cm].unsqueeze(1)
            protos[c] = (emb_c * w).sum(0) / w.sum()
        else:
            protos[c] = emb_c.mean(0)
    return protos, counts


def chain_prototype_inheritance(client_emb_protos, client_cls_counts,
                                  clients, rnd, cfg):
    """Chain prototype inheritance across clients."""
    if rnd == 0 or client_emb_protos is None:
        return None
    n = len(client_emb_protos)
    client_protos = {}
    for i in range(n):
        if i == 0 or client_emb_protos[i-1] is None:
            own = client_emb_protos[i]
            client_protos[i] = {
                c: F.normalize(own[c].unsqueeze(0), dim=1).squeeze(0)
                for c in [0, 1]
            }
        else:
            pred = client_emb_protos[i-1]
            own  = client_emb_protos[i]
            blended = {}
            for c in [0, 1]:
                p = F.normalize(pred[c].unsqueeze(0), dim=1).squeeze(0)
                o = F.normalize(own[c].unsqueeze(0), dim=1).squeeze(0)
                alpha = cfg.ema_momentum if c == 1 else 0.3
                blended[c] = F.normalize(
                    (alpha * p + (1 - alpha) * o).unsqueeze(0), dim=1
                ).squeeze(0)
            client_protos[i] = blended
    return client_protos


def supervised_round_client(model, data, client, device, cfg,
                              global_model=None, global_protos=None, lam=0.0,
                              proto_dim=None):
    """One round of supervised local training for a single client."""
    model.train()
    opt   = Adam(model.parameters(), lr=cfg.lr)
    sched = CosineAnnealingWarmRestarts(opt, T_0=max(1, cfg.sup_epochs // 2), eta_min=cfg.lr_min)
    ei    = get_local_edge_index(data, client['train_mask'], device)
    vmask = client['train_mask'] & (data.y >= 0)
    lbls  = data.y[vmask].to(device)
    enc_ref = (
        {n: p.detach().clone() for n, p in global_model.encoder.named_parameters()}
        if global_model is not None and cfg.use_fedprox else None
    )
    if vmask.sum() < 2:
        return  # skip training — global weights unchanged for this client
    for _ in range(cfg.sup_epochs):
        opt.zero_grad()
        logits, z = model(data.x.to(device), ei, trunc_dim=proto_dim)
        loss = supervised_loss(logits[vmask.to(device)], lbls, device, cfg)
        if cfg.use_fedprox and enc_ref is not None and cfg.mu_encoder > 0:
            enc_prox = sum(
                ((p - enc_ref[n]) ** 2).sum()
                for n, p in model.encoder.named_parameters()
            )
            loss = loss + (cfg.mu_encoder / 2) * enc_prox
        if cfg.use_protos and global_protos is not None and lam > 0:
            if proto_dim is not None:
                _protos = {c: v[:proto_dim] for c, v in global_protos.items()}
                _z = z[:, :proto_dim]
            else:
                _protos, _z = global_protos, z
            proto_loss = prototype_supcon_loss(
                _z, data.y, client['train_mask'], _protos, device, cfg
            )
            loss = loss + lam * proto_loss
        loss.backward()
        opt.step()
        sched.step()


def supervised_round_client_ep(model, data, client, device, cfg,
                                 global_model, global_protos, lam,
                                 client_tier, tier_dims):
    """EP-FedProto per-client supervised training with multi-budget prototype loss."""
    model.train()
    opt   = Adam(model.parameters(), lr=cfg.lr)
    sched = CosineAnnealingWarmRestarts(opt, T_0=max(1, cfg.sup_epochs // 2), eta_min=cfg.lr_min)
    ei    = get_local_edge_index(data, client['train_mask'], device)
    vmask = client['train_mask'] & (data.y >= 0)
    lbls  = data.y[vmask].to(device)
    enc_ref = (
        {n: p.detach().clone() for n, p in global_model.encoder.named_parameters()}
        if global_model is not None and cfg.use_fedprox else None
    ) if global_model is not None else None
    if vmask.sum() < 2:
        return
    own_budgets = tuple(d for d in tier_dims if d <= client_tier)
    for _ in range(cfg.sup_epochs):
        opt.zero_grad()
        logits, z = model(data.x.to(device), ei)
        loss = supervised_loss(logits[vmask.to(device)], lbls, device, cfg)
        if cfg.use_fedprox and enc_ref is not None and cfg.mu_encoder > 0:
            enc_prox = sum(
                ((p - enc_ref[n]) ** 2).sum()
                for n, p in model.encoder.named_parameters()
            )
            loss = loss + (cfg.mu_encoder / 2) * enc_prox
        if cfg.use_protos and global_protos is not None and lam > 0:
            proto_loss = multi_budget_proto_loss(
                z, data.y, client['train_mask'], global_protos,
                device, cfg, budgets=own_budgets
            )
            loss = loss + lam * proto_loss
        loss.backward()
        opt.step()
        sched.step()


def fedper_head_finetune(global_model, data, clients, device, cfg,
                          verbose=True, proto_dim=None):
    """FedPer: each client fine-tunes its own head locally, encoder frozen.
    Heads are NOT averaged back. Returns (client_results, aggregated_metrics)."""
    if not cfg.use_fedper or cfg.head_finetune_rounds == 0:
        all_train = torch.zeros(data.num_nodes, dtype=torch.bool)
        all_test  = torch.zeros(data.num_nodes, dtype=torch.bool)
        for c in clients:
            all_train |= c['train_mask']
            all_test  |= c['test_mask']
        m = evaluate_tuned(global_model, data, all_train, all_test, device, cfg,
                            proto_dim=proto_dim)
        return [(global_model, m)], m

    if verbose:
        print(f'  [FedPer] Local head FT ({cfg.head_finetune_rounds} rounds, encoder frozen)...')

    client_results = []
    for i, client in enumerate(clients):
        lm = FullGAT(data.num_node_features, cfg).to(device)
        lm.load_state_dict(global_model.state_dict())
        for p in lm.encoder.parameters():
            p.requires_grad_(False)
        lm.train()
        opt   = Adam(lm.head.parameters(), lr=cfg.lr_min * 5)
        sched = CosineAnnealingWarmRestarts(opt, T_0=max(1, cfg.head_finetune_rounds // 2), eta_min=cfg.lr_min)
        ei    = get_local_edge_index(data, client['train_mask'], device)
        vmask = client['train_mask'] & (data.y >= 0)
        lbls  = data.y[vmask].to(device)
        for _ in range(cfg.head_finetune_rounds):
            for _ in range(cfg.sup_epochs):
                opt.zero_grad()
                logits, _ = lm(data.x.to(device), ei, trunc_dim=proto_dim)
                supervised_loss(logits[vmask.to(device)], lbls, device, cfg).backward()
                opt.step()
                sched.step()
        for p in lm.encoder.parameters():
            p.requires_grad_(True)
        m = evaluate_tuned(lm, data, client['train_mask'], client['test_mask'], device, cfg,
                            proto_dim=proto_dim)
        if verbose:
            print(f'    Client {i}: F1={m["f1"]:.4f} Prec={m["prec"]:.4f} Rec={m["rec"]:.4f}')
        client_results.append((lm, m))

    agg = avg_metrics([r[1] for r in client_results])
    if verbose:
        print(f'  [FedPer] Agg F1={agg["f1"]:.4f} ± {agg["f1_std"]:.4f}')
    return client_results, agg
