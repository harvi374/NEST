"""
Federated training loops — run_pgfcl (PGFCL/FedProto) and run_ep_fedproto.
"""

import time
import torch
import numpy as np
from torch_geometric.data import Data

from src.config import ExperimentConfig, config_hash
from src.utils import set_seed
from src.models.full_gat import FullGAT
from src.training.ssl_pretrain import ssl_pretrain_phase
from src.training.supervised import (
    compute_embedding_prototypes,
    chain_prototype_inheritance,
    supervised_round_client,
    supervised_round_client_ep,
    fedper_head_finetune,
)
from src.aggregation.fedavg import fedavg_state_contrib
from src.aggregation.contrib_agg import (
    compute_client_contrib_weights,
    rank_aware_contrib_agg,
)
from src.aggregation.grad_guard import grad_guard_weights
from src.evaluation.metrics import evaluate_tuned
from src.evaluation.calibration import compute_ece
from src.evaluation.saliency import extract_node_saliency


def run_pgfcl(data: Data, clients: list, device: torch.device,
               cfg: ExperimentConfig, seed: int,
               verbose: bool = True, label: str = 'PGFCL',
               proto_dim: int = None):
    """
    PGFCL main training loop (v10).
    - Phase 1: SSL pre-warm
    - Phase 2: Main supervised + prototype federated loop
    - Phase 3: FedPer head fine-tuning
    """
    set_seed(seed)
    global_model = FullGAT(data.num_node_features, cfg).to(device)
    prev_protos  = None
    drift_curve, f1_curve = [], []
    proto_gen_times = []
    best_f1 = 0.
    best_m = {'acc':0.,'f1':0.,'auc':0.,'prec':0.,'rec':0.,'cm':None}

    all_train = torch.zeros(data.num_nodes, dtype=torch.bool)
    all_test  = torch.zeros(data.num_nodes, dtype=torch.bool)
    for c in clients:
        all_train |= c['train_mask']
        all_test  |= c['test_mask']

    global_model = ssl_pretrain_phase(global_model, data, clients, device, cfg, verbose)

    sizes    = [c['n_train'] for c in clients]
    ei_full  = data.edge_index.to(device)
    saliency_history = []

    for rnd in range(cfg.global_rounds):
        lam = cfg.lam_max * min(1.0, rnd / max(cfg.lam_warmup_rounds, 1))
        if verbose:
            print(f'\n  === Round {rnd+1}/{cfg.global_rounds} | lam={lam:.3f} ===')

        x = data.x.to(device)
        client_emb_ps, client_counts = [], []

        _proto_t0 = time.time()
        for j, client in enumerate(clients):
            with torch.no_grad():
                z = global_model.encoder(x, ei_full)
            emb_p, counts = compute_embedding_prototypes(
                z, data, client['train_mask'], device, cfg
            )
            client_emb_ps.append(emb_p)
            client_counts.append(counts)
        proto_gen_times.append(time.time() - _proto_t0)

        client_protos = None
        global_protos = None
        if cfg.use_protos:
            client_protos = chain_prototype_inheritance(
                client_emb_ps, client_counts, clients, rnd, cfg
            )
            global_protos = client_protos[0] if client_protos else None
            if global_protos is not None and prev_protos is not None:
                d0 = (global_protos[0].float() - prev_protos[0].float()).norm().item()
                d1 = (global_protos[1].float() - prev_protos[1].float()).norm().item()
                drift_curve.append({'round': rnd+1, 'drift_licit': d0, 'drift_illicit': d1})
                if verbose:
                    print(f'  Drift licit={d0:.4f} illicit={d1:.4f}')
            if global_protos is not None:
                prev_protos = {c: v.detach().clone() for c, v in global_protos.items()}

        local_models = []
        for i, client in enumerate(clients):
            lm = FullGAT(data.num_node_features, cfg).to(device)
            lm.load_state_dict(global_model.state_dict())
            _cproto = client_protos[i] if (client_protos is not None and lam > 0) else None
            supervised_round_client(lm, data, client, device, cfg,
                global_model=global_model if cfg.use_fedprox else None,
                global_protos=_cproto, lam=lam, proto_dim=proto_dim)
            local_models.append(lm)

        if cfg.use_grad_guard:
            agg_weights, _ = grad_guard_weights(global_model, local_models, sizes, cfg)
        elif cfg.use_contrib_agg and global_protos is not None:
            agg_weights = compute_client_contrib_weights(client_emb_ps, sizes, global_protos, cfg)
        else:
            total = sum(sizes)
            agg_weights = [s / total for s in sizes]

        global_model = fedavg_state_contrib(global_model, local_models, agg_weights)

        m = evaluate_tuned(global_model, data, all_train, all_test, device, cfg,
                            proto_dim=proto_dim)
        f1_curve.append({'round': rnd+1, 'f1': m['f1'], 'auc': m['auc']})
        if m['f1'] > best_f1:
            best_f1, best_m = m['f1'], m
        if verbose:
            print(f'  Global | F1={m["f1"]:.4f} | AUC={m["auc"]:.4f} | '
                  f'Prec={m["prec"]:.4f} | Rec={m["rec"]:.4f} | thresh={m["thresh"]:.2f}')

        if cfg.use_saliency and (rnd + 1) % 10 == 0:
            sal = extract_node_saliency(global_model, data, all_test, device, cfg)
            saliency_history.append({'round': rnd+1, **sal})
            if verbose:
                n_correct = int((sal['true_labels'] == 1).sum())
                print(f'  Saliency top-{cfg.saliency_top_k}: ' +
                      f'{n_correct}/{cfg.saliency_top_k} are truly illicit')

    if cfg.head_finetune_rounds > 0:
        client_results, per_results = fedper_head_finetune(
            global_model, data, clients, device, cfg, verbose=verbose,
            proto_dim=proto_dim
        )
        if per_results['f1'] > best_f1:
            best_f1, best_m = per_results['f1'], per_results
            best_m['fedper_client_results'] = [(None, r[1]) for r in client_results]
        if verbose:
            print(f'  [FedPer] Final agg F1={per_results["f1"]:.4f}')

    if cfg.use_calibration:
        ece, bin_data = compute_ece(global_model, data, all_test, device, cfg,
                                    train_mask=all_train, proto_dim=proto_dim)
        best_m['ece']      = ece
        best_m['ece_bins'] = bin_data
        if verbose:
            print(f'  ECE = {ece:.4f}')

    best_m['proto_gen_time_s'] = float(np.mean(proto_gen_times)) if proto_gen_times else 0.0
    best_m['proto_dim'] = proto_dim if proto_dim is not None else cfg.emb_dim
    best_m['config_hash'] = config_hash(cfg)
    best_m['f1_curve'] = f1_curve

    print(f'\n[{label}] Best F1={best_m["f1"]:.4f} | AUC={best_m["auc"]:.4f} | ' +
          f'hash={best_m["config_hash"]}')
    return best_m, drift_curve, f1_curve, saliency_history
