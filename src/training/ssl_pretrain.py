"""
SSL pre-training functions — contrastive pre-warm phase.

SSL runs ONLY during the pre-warm phase, never inside the per-round loop.
"""

import torch
from torch.optim import Adam
from torch_geometric.data import Data

from src.config import ExperimentConfig
from src.models.encoder import SAGEGATEncoder
from src.models.full_gat import FullGAT
from src.training.losses import label_aware_edge_supcon
from src.aggregation.fedavg import fedavg_encoder_only


def ssl_pretrain_client(encoder, data: Data, client: dict,
                         device: torch.device, cfg: ExperimentConfig,
                         opt=None):
    """Label-aware SupCon on one client encoder.
    Called ONLY during ssl_pretrain_phase, never inside the main training loop."""
    encoder.train()
    if opt is None:
        opt = Adam(encoder.parameters(), lr=cfg.lr)
    x       = data.x.to(device)
    lm      = client['train_mask'].to(device)
    ei_full = data.edge_index.to(device)
    for _ in range(cfg.ssl_epochs):
        opt.zero_grad()
        _, z_proj = encoder.encode_with_proj(x, ei_full)
        loss = label_aware_edge_supcon(z_proj, data, lm, device, cfg)
        loss.backward()
        opt.step()


def ssl_pretrain_phase(global_model: FullGAT, data: Data, clients: list,
                        device: torch.device, cfg: ExperimentConfig,
                        verbose: bool = True) -> FullGAT:
    """SSL pre-warm phase. Runs ONLY here, never inside the per-round loop."""
    if not cfg.use_ssl or cfg.ssl_pretrain_rounds == 0:
        return global_model
    if verbose:
        print(f'  [SSL pre-warm] {cfg.ssl_pretrain_rounds} rounds...')
    sizes = [c['n_train'] for c in clients]
    client_ssl_opt_states = [None] * len(clients)
    for pre_rnd in range(cfg.ssl_pretrain_rounds):
        local_encoders = []
        for j, client in enumerate(clients):
            enc = SAGEGATEncoder(data.num_node_features, cfg).to(device)
            enc.load_state_dict(global_model.encoder.state_dict())
            opt = Adam(enc.parameters(), lr=cfg.lr)
            if client_ssl_opt_states[j] is not None:
                saved = client_ssl_opt_states[j]
                new_state = opt.state_dict()
                for new_pid in saved['state']:
                    if new_pid in new_state.get('state', {}):
                        new_state['state'][new_pid] = saved['state'][new_pid]
                try:
                    opt.load_state_dict(new_state)
                except (ValueError, KeyError):
                    pass
            ssl_pretrain_client(enc, data, client, device, cfg, opt=opt)
            client_ssl_opt_states[j] = opt.state_dict()
            local_encoders.append(enc)
        global_model = fedavg_encoder_only(global_model, local_encoders, sizes)
        if verbose:
            print(f'    Pre-warm {pre_rnd+1}/{cfg.ssl_pretrain_rounds} done')
    return global_model
