"""
ExperimentConfig — single source of truth for all hyperparameters.

Previously duplicated 6+ times across notebooks. Any notebook or script
should import this instead of re-defining its own copy.
"""

import hashlib
from dataclasses import dataclass, asdict


@dataclass
class ExperimentConfig:
    # Architecture
    hidden:       int   = 128
    emb_dim:      int   = 64
    heads:        int   = 4
    dropout:      float = 0.4
    head_dropout: float = 0.3

    # Federation
    n_clients:  int   = 4
    seeds:      tuple = (42, 123, 7, 456, 789)
    test_ratio: float = 0.3

    # Feature dimensionality flag
    use_dev_features: bool = True  # True=330-dim (raw+dev), False=165-dim (raw only)

    # Training schedule
    global_rounds:        int   = 100
    head_finetune_rounds: int   = 20
    ssl_pretrain_rounds:  int   = 6
    ssl_epochs:           int   = 20
    sup_epochs:           int   = 25
    lr:                   float = 0.005
    lr_min:               float = 0.0005

    # FedProx
    mu_encoder: float = 0.01
    mu_head:    float = 0.0

    # SSL — used ONLY during the pre-warm phase, never in the main training loop
    tau:            float = 0.7
    ssl_edge_floor: int   = 15
    feat_drop:      float = 0.3
    edge_drop:      float = 0.2

    # Prototype aggregation
    lam_max:              float = 0.6
    lam_warmup_rounds:    int   = 2
    ema_momentum:         float = 0.85
    raw_blend:            float = 0.2
    use_degree_weighting: bool  = True

    # Ablation flags
    use_ssl:     bool = True
    use_protos:  bool = True
    use_fedprox: bool = True

    # Contribution flags
    use_contrib_agg:  bool  = True
    contrib_floor:    float = 0.1
    use_grad_guard:   bool  = True
    anomaly_thresh:   float = -0.1
    use_saliency:     bool  = True
    saliency_top_k:   int   = 20
    use_calibration:  bool  = True
    ece_bins:         int   = 15

    # Focal loss
    use_focal_loss:   bool  = True
    focal_gamma:      float = 1.0

    # Label propagation (disabled by default — ablation showed it hurts)
    use_label_prop:   bool  = False
    lp_alpha:         float = 0.9
    lp_steps:         int   = 1

    # FedPer: local-only head fine-tuning, NO head averaging
    use_fedper:       bool  = True

    cosine_T0:        int   = 50

    # Baselines
    baseline_rounds:       int = 50
    baseline_local_epochs: int = 25
    centralized_epochs:    int = 150


def config_hash(cfg: ExperimentConfig) -> str:
    """Short deterministic hash of config — used to detect checkpoint key collisions."""
    d = {k: v for k, v in asdict(cfg).items() if k != 'seeds'}
    raw = str(sorted(d.items())).encode()
    return hashlib.md5(raw).hexdigest()[:8]
