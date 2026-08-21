"""Training subpackage — losses, SSL, supervised training, federated loops."""

from src.training.losses import (
    weighted_ce,
    focal_loss,
    supervised_loss,
    infonce_loss,
    label_aware_edge_supcon,
    prototype_supcon_loss,
    multi_budget_proto_loss,
)
