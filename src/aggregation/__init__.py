"""Aggregation subpackage — FedAvg, ContribAgg, GradGuard."""

from src.aggregation.fedavg import (
    fedavg_state_selective,
    fedavg_state_contrib,
    fedavg_encoder_only,
)
from src.aggregation.contrib_agg import (
    compute_client_contrib_weights,
    rank_aware_contrib_agg,
    uniform_dimwise_agg,
)
from src.aggregation.grad_guard import (
    compute_update_directions,
    grad_guard_weights,
)
