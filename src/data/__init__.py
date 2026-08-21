"""Data subpackage."""

from src.data.elliptic import load_elliptic
from src.data.splits import temporal_federated_split, make_mask
from src.data.graph_utils import get_local_edge_index, get_inductive_edge_index
