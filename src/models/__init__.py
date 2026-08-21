"""Models subpackage — SAGEGATEncoder, ClassHead, FullGAT, SAGEModel."""

from src.models.encoder import SAGEGATEncoder
from src.models.head import ClassHead
from src.models.full_gat import FullGAT
from src.models.sage_baseline import SAGEModel

# Backward-compatible alias used in some notebooks
GATEncoder = SAGEGATEncoder
