"""Goal-conditioned Shared-WAM v4 with a dynamic interaction graph."""
from .model import SharedWAMV4
from .interaction import C_NAMES, C_DIM, C_HORIZON, interaction_target

__all__ = ["SharedWAMV4", "C_NAMES", "C_DIM", "C_HORIZON", "interaction_target"]
