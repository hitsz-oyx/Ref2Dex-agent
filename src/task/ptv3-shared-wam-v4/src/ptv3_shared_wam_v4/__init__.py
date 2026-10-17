"""Goal-conditioned Shared-WAM v4 with an object-centric interaction head."""
from .model import SharedWAMV4
from .interaction import C_NAMES, C_DIM, interaction_target

__all__ = ["SharedWAMV4", "C_NAMES", "C_DIM", "interaction_target"]
