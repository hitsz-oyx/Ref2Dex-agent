"""WAM-v4 interaction-field contract.

The data reader supplies fixed object interaction queries and voxel-derived
labels. The model predicts contact confidence and semantic hand-part
probabilities; query position and normal are input attributes rather than
redundant regression channels.
"""
import torch


C_NAMES = (
    "contact_confidence",
    "palm", "thumb", "index", "middle", "ring", "little",
)
C_DIM = len(C_NAMES)
C_HORIZON = 24
C_ANCHORS = C_HORIZON  # compatibility alias for older callers


def interaction_target(batch):
    """Read the precomputed target-object voxel field from a v4 batch."""
    if "interaction_target" not in batch or "interaction_valid" not in batch:
        raise ValueError("v4 batches must carry voxel-derived interaction_target and interaction_valid")
    target = batch["interaction_target"].float()
    valid = batch["interaction_valid"].bool()
    if target.shape[-1] != C_DIM or valid.shape != target.shape[:-1]:
        raise ValueError("interaction target must be [B,O,24,128,7] with matching mask")
    return target, valid


def interaction_scales(config=None, device=None, dtype=torch.float32):
    values = (1., 1., 1., 1., 1., 1., 1.)
    if config is not None:
        values = tuple(config.get("interaction_scales", values))
    if len(values) != C_DIM or any(v <= 0 for v in values):
        raise ValueError("interaction_scales must contain seven positive values")
    return torch.tensor(values, device=device, dtype=dtype)
