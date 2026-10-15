"""One TRAIN-only physical normalization shared across all nine sources."""
import torch
from torch import nn
from .paths import STATS_SCHEMA, SOURCE_IDS

class HorizonNormalizer(nn.Module):
    def __init__(self, stats, enabled=True, fallback_scale_m=.05):
        super().__init__()
        if stats.get("schema") != STATS_SCHEMA or stats.get("split") != "train":
            raise ValueError("v3 TRAIN-only horizon statistics required")
        if stats.get("source_ids") != list(SOURCE_IDS):
            raise ValueError("one shared nine-domain statistics artifact required")
        for name in ("scene", "hand"):
            mean = torch.tensor(stats[name]["mean"], dtype=torch.float32)
            std = torch.tensor(stats[name]["std"], dtype=torch.float32)
            if mean.shape != (24, 3) or std.shape != (24, 3):
                raise ValueError("horizon statistics must have shape24x3")
            if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
                raise ValueError("finite positive horizon statistics required")
            if not enabled:
                if fallback_scale_m <= 0:
                    raise ValueError("positive fallback scale required")
                mean, std = torch.zeros_like(mean), torch.full_like(std, fallback_scale_m)
            self.register_buffer(name + "_mean", mean)
            self.register_buffer(name + "_std", std)

    def parameters_for(self, name, values):
        # Scene: B,O,T,P,3. Hand: B,T,2,11,3.
        shape = (1, 1, 24, 1, 3) if name == "scene" else (1, 24, 1, 1, 3)
        return (getattr(self, name + "_mean").reshape(shape),
                getattr(self, name + "_std").reshape(shape))

    def normalize(self, name, values):
        mean, std = self.parameters_for(name, values)
        return (values.float() - mean) / std

    def denormalize(self, name, values):
        mean, std = self.parameters_for(name, values)
        return values.float() * std + mean
