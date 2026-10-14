"""Shared TRAIN-only per-horizon XYZ displacement moments, always in metres."""
import numpy as np
import torch
from torch import nn
from .paths import STATS_SCHEMA


class Moments:
    def __init__(self):
        self.count = np.zeros((24, 1), dtype=np.int64)
        self.mean = np.zeros((24, 3), dtype=np.float64)
        self.m2 = np.zeros((24, 3), dtype=np.float64)

    def update(self, values, valid):
        # Caller puts time first; invalid NaNs never enter the statistics.
        values = np.asarray(values, dtype=np.float64).reshape(24, -1, 3)
        valid = np.asarray(valid, dtype=bool).reshape(24, -1)
        if not np.isfinite(values[valid]).all():
            raise ValueError("nonfinite valid displacement")
        for h in range(24):
            x = values[h, valid[h]]
            if not len(x):
                continue
            n, mean = len(x), x.mean(0)
            total = self.count[h] + n
            delta = mean - self.mean[h]
            self.m2[h] += ((x - mean)**2).sum(0) + delta**2 * self.count[h] * n / total
            self.mean[h] += delta * n / total
            self.count[h] = total

    def result(self, floor_m):
        if (self.count == 0).any() or not np.isfinite(floor_m) or floor_m <= 0:
            raise ValueError("all horizons need observations and a positive std floor")
        std = np.sqrt(self.m2 / self.count)
        return dict(mean=self.mean.tolist(), std=np.maximum(std, floor_m).tolist(),
                    raw_std=std.tolist(), count=self.count[:, 0].tolist())


class HorizonNormalizer(nn.Module):
    def __init__(self, stats, enabled=True, fallback_scale_m=.05):
        super().__init__()
        if stats.get("schema") != STATS_SCHEMA or stats.get("split") != "train":
            raise ValueError("v2 TRAIN-only horizon statistics required")
        if stats.get("sources") != ["oakink2", "grab", "arctic"]:
            raise ValueError("one shared three-domain statistics artifact required")
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
