"""Frozen WAM source/readout controls on an immutable control-data split."""
import numpy as np
import torch
from torch import nn

from .wrist_shared_probe import pool_shared


class FeatureControl(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        for n, m in zip([1442, 1024, 1024, 1024], [1024, 1024, 1024, 512]):
            layers.extend([nn.Linear(n, m), nn.ReLU()])
        head = nn.Linear(512, 12)
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)
        self.raw = nn.Sequential(*layers, head)
        with torch.random.fork_rng(devices=[]):
            self.feature = nn.Linear(2304, 1024, bias=False)
        nn.init.zeros_(self.feature.weight)

    def forward(self, x, z):
        value = self.raw[0](x) + self.feature(z)
        for layer in list(self.raw)[1:]:
            value = layer(value)
        return value


def readouts(tokens, valid, scene_count):
    mean = pool_shared(tokens, valid, scene_count)
    if not valid[:, scene_count].all():
        raise ValueError('Right wrist token must be observed')
    wrist = torch.cat((mean[:, :768], tokens[:, scene_count].float(), mean[:, 1536:]), -1)
    return dict(mean=mean, wrist=wrist)


def train_moments(values):
    """Only call with seen rows; heldout data must never fit feature statistics."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Finite nonempty training feature matrix required')
    return values.mean(0).astype(np.float32), values.std(0).clip(1e-3).astype(np.float32)


def random_projection(x, seed=180):
    """An equally wide deterministic raw-only placebo, without new information."""
    rng = np.random.default_rng(seed)
    matrix = rng.choice(np.array([-1., 1.], np.float32), size=(x.shape[-1], 2304))
    return matrix / np.sqrt(x.shape[-1])
