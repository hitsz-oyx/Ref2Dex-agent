"""Matched wrist MLPs with an optional frozen shared-feature condition."""
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

FEATURE_SCHEMA = 'ref2dex.wrist-shared-cache.v1'
MODEL_SCHEMA = 'ref2dex.wrist-shared-mlp.v1'


def write_json(path, value):
    path = Path(path)
    part = path.with_suffix(path.suffix + '.part')
    part.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    part.replace(path)


def pool_shared(tokens, valid, scene_count):
    """Keep separate scene, hand and goal summaries; exclude padded payloads."""
    if tokens.shape[:2] != valid.shape or tokens.shape[1] <= scene_count + 22:
        raise ValueError('Scene/22 hand/goal token boundaries required')
    groups = []
    for lo, hi in ((0, scene_count), (scene_count, scene_count + 22),
                   (scene_count + 22, tokens.shape[1])):
        mask = valid[:, lo:hi].bool()
        if not mask.any(1).all():
            raise ValueError('Nonempty scene, hand and goal required')
        x = torch.where(mask[..., None], tokens[:, lo:hi], 0.).float()
        groups.append(x.sum(1) / mask.sum(1, keepdim=True))
    pooled = torch.cat(groups, -1)
    if not torch.isfinite(pooled).all():
        raise FloatingPointError('Nonfinite valid shared features')
    return pooled


class WristSharedMLP(nn.Module):
    """The original811D MLP is intact; z adds only to its first preactivation."""
    def __init__(self, shared_dim=2304):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(811, 1024), nn.ReLU(),
            nn.Linear(1024, 1024), nn.ReLU(), nn.Linear(1024, 1024), nn.ReLU(),
            nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, 6))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        # This is equivalent to zero-padding the first layer's input weights.
        # Preserve the original network's initialization and raw-input path.
        with torch.random.fork_rng(devices=[]):
            self.shared_input = nn.Linear(shared_dim, 1024, bias=False)
        nn.init.zeros_(self.shared_input.weight)

    def forward(self, x, shared):
        if x.ndim != 2 or x.shape[1] != 811 or shared.shape != (len(x), self.shared_input.in_features):
            raise ValueError('Matched811D input and shared feature rows required')
        value = self.net[0](x) + self.shared_input(shared)
        for layer in list(self.net)[1:]:
            value = layer(value)
        return value


def phase_draw(ticks, sequence_ids, batch_size, rng):
    """Same half-startup/sequence-uniform sampling as the original wrist MLP."""
    selected = []
    for startup in (True, False):
        pools = [np.flatnonzero((sequence_ids == seq) & ((ticks < 16) == startup))
                 for seq in np.unique(sequence_ids)]
        pools = [p for p in pools if len(p)]
        table = np.full((len(pools), max(map(len, pools))), -1, np.int64)
        for i, rows in enumerate(pools):
            table[i, :len(rows)] = rows
        selected.append((table, np.asarray(list(map(len, pools)))))
    def draw():
        rows = []
        for table, lengths in selected:
            seq = rng.integers(len(table), size=batch_size // 2)
            rows.append(table[seq, (rng.random(batch_size // 2) * lengths[seq]).astype(np.int64)])
        return np.concatenate(rows)
    return draw


def condition_rows(z, mode, seed):
    if mode == 'none':
        return np.zeros_like(z)
    if mode == 'aligned':
        return z.copy()
    if mode == 'shuffled':
        # Caller operates on each split independently; no cross-split access.
        return z[np.random.default_rng(seed).permutation(len(z))].copy()
    raise ValueError(mode)
