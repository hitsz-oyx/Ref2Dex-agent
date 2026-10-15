"""RL660 robot windows with an explicit q_ref/q_pd/q_real data contract.

The v2 adapter deliberately trains on the feedback-compensated PD target
(`q_pd`).  `q_ref` is loaded from the read-only geometric DExplore source and
`q_real` is loaded from the RL execution, but neither is silently substituted
for the other.  Keeping all three tensors in each sample makes the choice
auditable and leaves a future retargeter free to consume q_ref explicitly.
"""
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
import torch

# Importing paths installs the legacy/v1 dependency roots before any v1
# transport module is resolved.
from . import paths as _paths
from ptv3_shared_wam.data import shared_collate
from ptv3_shared_wam.rl_robot import RLRobotWindows, CORPUS_SCHEMA
from ptv3_shared_wam.loading import forecasting_collate


class V2RLRobotWindows(RLRobotWindows):
    """RL660 windows with immutable geometric reference-q provenance."""

    def __init__(self, source, split, point_count=32, action_representation=None,
                 encoder_workspace=None):
        if action_representation is None:
            from ptv3_shared_wam.action_targets import NATIVE
            action_representation = NATIVE
        super().__init__(source, split, point_count, action_representation)
        membership = Path(self.manifest["source_membership"]).resolve()
        source_manifest = json.loads((membership / "manifest.json").read_text())
        if source_manifest.get("source_geometric_root") is None:
            raise ValueError("RL660 source membership has no geometric q_ref root")
        self.geometric_root = Path(source_manifest["source_geometric_root"]).resolve()
        if not self.geometric_root.is_dir():
            raise FileNotFoundError(self.geometric_root)
        self.workspace_admission = None
        if encoder_workspace is not None:
            from .robot_workspace import SCHEMA, restrict_observed_workspace
            if encoder_workspace.get('schema') != SCHEMA:
                raise ValueError('explicit versioned encoder workspace admission required')
            self.workspace_admission = restrict_observed_workspace(
                self, encoder_workspace['origin_m'], encoder_workspace['voxel_m'])

    @lru_cache(maxsize=16)
    def reference_sequence(self, seq):
        record = self.records[seq]
        path = self.geometric_root / record["sequence"] / "interaction_hand_inspire.pt"
        raw = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(raw, torch.Tensor):
            raw = raw.numpy()
        raw = np.asarray(raw)
        if (raw.ndim != 2 or raw.shape[0] != record["frames"] or raw.shape[1] < 391):
            raise ValueError("q_ref source length/raw schema differs from RL660 record")
        q_ref = raw[:, 373:391].astype(np.float32, copy=True)
        if not np.isfinite(q_ref).all():
            raise ValueError("nonfinite geometric q_ref")
        return q_ref

    def __getitem__(self, index):
        tick, seq = self.keys[index]
        sample = super().__getitem__(index)
        data, _ = self.sequence(seq)
        q_ref = self.reference_sequence(seq)
        # q_pd[t] is the feedback-compensated controller target for command t;
        # q_real[t] is measured state at that command; q_ref_next is the
        # geometric nominal pose at the state reached by command t.
        sample.update(
            q_ref_next=q_ref[tick + 1:tick + 9].copy(),
            q_pd=data["pd_targets"][tick:tick + 8].astype(np.float32, copy=True),
            q_real=data["q"][tick:tick + 8].astype(np.float32, copy=True),
            q_real_next=data["q"][tick + 1:tick + 9].astype(np.float32, copy=True),
        )
        return sample


def robot_collate(samples):
    """Existing shared collate plus the explicit three-way q contract."""
    batch = shared_collate(samples)
    for key in ("q_ref_next", "q_pd", "q_real", "q_real_next"):
        if key not in samples[0]:
            raise ValueError("missing explicit q contract field: " + key)
        batch[key] = torch.as_tensor(np.stack([sample[key] for sample in samples]))
    # Keep the same input pruning as the v1 forecasting loader.  This removes
    # large raw arrays while retaining action_target/pd_target/future_q.
    for key in ("action", "action_valid", "effect", "points", "radius", "object_features"):
        batch.pop(key, None)
    return batch


__all__ = ["V2RLRobotWindows", "robot_collate"]
