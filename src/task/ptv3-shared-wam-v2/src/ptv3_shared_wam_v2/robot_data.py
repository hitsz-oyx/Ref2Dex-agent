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
                 encoder_workspace=None, rollout_admission=None,include_current_observation=False):
        if action_representation is None:
            from ptv3_shared_wam.action_targets import NATIVE
            action_representation = NATIVE
        super().__init__(source, split, point_count, action_representation)
        self.include_current_observation=include_current_observation
        if include_current_observation:
            c=self.manifest.get('observation_contract',{})
            if (not self.manifest.get('capture_current_observation') or c.get('reference_delta')!=0
                    or c.get('dimension')!=721 or c.get('future_reference_inputs_saved') is not False
                    or c.get('live_future_invariance_passed') is not True):
                raise ValueError('Audited current-only delta0 corpus required')
            audit=json.loads((self.root/self.manifest['native_observation_audit']).read_text())
            if audit.get('status')!='PASS' or audit.get('sequences')!=self.manifest['num_sequences']:
                raise ValueError('Full native current observation audit required')
        membership = Path(self.manifest["source_membership"]).resolve()
        source_manifest = json.loads((membership / "manifest.json").read_text())
        if source_manifest.get("source_geometric_root") is None:
            raise ValueError("RL660 source membership has no geometric q_ref root")
        self.geometric_root = Path(source_manifest["source_geometric_root"]).resolve()
        if not self.geometric_root.is_dir():
            raise FileNotFoundError(self.geometric_root)
        self.rollout_admission = None
        if rollout_admission is not None:
            if rollout_admission.get("history") != 4 or rollout_admission.get("horizon") != 24:
                raise ValueError("robot windows require fixed H4/future24 admission")
            if not self.manifest.get("capture_current_observation"):
                raise ValueError("strict v2 robot training requires native supplement rollout")
            from .rollout_admission import restrict_rollout_admission
            self.rollout_admission = restrict_rollout_admission(self, rollout_admission)
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
        if self.include_current_observation:
            obs=data['native_current_obs']
            if (obs.shape!=(len(data['q']),721) or not np.array_equal(data['frame_tick'],np.arange(len(obs)))
                    or not np.isfinite(obs[tick]).all()):
                raise ValueError('Current observation clock/shape differs')
            sample['native_current_obs']=obs[tick].astype(np.float32,copy=True)
        return sample

    def current_observation_statistics(self):
        if self.split!='train' or not self.include_current_observation:
            raise ValueError('Current normalizers require TRAIN delta0 windows')
        total=np.zeros(721,np.float64);squares=total.copy();count=0
        by_sequence={}
        for tick,seq in self.keys:by_sequence.setdefault(seq,[]).append(tick)
        for seq,ticks in by_sequence.items():
            values=self.sequence(seq)[0]['native_current_obs'][ticks].astype(np.float64)
            if not np.isfinite(values).all():raise ValueError('Nonfinite TRAIN current observations')
            total+=values.sum(0);squares+=np.square(values).sum(0);count+=len(values)
        mean=total/count
        return mean.astype(np.float32),np.sqrt(np.maximum(squares/count-mean**2,0)).clip(1e-3).astype(np.float32)


def robot_collate(samples):
    """Existing shared collate plus the explicit three-way q contract."""
    batch = shared_collate(samples)
    if 'native_current_obs' in samples[0]:
        batch['native_current_obs']=torch.as_tensor(np.stack([s['native_current_obs'] for s in samples]))
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
