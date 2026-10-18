"""WAM-v4 target-object data views for native and main3 sources.

The source readers remain frozen.  This module creates a derived, target-only
view with one explicit contract: 512 surface points with normals, 32 flow
queries, and 128 interaction queries.  It never edits source payloads.
"""
from __future__ import annotations

from dataclasses import asdict

import numpy as np
import torch
from torch.utils.data import Dataset

from .v4_interaction import V4SamplingPolicy, fixed_subset, resampled_subset
from .v4_windows import v4_pose_mesh_sample


def _goal_windows_class():
    """Return an importable class so spawn workers can pickle the dataset."""
    from goal_point_wam.data import GoalWindows
    return GoalWindows


def _transform(points, pose):
    return points @ pose[:3, :3].T + pose[:3, 3]


def _normalise(normals):
    normals = np.asarray(normals, dtype=np.float32)
    if normals.ndim != 2 or normals.shape[-1] != 3 or not np.isfinite(normals).all():
        raise ValueError("finite xyz normals required")
    return normals / np.linalg.norm(normals, axis=-1, keepdims=True).clip(1e-8)


def _main3_target_sample(sample, normals, policy=V4SamplingPolicy()):
    """Convert one existing GoalWindows sample to target-object v4 tensors."""
    points = np.asarray(sample["points"], dtype=np.float32)
    effects = np.asarray(sample["effect"], dtype=np.float32)
    features = np.asarray(sample["features"], dtype=np.float32)
    scene_object = np.asarray(sample["scene_object"])
    target_slots = np.flatnonzero(np.asarray(sample["object_features"])[:, 13] > .5)
    if len(target_slots) != 1:
        raise ValueError("main3 sample must identify exactly one target object")
    target = int(target_slots[0])
    source_surface = points[target]
    if source_surface.shape != (512, 3):
        raise ValueError("main3 source must expose exactly 512 canonical object points")
    normals = _normalise(normals)
    if normals.shape != source_surface.shape:
        raise ValueError("main3 canonical normals must align with 512 points")
    surface, surface_ids = resampled_subset(source_surface, policy.surface_points, name="main3 target surface")
    surface_normals = normals[surface_ids]
    flow, flow_ids = resampled_subset(source_surface, policy.flow_points, name="main3 flow queries")
    interaction, interaction_ids = resampled_subset(source_surface, policy.interaction_points,
                                                     name="main3 interaction queries")
    interaction_normals = normals[interaction_ids]

    target_rows = np.flatnonzero(scene_object == target)
    if len(target_rows) != 512:
        raise ValueError("main3 target scene geometry must contain all 512 points")
    target_features = features[target_rows].copy()
    target_xyz = np.asarray(sample["xyz"], dtype=np.float32)[target_rows]
    if not np.isfinite(target_xyz).all() or not np.isfinite(target_features).all():
        raise ValueError("nonfinite main3 target geometry")
    # The target object is the current anchor in the source contract.  Keep
    # source-derived motion features, but bind all v4 geometry to canonical
    # point identity and the canonical normal artifact.
    target_xyz = source_surface.copy()
    target_features[:, :3] = normals
    target_features[:, 14:] = 0.
    future = np.einsum("tij,pj->tpi", effects[target, :, :3, :3], flow)
    future += effects[target, :, None, :3, 3]
    flow_target = future - flow[None]
    valid = np.isfinite(future).all(-1)
    hand = np.asarray(sample["start_hand"], dtype=np.float32)
    hand_future = hand[None] + np.asarray(sample["trajectory_target"], dtype=np.float32)
    hand_valid = np.asarray(sample["trajectory_valid"], dtype=bool)
    interaction_future = np.einsum("tij,qj->tqi", effects[target, :, :3, :3], interaction)
    interaction_future += effects[target, :, None, :3, 3]
    from .v4_interaction import classify_window, interaction_field
    interaction_target, interaction_valid = interaction_field(
        hand_future, hand_valid, interaction_future, policy)
    flow_valid = np.broadcast_to(valid, (24, policy.flow_points)).copy()
    kind = classify_window(hand_future, hand_valid, interaction_future, interaction_target, policy)
    return dict(
        schema="ref2dex.wam-v4-target-object.v3", source_id=np.int64(sample["source_id"]),
        sample_id=np.asarray(sample["sample_id"], np.int64), target_object_only=np.array(True),
        xyz=target_xyz.astype(np.float32), features=target_features.astype(np.float32),
        scene_object=np.zeros(policy.surface_points, np.int64),
        point_valid=np.ones(policy.surface_points, bool), object_valid=np.ones(1, bool),
        scene_points=flow[None].astype(np.float32), scene_target=flow_target[None].astype(np.float32),
        scene_valid=flow_valid[None], object_surface_points=surface[None].astype(np.float32),
        object_surface_normals=surface_normals[None].astype(np.float32),
        surface_valid=np.ones((1, policy.surface_points), bool),
        interaction_points=interaction[None].astype(np.float32),
        interaction_normals=interaction_normals[None].astype(np.float32),
        interaction_target=interaction_target[None].astype(np.float32),
        interaction_valid=interaction_valid[None],
        start_hand=np.asarray(sample["start_hand"], np.float32),
        start_hand_history=np.asarray(sample["start_hand_history"], np.float32),
        start_hand_valid=np.asarray(sample["start_hand_valid"], bool),
        trajectory_target=np.asarray(sample["trajectory_target"], np.float32),
        trajectory_valid=hand_valid,
        start_object_points=flow.astype(np.float32),
        goal_object_points=(flow + flow_target[-1]).astype(np.float32),
        object_goal_valid=flow_valid[-1].copy(), goal_object_slot=np.int64(0),
        target_surface_count=np.int64(policy.surface_points), interaction_kind=np.array(kind),
        provenance_profile=np.array("main3-canonical512-normals.v1"),
    )


class V4NativeWindows(Dataset):
    def __init__(self, root, split, *, engineering=True, policy=V4SamplingPolicy(), surface_geometry_manifest=None):
        from .clean_data import CleanPoseMeshWindows
        self.base = CleanPoseMeshWindows(root, split, engineering=engineering, cache_size=8)
        if surface_geometry_manifest:
            from .dense_surface import apply_surface_overlay
            apply_surface_overlay(self.base.geometry, surface_geometry_manifest)
        # Fail at construction, before GPU allocation, if any target geometry
        # would only meet the 512 tensor shape by repeating existing points.
        target_keys = {row['objects'][row['target_slot']] for row in self.base.meta['sequences']
                       if row['split'] == split and row.get('clock_verified') is True}
        for key in target_keys:
            resampled_subset(self.base.geometry[key]['surface'], policy.surface_points, name=key)
        self.policy = policy

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        window = self.base[index]
        row = self.base.meta['sequences'][window.provenance.row]
        sample = v4_pose_mesh_sample(
            self.base._sequence(window.provenance.row),
            [self.base.geometry[k] for k in row['objects']],
            window.provenance.tick, window.provenance.target_slot,
            window.provenance.source_id,
            [window.provenance.source_id, window.provenance.row, window.provenance.tick],
            window.provenance.profile, self.policy)
        sample["provenance"] = asdict(window.provenance)
        return sample


class V4Main3Windows(Dataset):
    def __init__(self, root, split, *, point_count=32, policy=V4SamplingPolicy()):
        from oakink_wm.multisource import MixedWindows
        GoalWindows = _goal_windows_class()
        self.base = MixedWindows(root, split)
        self.goals = GoalWindows(self.base, "object_points", point_count)
        self.policy = policy
        self._canonical_normals = {}

    def __len__(self):
        return len(self.base)

    def _canonical_target(self, index, sample):
        source_id = int(sample["source_id"])
        source = self.base.sources[source_id]
        local_index = int(index - self.base.offsets[source_id])
        sequence_id, anchor, tick, _ = map(int, source.rows[local_index])
        sequence = source.sequence(source.sequences[sequence_id])
        from oakink_wm.data import local_objects
        selected = local_objects(sequence["poses"][tick], sequence["pose_valid"][tick], anchor,
                                 sequence["centers"])
        target_slot = int(np.flatnonzero(selected == anchor)[0])
        name = sequence["objects"][anchor]
        cache_key = (source_id, name)
        cached = self._canonical_normals.get(cache_key)
        if cached is None:
            canonical = source.canonical(name)
            if len(canonical) == 4:
                points, normals, _, _ = canonical
            else:
                points, normals = canonical[:2]
            cached = (np.asarray(points, dtype=np.float32), _normalise(normals))
            self._canonical_normals[cache_key] = cached
        points, normals = cached
        if not np.allclose(points, sample["points"][target_slot], atol=1e-6, rtol=1e-5):
            raise ValueError("main3 canonical identity differs from sampled object points")
        return normals

    def __getitem__(self, index):
        sample = self.goals[index]
        normals = self._canonical_target(index, sample)
        return _main3_target_sample(sample, normals, self.policy)


def v4_collate(samples):
    if not samples:
        raise ValueError("empty v4 batch")
    fixed = {
        "xyz": (3, torch.float32), "features": (18, torch.float32),
        "scene_object": (None, torch.long), "point_valid": (None, torch.bool),
        "object_valid": (None, torch.bool), "scene_points": (None, torch.float32),
        "scene_target": (None, torch.float32), "scene_valid": (None, torch.bool),
        "object_surface_points": (None, torch.float32), "object_surface_normals": (None, torch.float32),
        "surface_valid": (None, torch.bool), "interaction_points": (None, torch.float32),
        "interaction_normals": (None, torch.float32), "interaction_target": (None, torch.float32),
        "interaction_valid": (None, torch.bool),
    }
    out = {}
    for key, (_, dtype) in fixed.items():
        values = [torch.as_tensor(s[key], dtype=dtype) for s in samples]
        out[key] = torch.stack(values)
    for key, dtype in (("start_hand", torch.float32), ("start_hand_history", torch.float32),
                       ("start_hand_valid", torch.bool), ("trajectory_target", torch.float32),
                       ("trajectory_valid", torch.bool), ("start_object_points", torch.float32),
                       ("goal_object_points", torch.float32), ("object_goal_valid", torch.bool),
                       ("goal_object_slot", torch.long), ("source_id", torch.long),
                       ("sample_id", torch.long)):
        out[key] = torch.as_tensor(np.stack([s[key] for s in samples]), dtype=dtype)
    out["goal_object_delta"] = out["goal_object_points"] - out["start_object_points"]
    return out


__all__ = ["V4NativeWindows", "V4Main3Windows", "v4_collate", "_main3_target_sample"]
