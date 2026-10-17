"""Target-object-only WAM-v4 sample construction from frozen native windows."""
from __future__ import annotations

import numpy as np

from .v4_interaction import (
    FLOW_POINTS, INTERACTION_POINTS, SURFACE_POINTS, V4SamplingPolicy,
    classify_window, fixed_subset, resampled_subset, interaction_field,
)


def _transform(points, pose):
    return points @ pose[:3, :3].T + pose[:3, 3]


def _normal_transform(normals, pose):
    return normals @ pose[:3, :3].T


def v4_pose_mesh_sample(sequence, geometries, tick, target_slot, source_id, sample_id,
                        profile, policy=V4SamplingPolicy()):
    """Create a target-only 512/32/128 sample without modifying source arrays."""
    if not 3 <= tick < len(sequence['time']) - 24:
        raise ValueError('H4/K24 outside sequence')
    window = slice(tick - 3, tick + 25)
    if (sequence['segment'][window] != sequence['segment'][tick]).any():
        raise ValueError('window crosses reset/gap segment')
    poses, known = sequence['object_poses'], sequence['object_pose_valid']
    if target_slot not in np.flatnonzero(known[tick - 3:tick + 1].all(0)) or not known[tick + 24, target_slot]:
        raise ValueError('observed target and endpoint pose must be known')
    geometry = geometries[target_slot]
    source_surface, source_normals = np.asarray(geometry['surface']), np.asarray(geometry['normals'])
    if (source_surface.ndim != 2 or source_surface.shape[1:] != (3,)
            or source_normals.shape != source_surface.shape):
        raise ValueError('target surface/normal shape mismatch')
    surface, surface_ids = resampled_subset(source_surface, policy.surface_points, name='target surface')
    normals = source_normals[surface_ids].astype(np.float32, copy=True)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True).clip(1e-8)
    flow_source = geometry.get('query', geometry['surface'][
        np.linspace(0, len(geometry['surface']) - 1, FLOW_POINTS, dtype=np.int64)])
    flow = fixed_subset(flow_source, FLOW_POINTS, name='flow queries')
    interaction, interaction_ids = resampled_subset(source_surface, INTERACTION_POINTS, name='interaction queries')
    interaction_normals = source_normals[interaction_ids].astype(np.float32, copy=True)
    interaction_normals /= np.linalg.norm(interaction_normals, axis=-1, keepdims=True).clip(1e-8)

    anchor = np.linalg.inv(poses[tick, target_slot])
    observed = anchor @ poses[tick - 3:tick + 1, target_slot]
    future_known = known[tick + 1:tick + 25, target_slot]
    safe = np.where(future_known[:, None, None], poses[tick + 1:tick + 25, target_slot], np.eye(4))
    relative = anchor @ safe

    raw_hand = sequence['hand'][window]
    hand_valid = sequence['hand_valid'][window]
    hand = np.where(hand_valid[..., None], raw_hand @ anchor[:3, :3].T + anchor[:3, 3], 0.)
    present = hand_valid[:4].all(0)
    if not present.any():
        raise ValueError('no observed native hand point')
    history = np.concatenate(((hand[3] - hand[2]) * 30,
                              (hand[3] - 2 * hand[2] + hand[1]) * 900,
                              hand[3] - hand[0]), -1)
    future_hand_valid = hand_valid[4:] & present[None]
    hand_future = hand[4:]

    surface_current = _transform(surface, observed[-1])
    surface_normals = _normal_transform(normals, observed[-1])
    surface_history = np.stack([_transform(surface, pose) for pose in observed], 0)
    features = np.zeros((SURFACE_POINTS, 18), np.float32)
    features[:, :3] = surface_normals
    features[:, 3:6] = (surface_history[3] - surface_history[2]) * 30
    features[:, 6:9] = (surface_history[3] - 2 * surface_history[2] + surface_history[1]) * 900
    features[:, 9:12] = surface_history[3] - surface_history[0]
    future_flow = np.einsum('tij,pj->tpi', relative[:, :3, :3], flow) + relative[:, None, :3, 3]
    future_interaction = np.einsum('tij,pj->tpi', relative[:, :3, :3], interaction) + relative[:, None, :3, 3]
    interaction_target, interaction_valid = interaction_field(
        hand_future, future_hand_valid, future_interaction, policy)
    flow_start = _transform(flow, observed[-1])
    flow_target = future_flow - flow_start
    flow_valid = np.broadcast_to(future_known[:, None], (24, FLOW_POINTS)).copy()
    kind = classify_window(hand_future, future_hand_valid, future_interaction, interaction_target, policy)
    return dict(
        schema='ref2dex.wam-v4-target-object.v1', source_id=np.int64(source_id),
        sample_id=np.asarray(sample_id, np.int64), source_profile=np.array(profile),
        training_allowed=np.array(False), target_object_only=np.array(True),
        xyz=surface_current.astype(np.float32), features=features,
        scene_object=np.zeros(SURFACE_POINTS, np.int64),
        point_valid=np.ones(SURFACE_POINTS, bool),
        object_valid=np.ones(1, bool),
        scene_points=flow_start[None].astype(np.float32),
        scene_target=flow_target[None].astype(np.float32),
        scene_valid=flow_valid[None],
        object_surface_points=surface_current[None].astype(np.float32),
        object_surface_normals=surface_normals[None].astype(np.float32),
        surface_valid=np.ones((1, SURFACE_POINTS), bool),
        interaction_points=_transform(interaction, observed[-1])[None].astype(np.float32),
        interaction_normals=_normal_transform(interaction_normals, observed[-1])[None].astype(np.float32),
        interaction_target=interaction_target[None].astype(np.float32),
        interaction_valid=interaction_valid[None],
        start_hand=np.where(present[..., None], hand[3], 0.).astype(np.float32),
        start_hand_history=np.where(present[..., None], history, 0.).astype(np.float32),
        start_hand_valid=present,
        trajectory_target=np.where(future_hand_valid[..., None], hand_future - hand[3], 0.).astype(np.float32),
        trajectory_valid=future_hand_valid,
        start_object_points=flow_start.astype(np.float32),
        goal_object_points=(flow_start + flow_target[-1]).astype(np.float32),
        object_goal_valid=flow_valid[-1].copy(), goal_object_slot=np.int64(0),
        target_surface_count=np.int64(SURFACE_POINTS), interaction_kind=np.array(kind),
        anchor_world_to_sample=anchor, profile=np.array(profile),
    )
