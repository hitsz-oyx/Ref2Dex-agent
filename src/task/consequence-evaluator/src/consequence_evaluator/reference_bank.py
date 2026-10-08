"""Frozen train-only physical references; no evaluation-driven selection."""
from pathlib import Path

import numpy as np

from .contracts import HAND_LINKS, K, is_within
from .data import sha
from .reference_motion import REFERENCE_SCHEMA
from .reference_progress import trajectory_features
from .value_outcomes import RAW_SCHEMA, task_trace, validate_plan_execution

BANK_SCHEMA = 'ref2dex.consequence-reference-progress.physical-bank.v1'
BANK_ORIGIN = 'measured_successful_robot_rollouts'


def select_physical_references(source, raw):
    """First eight qualified clean train230 episodes, in recorded order.

    Full-episode weak success qualifies fixed training demonstrations only.
    It is never an actual trajectory input or a candidate value label.
    """
    source = Path(source).resolve()
    if (raw.get('schema') != RAW_SCHEMA or raw.get('status') != 'COMPLETED'
            or raw.get('seed') != 230 or raw.get('fps') != 30 or raw.get('units') != 'm'
            or raw.get('rollout_kind') != 'continuous' or raw.get('audit_only') is not False
            or raw.get('horizon') != K or raw.get('execution_horizon') != K):
        raise ValueError('completed independent full train230 physical source required')
    members, packets, frozen = [], [], {}
    for record in raw['episodes']:
        if record['split'] != 'train' or record['assigned_phase'] != 'clean' or record['perturbation_tick'] >= 0:
            continue
        if record['split_group'] != 'source_seed:230':
            raise ValueError('reference training group mismatch')
        paths = [source / record['path'], source / record['diagnostics']]
        for path, key in zip(paths, ('sha256', 'diagnostics_sha256')):
            if not is_within(path, source) or sha(path) != record[key]:
                raise ValueError('reference packet drift/path escape')
            frozen[str(path)] = record[key]
        with np.load(paths[0], allow_pickle=False) as f, np.load(paths[1], allow_pickle=False) as d:
            packet = {k: f[k] for k in f.files}
            diagnostics = {k: d[k] for k in d.files}
        validate_plan_execution(packet, record, raw.get('requested_residual_bound', .2))
        if not task_trace(packet, diagnostics)['task_success']:
            continue
        if (record['steps'] != len(packet['action']) or len(packet['object_pose']) != 543
                or not np.allclose(packet['timestamps'], np.arange(543) / 30, atol=1e-9, rtol=0)):
            raise ValueError('this bank Probe requires full543frame native30Hz trajectories')
        trajectory_features(packet['object_pose'], packet['hand_keypoints'], packet['timestamps'])
        members.append(record); packets.append(packet)
        if len(members) == 8:
            break
    if (len(members) != 8 or len({m['episode'] for m in members}) != 8
            or len({(m['task'], m['motion']) for m in members}) != 1):
        raise ValueError('eight distinct qualified references from one task/motion required')
    return members, packets, frozen


def load_reference_features(directory, meta):
    """Return reference-only features, preserving the old single-R contract."""
    path = Path(directory) / 'reference.npz'
    if (meta.get('status') != 'COMPLETED' or meta.get('hand_links') != list(HAND_LINKS)
            or sha(path) != meta['reference_sha256']):
        raise ValueError('completed matching-hand reference with fixed hash required')
    with np.load(path, allow_pickle=False) as data:
        poses, points, times = data['object_pose'], data['hand_keypoints'], data['timestamps']
        if meta.get('schema') == REFERENCE_SCHEMA and meta.get('origin') == 'original_successful_retargeted_motion':
            return [trajectory_features(poses, points, times)], poses[:, 2, 3].copy()
        if (meta.get('schema') != BANK_SCHEMA or meta.get('origin') != BANK_ORIGIN
                or meta.get('source_seed') != 230 or meta.get('aggregation') != 'fixed_uniform_mean'
                or poses.shape != (8, 543, 4, 4) or points.shape != (8, 543, 11, 3)
                or times.shape != (8, 543) or len(meta.get('members', [])) != 8
                or len({m['episode'] for m in meta['members']}) != 8):
            raise ValueError('fixed eight-member measured train230 reference bank required')
        features = [trajectory_features(p, h, t) for p, h, t in zip(poses, points, times)]
        return features, poses[0, :, 2, 3].copy()
