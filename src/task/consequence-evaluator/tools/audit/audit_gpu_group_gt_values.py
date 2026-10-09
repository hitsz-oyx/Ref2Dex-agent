"""Audit physical-bank GT progress on a native GPU engineering group packet.

This is deliberately separate from the production score worker: engineering
packets may contain a measured zero-pair drift and therefore cannot enter a
strict same-state Gate1 score. The output is an auditable diagnostic only.
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
import sys
sys.path[:0] = [str(ROOT), str(TASK / 'src')]

from consequence_evaluator.gate1 import CANDIDATES, K, choose_candidate
from consequence_evaluator.provenance import sha
from consequence_evaluator.reference_bank import BANK_SCHEMA, load_reference_features
from consequence_evaluator.reference_progress import trajectory_features
from consequence_evaluator.temporal_phase import ENCODER_SCHEMA, LearnedReferenceProgress


def load_packet(path):
    with Path(path).open('rb') as stream:
        packet = pickle.load(stream)
    if packet.get('engineering_only') is not True:
        raise ValueError('engineering-only packet required')
    if packet.get('group_envs', 0) < 4 or packet.get('query_tick') != 48:
        raise ValueError('native four-role query-tick48 group packet required')
    actions = np.asarray(packet.get('actions'))
    if actions.ndim != 3 or actions.shape[0] < 48 + K or actions.shape[1] < 4 or actions.shape[2] != 18:
        raise ValueError('packet must contain a complete 24-step candidate window')
    if not np.isfinite(actions).all() or np.abs(actions).max() > 1 + 1e-6:
        raise ValueError('packet actions are not finite native controls')
    for key in ('object_pose', 'hand_keypoints', 'timestamps'):
        if key not in packet:
            raise ValueError('packet missing %s' % key)
    if len(packet['object_pose']) != len(packet['timestamps']):
        raise ValueError('packet state/timestamp length mismatch')
    return packet


def summarize_value_noise(values, base_roles, zero_roles, role_indices=None):
    """Return stable role names and descriptive scalar-Y pair noise.

    The four candidate roles are fixed at indices 0--3.  Extra environments in
    a multi-zero group are nominal roles and must remain addressable in the
    report even though they are not candidates.
    """
    values = np.asarray(values, dtype=np.float64)
    base_roles = list(base_roles)
    zero_roles = [int(index) for index in zero_roles]
    if role_indices is None:
        role_indices = dict(baseline=0, zero_repeat=1, positive=2, negative=3)
    else:
        role_indices = {str(name): int(index) for name, index in role_indices.items()}
    required = ('baseline', 'zero_repeat', 'positive', 'negative')
    if (values.ndim != 1 or values.size < 4
            or base_roles[:4] != list(required)
            or any(name not in role_indices for name in required)
            or len(set(role_indices[name] for name in required)) != 4
            or any(role_indices[name] < 0 or role_indices[name] >= values.size for name in required)
            or len(set(zero_roles)) != len(zero_roles)
            or any(index < 0 or index >= values.size for index in zero_roles)
            or any(index in (role_indices['positive'], role_indices['negative']) for index in zero_roles)):
        raise ValueError('native group value-noise role contract mismatch')
    roles = ['zero_role_%d' % index for index in range(values.size)]
    for name, index in role_indices.items():
        if name in required:
            roles[index] = name
    zero_pair_deltas = []
    for left_offset, left in enumerate(zero_roles):
        for right in zero_roles[left_offset + 1:]:
            zero_pair_deltas.append(dict(left=left, right=right,
                                         delta_y=float(values[right] - values[left])))
    zero_pair_abs = np.asarray([abs(item['delta_y']) for item in zero_pair_deltas], dtype=np.float64)
    baseline_index = role_indices['baseline']
    baseline_zero_deltas = [dict(role=role, delta_y=float(values[role] - values[baseline_index]))
                            for role in zero_roles if role != baseline_index]
    baseline_zero_abs = np.asarray([abs(item['delta_y']) for item in baseline_zero_deltas], dtype=np.float64)
    candidate_deltas = [dict(role=role_indices[name], candidate=CANDIDATES[offset],
                             delta_y=float(values[role_indices[name]] - values[baseline_index]))
                       for offset, name in enumerate(('positive', 'negative'), start=1)]
    zero_pair_median = float(np.median(zero_pair_abs)) if zero_pair_abs.size else 0.0
    baseline_zero_median = float(np.median(baseline_zero_abs)) if baseline_zero_abs.size else 0.0
    for item in candidate_deltas:
        item['abs_over_zero_pair_median'] = (
            float(abs(item['delta_y']) / zero_pair_median) if zero_pair_median > 0 else None)
        item['abs_over_baseline_zero_median'] = (
            float(abs(item['delta_y']) / baseline_zero_median) if baseline_zero_median > 0 else None)
    return roles, dict(
        role_indices=role_indices,
        zero_pair_deltas=zero_pair_deltas,
        zero_pair_abs_median=zero_pair_median,
        baseline_relative_zero_deltas=baseline_zero_deltas,
        baseline_relative_zero_abs_median=baseline_zero_median,
        candidate_deltas_vs_baseline=candidate_deltas,
        interpretation='descriptive Y-space noise calibration; no strict same-state or Gate1 claim')


def audit_packet(packet, packet_path, reference, encoder, device):
    meta = json.loads((Path(reference) / 'manifest.json').read_text())
    trained = json.loads((Path(encoder) / 'manifest.json').read_text())
    checkpoint = Path(trained['checkpoint'])
    if (meta.get('schema') != BANK_SCHEMA or trained.get('schema') != ENCODER_SCHEMA
            or trained.get('status') != 'COMPLETED'
            or trained.get('reference_sha256') != meta.get('reference_sha256')
            or sha(checkpoint) != trained.get('checkpoint_sha256')):
        raise ValueError('frozen physical reference-bank/encoder contract mismatch')
    references, _ = load_reference_features(reference, meta)
    matcher = LearnedReferenceProgress(
        references, torch.load(checkpoint, map_location='cpu', weights_only=False), device)

    query = int(packet['query_tick'])
    end = query + K
    values = []
    progress_start = []
    for env_index in range(int(packet['group_envs'])):
        trace = matcher.align(trajectory_features(
            np.asarray(packet['object_pose'])[:, env_index],
            np.asarray(packet['hand_keypoints'])[:, env_index],
            np.asarray(packet['timestamps'])))
        values.append(float(trace['progress'][end] - trace['progress'][query]))
        progress_start.append(float(trace['progress'][query]))
    values = np.asarray(values, dtype=np.float64)
    start = np.asarray(progress_start, dtype=np.float64)
    base_roles = list(packet.get('candidate_roles') or
                      ('baseline', 'zero_repeat', 'positive', 'negative'))
    zero_roles = [int(index) for index in packet.get('zero_role_indices', (0, 1))]
    role_indices = packet.get('role_index_map') or packet.get('role_indices')
    if role_indices is None:
        candidate_env_pair = packet.get('candidate_env_pair', [2, 3])
        zero_pair = packet.get('zero_env_pair', [0, 1])
        if list(zero_pair) != [0, 1] or list(candidate_env_pair) != [2, 3]:
            raise ValueError('swapped group role metadata is required for Value audit')
        role_indices = dict(baseline=0, zero_repeat=1, positive=2, negative=3)
    roles, value_noise = summarize_value_noise(values, base_roles, zero_roles, role_indices)
    baseline_index = int(role_indices['baseline'])
    positive_index = int(role_indices['positive'])
    negative_index = int(role_indices['negative'])
    candidate_values = values[[baseline_index, positive_index, negative_index]]
    frozen_choice = choose_candidate(candidate_values)
    return dict(
        schema='ref2dex.consequence-gate1.gpu-group-gt-value-audit.v1',
        engineering_only=True,
        packet=str(Path(packet_path).resolve()),
        packet_sha256=sha(packet_path),
        group_mode=packet.get('group_mode'), group_envs=int(packet['group_envs']),
        query_tick=query, horizon=K, roles=roles,
        values=values.tolist(), progress_start=start.tolist(),
        progress_start_range=float(start.max() - start.min()),
        strict_same_state=bool(start.max() - start.min() <= 1e-9),
        strict_progress_start_tolerance=1e-9,
        role_names=roles, zero_role_indices=zero_roles,
        role_indices=role_indices,
        candidate_env_pair=[positive_index, negative_index],
        argmax_role=roles[int(np.argmax(values))],
        frozen_choice=int(frozen_choice),
        frozen_choice_role=CANDIDATES[int(frozen_choice)],
        frozen_deadzone=.01,
        chosen_vs_baseline=float(candidate_values[frozen_choice] - values[baseline_index]),
        value_noise=value_noise,
        reference_sha256=meta['reference_sha256'],
        encoder_checkpoint_sha256=trained['checkpoint_sha256'],
        note='engineering diagnostic; no GT Gate1 score or scientific candidate claim',
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--encoder', type=Path, required=True)
    parser.add_argument('--packet', type=Path, nargs='+', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for packet_path in args.packet:
        if not packet_path.is_file():
            raise FileNotFoundError(str(packet_path))
        results.append(audit_packet(load_packet(packet_path), packet_path,
                                    args.reference, args.encoder, args.device))
    output = args.output_dir / 'gt-value-audit.json'
    if output.exists():
        raise FileExistsError(str(output))
    output.write_text(json.dumps(dict(
        schema='ref2dex.consequence-gate1.gpu-group-gt-value-audit.v1',
        engineering_only=True, results=results), indent=2) + '\n')
    print(output)


if __name__ == '__main__':
    main()
