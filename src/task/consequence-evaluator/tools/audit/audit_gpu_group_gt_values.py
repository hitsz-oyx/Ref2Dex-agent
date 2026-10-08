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
    roles = list(packet.get('candidate_roles') or ('baseline', 'zero_repeat', 'positive', 'negative'))
    if roles[:4] != ['baseline', 'zero_repeat', 'positive', 'negative']:
        raise ValueError('native group role contract mismatch')
    candidate_values = values[[0, 2, 3]]
    frozen_choice = choose_candidate(candidate_values)
    return dict(
        schema='ref2dex.consequence-gate1.gpu-group-gt-value-audit.v1',
        engineering_only=True,
        packet=str(Path(packet_path).resolve()),
        packet_sha256=sha(packet_path),
        group_mode=packet.get('group_mode'), group_envs=int(packet['group_envs']),
        query_tick=query, horizon=K, roles=roles[:4],
        values=values.tolist(), progress_start=start.tolist(),
        progress_start_range=float(start.max() - start.min()),
        strict_same_state=bool(start.max() - start.min() <= 1e-9),
        strict_progress_start_tolerance=1e-9,
        argmax_role=roles[int(np.argmax(values))],
        frozen_choice=int(frozen_choice),
        frozen_choice_role=CANDIDATES[int(frozen_choice)],
        frozen_deadzone=.01,
        chosen_vs_baseline=float(values[[0, 2, 3]][frozen_choice] - values[0]),
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
