"""Inventory actual future-hand/native-action windows without creating training data."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[5]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training', type=Path, nargs='+', required=True)
    parser.add_argument('--teacher-packet', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.output.exists() or ROOT/'outputs/trajectory-policy' not in args.output.parents:
        raise ValueError('fresh task-owned inventory required')
    started = time.monotonic()
    records, files = [], [Path(__file__), args.teacher_packet]
    for folder in args.training:
        manifest = json.loads((folder/'manifest.json').read_text())
        result = json.loads((folder/'result.json').read_text())
        if manifest['status'] != 'COMPLETED' or manifest['engineering_smoke']:
            raise ValueError('completed scientific training trace required')
        if sha(folder/'final.pt') != manifest['final_checkpoint_sha256']:
            raise ValueError('final training identity drift')
        with np.load(folder/'low.npz') as f:
            data = {key:f[key].copy() for key in ('q', 'dq', 'hand', 'obj', 'velocity', 'action', 'applied', 'episode_start', 'episode_tick')}
        length, envs = data['q'].shape[:2]
        if data['hand'].shape != (length, envs, 11, 3) or data['action'].shape != (length, envs, 18):
            raise ValueError('state/action shapes differ')
        if not all(np.isfinite(v).all() for v in data.values()) or not np.array_equal(data['action'], data['applied']):
            raise ValueError('invalid or unpaired applied actions')
        if np.any(data['action'][:, :, [7, 9, 11, 13, 16, 17]]):
            raise ValueError('passive native action columns differ from expected zero convention')
        valid = np.arange(max(length-24, 0))
        offsets = np.arange(25)
        indices = valid[:, None] + offsets
        valid = valid[np.all(data['episode_start'][indices] == data['episode_start'][valid, None], axis=1)
            & np.all(data['episode_tick'][indices] == data['episode_tick'][valid, None]+offsets, axis=1)]
        episode_rows = []
        for start in np.unique(data['episode_start']):
            saved = int(np.sum(data['episode_start'] == start))
            windows = int(np.sum(data['episode_start'][valid] == start))
            episode_rows.append(dict(low_start=int(start), saved_before_states=saved,
                environment_episodes=envs, complete=saved == manifest['episode_controls'], valid_windows_per_env=windows))
        records.append(dict(folder=str(folder), seed=manifest['seed'], source_git_commit=manifest['git_commit'],
            available_state_fields=['q', 'dq', 'hand', 'obj', 'velocity'], native_action='clamped requested action == applied action; 12 active coordinates, 6 passive zeros',
            valid_low_indices=valid.tolist(), window_count=int(len(valid)*envs),
            complete_environment_episodes=sum(row['environment_episodes'] for row in episode_rows if row['complete']),
            partial_environment_episodes=sum(row['environment_episodes'] for row in episode_rows if not row['complete']),
            stable_completed_episodes=sum(frames >= 433 and held for episode in result['completed_episodes'] for frames, held in zip(episode['maximum_held'], episode['terminal_held'])),
            episodes=episode_rows))
        files.extend(folder/name for name in ('manifest.json', 'result.json', 'low.npz', 'final.pt'))
    with args.teacher_packet.open('rb') as f:
        teacher = pickle.load(f)
    report = dict(status='CONTRACT_PASS', horizon=24, action_prefix=8,
        tau_definition='Actual measured hand at t+1:t+24, never proposed/predicted tau; input state only at t; target applied native actions t:t+7',
        sources=records, total_windows=sum(row['window_count'] for row in records),
        teacher_packet=dict(path=str(args.teacher_packet), training_allowed=teacher['training_allowed'],
            engineering_only=teacher['engineering_only'], eligible_for_training=bool(teacher['training_allowed'] and not teacher['engineering_only'])),
        limitations=['No independent-demo equivalence: overlapping windows and repeated seed/initial state',
            'Split by whole source episode before normalization and window generation',
            'Failure-rich PPO traces do not replace competent behavior coverage',
            'No terminal/reset padding or labels inferred from q; actual native-action labels only',
            'This inventory proves availability/alignment, not inverse uniqueness, learning, or execution'],
        device='cpu', cpu_reason='File/array contract and index statistics only; no model calculation',
        elapsed_s=time.monotonic()-started)
    args.output.mkdir(parents=True)
    (args.output/'inventory.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    (args.output/'manifest.json').write_text(json.dumps(dict(status='COMPLETED',
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        input_sha256={str(p.resolve()):sha(p) for p in files}), indent=2)+'\n')
    print(json.dumps(dict(total_windows=report['total_windows'], teacher_packet=report['teacher_packet'],
        sources=[{k:row[k] for k in ('folder', 'seed', 'window_count', 'complete_environment_episodes', 'partial_environment_episodes', 'stable_completed_episodes')} for row in records]), indent=2))


if __name__ == '__main__':
    main()
