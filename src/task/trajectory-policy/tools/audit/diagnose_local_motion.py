"""Decompose a frozen local-motion inverse result without retraining or physics."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch


TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
ACTIVE = (0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 15)
NAMES = ('wx', 'wy', 'wz', 'rx', 'ry', 'rz', 'thumb', 'index', 'middle',
         'ring', 'little', 'spread')


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def reconstruct_validation(data):
    """Rebuild only the validation labels from one complete reset wave."""
    indices = []
    for tick in range(len(data['q']) - 24):
        if int(data['episode_start'][tick]) != 2168:
            continue
        rows = np.arange(tick, tick + 25)
        if not np.all(data['episode_start'][rows] == data['episode_start'][tick]):
            continue
        if not np.array_equal(data['episode_tick'][rows], data['episode_tick'][tick] + np.arange(25)):
            raise ValueError('non-contiguous validation episode ticks')
        indices.append(tick)
    indices = np.asarray(indices, dtype=np.int64)
    action = np.stack(
        [data['applied'][indices + step][..., ACTIVE] for step in range(8)], axis=2
    )
    return action.reshape(-1, 8, 12), np.repeat(data['episode_tick'][indices], 16)


def group_metrics(error, action_scale, controller_scale, prediction):
    normalized = error / action_scale
    first_pd = error[:, 0] * controller_scale
    return {
        'normalized_l1': float(normalized.mean()),
        'normalized_l1_by_horizon': np.mean(normalized, axis=(0, 2)).tolist(),
        'normalized_l1_by_dof': dict(zip(NAMES, np.mean(normalized, axis=(0, 1)).tolist())),
        'raw_mae_by_dof': dict(zip(NAMES, np.mean(error, axis=(0, 1)).tolist())),
        'first_raw_mae_by_dof': dict(zip(NAMES, np.mean(error[:, 0], axis=0).tolist())),
        'first_pd_p95_by_dof': dict(zip(NAMES, np.quantile(first_pd, .95, axis=0).tolist())),
        'first_pd_max_p95': {
            'position_m': float(np.quantile(first_pd[:, :3].max(-1), .95)),
            'rotation_rad': float(np.quantile(first_pd[:, 3:6].max(-1), .95)),
            'fingers_rad': float(np.quantile(first_pd[:, 6:].max(-1), .95)),
        },
        'first_raw_clipping_rate': float((np.abs(prediction[:, 0]) > 1 + 1e-6).any(-1).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    fit = args.fit.resolve()
    audit = args.audit.resolve()
    output = args.output.resolve()
    if output.exists() or ROOT / 'outputs/trajectory-policy' not in output.parents:
        raise ValueError('fresh task-owned output required')
    started = time.monotonic()
    output.mkdir(parents=True)
    manifest = {
        'status': 'RUNNING',
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'fit': str(fit),
        'audit': str(audit),
        'execution': 'CPU-only pure statistics; no model forward, training, physics, or new data',
    }
    write(output / 'manifest.json', manifest)
    try:
        fit_manifest = json.loads((fit / 'manifest.json').read_text())
        fit_result = json.loads((fit / 'result.json').read_text())
        audit_manifest = json.loads((audit / 'manifest.json').read_text())
        if fit_manifest.get('status') != 'COMPLETED' or audit_manifest.get('audit') != 'PASS':
            raise ValueError('completed fit and PASS audit required')
        with np.load(fit / 'validation_predictions.npz', allow_pickle=False) as saved:
            predictions = {key: saved[key].copy() for key in saved.files}
        if set(predictions) != {'tau', 'state_only'}:
            raise ValueError('expected tau and state_only saved predictions')
        truth_parts, tick_parts = [], []
        input_hashes = {
            str(fit / 'manifest.json'): sha(fit / 'manifest.json'),
            str(fit / 'result.json'): sha(fit / 'result.json'),
            str(fit / 'validation_predictions.npz'): sha(fit / 'validation_predictions.npz'),
            str(fit / 'tau-best.pt'): sha(fit / 'tau-best.pt'),
            str(fit / 'state_only-best.pt'): sha(fit / 'state_only-best.pt'),
            str(audit / 'manifest.json'): sha(audit / 'manifest.json'),
            str(Path(__file__)): sha(Path(__file__)),
        }
        for metadata in fit_manifest['metadata']:
            folder = Path(metadata['source'])
            low = folder / 'low.npz'
            with np.load(low, allow_pickle=False) as source:
                data = {key: source[key] for key in (
                    'q', 'hand', 'applied', 'episode_start', 'episode_tick')}
            truth, ticks = reconstruct_validation(data)
            truth_parts.append(truth)
            tick_parts.append(ticks)
            input_hashes[str(low)] = sha(low)
            input_hashes[str(folder / 'manifest.json')] = sha(folder / 'manifest.json')
        truth = np.concatenate(truth_parts)
        ticks = np.concatenate(tick_parts)
        if truth.shape != (16576, 8, 12) or any(value.shape != truth.shape for value in predictions.values()):
            raise ValueError('validation shape drift')
        if not np.isfinite(truth).all() or any(not np.isfinite(value).all() for value in predictions.values()):
            raise ValueError('nonfinite frozen arrays')
        packet = torch.load(fit / 'tau-best.pt', map_location='cpu', weights_only=False)
        action_scale = np.asarray(packet['model']['action_scale'], dtype=np.float64)
        if action_scale.shape != (8, 12) or (action_scale <= 0).any():
            raise ValueError('invalid action scale')
        controller_scale = np.asarray(fit_manifest['native_controller']['scale'], dtype=np.float64)[list(ACTIVE)]
        controller_scale[6:] /= 2
        reports = {}
        for role, prediction in predictions.items():
            reports[role] = group_metrics(
                np.abs(prediction.astype(np.float64) - truth), action_scale, controller_scale, prediction)
        local = np.asarray(list(reports['tau']['normalized_l1_by_dof'].values()))
        baseline = np.asarray(list(reports['state_only']['normalized_l1_by_dof'].values()))
        relative_gain = (baseline - local) / baseline
        result = {
            'status': 'COMPLETED',
            'fit_status': fit_result['status'],
            'audit_status': audit_manifest['audit'],
            'samples': int(len(truth)),
            'validation_tick_minmax': [int(ticks.min()), int(ticks.max())],
            'tau': reports['tau'],
            'state_only': reports['state_only'],
            'local_relative_normalized_gain_by_dof': dict(zip(NAMES, relative_gain.tolist())),
            'target_mean_abs_by_dof': dict(zip(NAMES, np.mean(np.abs(truth), axis=(0, 1)).tolist())),
            'interpretation': {
                'command_identification': 'UNRESOLVED: local motion lowers action residuals but first-command residuals remain large in native units',
                'pd_conversion_wiring': 'NO_FAILURE_FOUND: the paired fit audit reconstructed native PD scales and labels exactly within tolerance',
                'native_execution_preload': 'UNTESTED_OFFLINE: this decomposition cannot establish closed-loop solver or contact/preload semantics',
                'next_decision': 'Do not enter physics with this checkpoint or add training budget; require a separate bounded execution/conditioning diagnosis before another architecture intervention.',
            },
        }
        manifest.update(status='COMPLETED', input_sha256=input_hashes,
                        elapsed_s=time.monotonic() - started, samples=int(len(truth)))
        write(output / 'result.json', result)
        write(output / 'manifest.json', manifest)
        print(json.dumps(result, indent=2))
    except BaseException as error:
        manifest.update(status='FAILED', error=repr(error), elapsed_s=time.monotonic() - started)
        write(output / 'manifest.json', manifest)
        raise


if __name__ == '__main__':
    main()
