#!/usr/bin/env python3
"""Fit/cal-only sensor qualification; no neural models or simulator execution."""
import argparse
import hashlib
import json
import time
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def run(qualification, output):
    import torch
    torch.set_num_threads(2)
    begin = time.monotonic()
    if output.exists():
        raise ValueError('unique output required')
    source = json.loads(qualification.read_text())
    if source['status'] != 'COMPLETED' or not source['supervision_adequate']:
        raise ValueError('audited qualified source required')
    parts = {'fit': [], 'cal': []}
    hashes = {str(qualification.resolve()): sha(qualification), str(Path(__file__).resolve()): sha(Path(__file__))}
    for name, expected in source['input_sha256'].items():
        path = Path(name)
        if sha(path) != expected:
            raise ValueError('source drift')
        hashes[name] = expected
        record = torch.load(path, weights_only=False, map_location='cpu')
        # Mixed records are filtered before any future/force calculations. No held metrics.
        bucket = torch.tensor([int(hashlib.sha256(f'9851/{int(i)}/{int(j)}'.encode()).hexdigest()[:8], 16) % 100
                               for i, j in zip(record['motion_id'], record['start_frame'])])
        for split, selected in [('fit', bucket < 50), ('cal', (bucket >= 50) & (bucket < 70))]:
            if not selected.any():
                continue
            future = record['future_state'][selected]
            if record['future_done'][selected].any():
                raise ValueError('reset-contaminated source')
            pre = torch.cat((record['state'][selected, None], future[:, :-1]), 1)
            post_force = record['future_object_force'][selected]
            pre_force = torch.cat((record['initial_object_force'][selected, None], post_force[:, :-1]), 1)
            mass = record['mass_kg'][selected, None, None]
            gravity = torch.zeros_like(post_force)
            gravity[..., 2] = -float(record['gravity_magnitude'])
            delta = future[:, :, 43:46] - pre[:, :, 43:46]
            clearance = torch.cat((record['initial_clearance'][selected, None], record['future_clearance'][selected, :-1]), 1)
            early = ~((pre[:, :, 38] - record['rest_z'][selected, None] >= .03) & (clearance >= .002))
            impulses = {'constant_velocity': torch.zeros_like(delta), 'gravity_only': gravity / 30,
                        'pre_force': (pre_force / mass + gravity) / 30,
                        'post_force': (post_force / mass + gravity) / 30,
                        'endpoint_average': ((pre_force + post_force) / (2 * mass) + gravity) / 30}
            parts[split].append(dict(delta=delta.reshape(-1, 3), early=early.flatten(),
                                     **{k: v.reshape(-1, 3) for k, v in impulses.items()}))
        if time.monotonic() - begin > 600:
            raise TimeoutError('10-minute qualification budget')
    metrics = {}
    for split, blocks in parts.items():
        data = {k: torch.cat([b[k] for b in blocks]) for k in blocks[0]}
        dynamic = data['delta'].norm(dim=-1) > .05
        masks = dict(all=torch.ones_like(dynamic), dynamic=dynamic, early=data['early'], early_dynamic=data['early'] & dynamic)
        metrics[split] = {}
        for label, mask in masks.items():
            count = int(mask.sum())
            metrics[split][label] = dict(rows=count, velocity_vector_rmse_m_s={})
            if not count:
                continue
            for estimator in ['constant_velocity', 'gravity_only', 'pre_force', 'post_force', 'endpoint_average']:
                error = data[estimator][mask] - data['delta'][mask]
                metrics[split][label]['velocity_vector_rmse_m_s'][estimator] = float(error.square().sum(-1).mean().sqrt())
    estimators = ['pre_force', 'post_force', 'endpoint_average']
    selected = min(estimators, key=lambda k: metrics['fit']['dynamic']['velocity_vector_rmse_m_s'][k])
    cal = metrics['cal']['dynamic']
    sufficient = cal['rows'] >= 500
    gain = 1 - cal['velocity_vector_rmse_m_s'][selected] / cal['velocity_vector_rmse_m_s']['constant_velocity']
    result = dict(experiment_id='P-20261002-contact-impulse-observability', status='COMPLETED',
                  label='PROMISING' if sufficient and gain >= .2 else ('UNPROMISING' if sufficient else 'UNCLEAR'),
                  selected_on_fit=selected, cal_relative_rmse_improvement=gain,
                  gate=dict(cal_dynamic_support=sufficient, cal_rmse20pct=gain >= .2), metrics=metrics,
                  input_sha256=hashes, elapsed_seconds=time.monotonic() - begin, dt_seconds=1 / 30,
                  device_reason='CPU label/statistical calculations only; no neural fitting or inference',
                  scope='sensor observability; not Cm prediction, action ranking or actual control utility')
    output.write_text(json.dumps(result, indent=2) + '\n')
    if output.stat().st_size > 64 << 20:
        raise ValueError('64MiB output budget')
    print(json.dumps({k: v for k, v in result.items() if k not in ['input_sha256', 'metrics']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qualification', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.qualification, args.output)
