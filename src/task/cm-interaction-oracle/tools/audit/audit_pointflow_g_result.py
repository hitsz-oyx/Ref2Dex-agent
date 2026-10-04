#!/usr/bin/env python3
"""Statistical audit of saved Ref5 results; no model fitting or inference."""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'run'))
from probe_pointflow_g import pose_effect, sha256
from src.task.CmResidual.dexplore_cm_geometry import dexplore_root_pose


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    output = args.run_dir / 'statistical_audit.json'
    if output.exists():
        raise FileExistsError(output)
    report = json.loads((args.run_dir / 'result.json').read_text())
    data = torch.load(args.run_dir / 'bridge.pt', map_location='cpu', weights_only=False)
    source_path = Path(report['args']['input'])
    if sha256(source_path) != report['input_sha256']:
        raise RuntimeError('source changed since probe')
    source = torch.load(source_path, map_location='cpu', weights_only=False, mmap=True)
    subset = data['indices']
    source_keys = torch.stack((source.get('source_namespace', torch.zeros_like(source['source_run'])),
                               source['source_run'], source['episode_id']), -1)[subset]
    test_keys = {tuple(key) for key in report['test_groups']}
    test = torch.tensor([tuple(key) in test_keys for key in source_keys.tolist()])
    assert torch.equal(source_keys[test], data['test_keys'])
    assert torch.allclose(pose_effect(source['effect'][subset, :1]), data['gt_effect'])
    state = source['state'][subset][test]
    rotation = dexplore_root_pose(state[:, 36:49])[:, :3, :3].transpose(-1, -2)
    rotate = lambda v: torch.einsum('bij,bj->bi', rotation, v)
    dt = float(source['metadata']['runs'][0]['control_dt'])
    # Small-step rigid-motion persistence from *current* world-frame twist.
    persistence = torch.cat((rotate(state[:, 43:46]) * dt,
                             rotate(state[:, 46:49]) * dt), -1)[:, None]
    error = persistence - data['gt_effect'][test]
    baseline = {'translation_rmse_m': float(error[..., :3].square().mean().sqrt()),
                'axis_angle_rmse_rad': float(error[..., 3:].square().mean().sqrt()),
                'train_standardized_rmse': float((error / data['effect_scale']).square().mean().sqrt()),
                'rotation_contract': 'first-order local angular velocity * dt, not integrated dynamics'}
    keys = data['test_keys']
    groups = [tuple(g) for g in report['test_groups']]
    mean_abs_target = [float(data['target'][(keys == torch.tensor(g)).all(-1)].abs().mean()) for g in groups]
    largest = max(range(len(groups)), key=lambda i: mean_abs_target[i])
    h = torch.tensor(report['arms']['H']['test_group_mae'])
    result = {'source_verified': True, 'effect_reconstruction_verified': True,
              'test_key_order_verified': True, 'current_twist_persistence': baseline,
              'largest_abs_target_group': groups[largest],
              'largest_group_mean_abs_target': mean_abs_target[largest],
              'largest_group_share_of_H_error': float(h[largest] / h.sum()),
              'comparisons': {}}
    for arm in ('HE_GT', 'HE_CM_swap', 'HE_CM_fit', 'HA', 'HAR'):
        if arm not in report['arms']:
            continue
        values = torch.tensor(report['arms'][arm]['test_group_mae'])
        delta = h - values
        keep = torch.arange(len(groups)) != largest
        result['comparisons'][arm] = {
            'positive_episode_count': int((delta > 0).sum()), 'total_episode_count': len(groups),
            'largest_target_group_absolute_mae_reduction': float(delta[largest]),
            'remaining_groups_mean_mae_reduction': float(delta[keep].mean()),
            'remaining_groups_relative_mae_reduction': float(delta[keep].mean() / h[keep].mean()),
            'largest_target_group_share_of_total_delta': float(delta[largest] / delta.sum()) if delta.sum().abs() > 1e-8 else None}
    result['interpretation'] = ('Largest-group exclusion is a sensitivity diagnostic, not a new primary metric. '
                                'CM predictions are deterministic geometric/action features; CM-fit G gain '
                                'alone cannot establish that accurate physical prediction caused gain.')
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
