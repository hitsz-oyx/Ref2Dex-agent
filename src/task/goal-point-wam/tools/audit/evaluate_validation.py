#!/usr/bin/env python3
"""Compare optimizer variants and endpoint baselines on saved validation indices."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK / 'tools/run'))
from train_three_domain import GoalPointWAM, GoalWindows, evaluate, atomic_json, digest
from oakink_wm.multisource import MixedWindows


class EndpointBaseline(torch.nn.Module):
    def __init__(self, linear=False):
        super().__init__()
        self.linear = linear

    def forward(self, batch):
        trajectory = torch.zeros_like(batch['trajectory_target'])
        if self.linear:
            delta = batch['goal_hand'] - batch['start_hand']
            valid = batch['goal_hand_valid'] & batch['start_hand_valid']
            delta = torch.where(valid[..., None], delta, torch.zeros_like(delta))
            time = torch.arange(1, 25, device=delta.device, dtype=delta.dtype) / 24
            trajectory = delta[:, None] * time[None, :, None, None, None]
        b, o = batch['object_valid'].shape
        return dict(trajectory=trajectory, translation=batch['points'].new_zeros((b, o, 24, 3)),
                    rotation=torch.eye(3, device=batch['points'].device).expand(b, o, 24, 3, 3))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--stats', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--indices', type=Path, required=True)
    parser.add_argument('--previous-indices', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    if not torch.cuda.is_available(): raise RuntimeError('GPU required')
    torch.set_num_threads(2)
    indices = np.load(args.indices)
    if not np.array_equal(indices, np.load(args.previous_indices)):
        raise ValueError('matched validation indices differ')
    # Only a val loader is created; no test data is used by this tool.
    dataset = GoalWindows(MixedWindows(args.data, 'val'))
    stats = json.loads(args.stats.read_text())
    device = torch.device('cuda')
    result = dict(schema='goal-point-wam.matched-validation.v1', split='val',
                  indices=indices.tolist(), indices_sha256=digest(args.indices),
                  evaluator_sha256=digest(Path(__file__)), diagnostic_only=True, arms={})
    state_identity = []
    for name, path in (('lr1e-3', args.checkpoint), ('lr1e-4', args.previous)):
        state = torch.load(path, map_location='cpu', weights_only=False)
        identity = state['identity']
        if identity['data_manifest_sha256'] != digest(args.data / 'processed/manifest.json'):
            raise ValueError('data identity drift')
        if identity['stats_sha256'] != digest(args.stats): raise ValueError('stats drift')
        if identity['model_sha256'] != digest(TASK / 'src/goal_point_wam/model.py'):
            raise ValueError('model drift')
        model = GoalPointWAM(stats, state['config']).to(device)
        model.load_state_dict(state['model'], strict=True)
        result['arms'][name] = dict(checkpoint=str(path), checkpoint_sha256=digest(path),
                                   step=state['step'], metrics=evaluate(model, dataset, indices, 2, 0, device))
        state_identity.append(identity)
        del model, state
        torch.cuda.empty_cache()
    if state_identity[0]['initial_parameter_sha256'] != state_identity[1]['initial_parameter_sha256']:
        raise ValueError('matched initial weights differ')
    result['initial_parameter_sha256'] = state_identity[0]['initial_parameter_sha256']
    for name, linear in (('static', False), ('linear-goal', True)):
        result['arms'][name] = dict(metrics=evaluate(EndpointBaseline(linear), dataset, indices, 2, 0, device))
    gains, regression = {}, {}
    for name in dataset.source_names:
        world = 'source/' + name + '/moving_world_point_epe_m_h24'
        trajectory = 'source/' + name + '/trajectory_epe_m'
        new = result['arms']['lr1e-3']['metrics']
        old = result['arms']['lr1e-4']['metrics']
        static = result['arms']['static']['metrics']
        gains[name] = 1 - new[world] / static[world]
        regression[name] = new[trajectory] / old[trajectory] - 1
    result['moving_gain_vs_static'] = gains
    result['trajectory_delta_vs_previous'] = regression
    result['positive_gate'] = sum(gain >= .05 for gain in gains.values()) >= 2 and max(regression.values()) <= .1
    atomic_json(args.output, result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
