"""Recompute captured reset context and post-control bootstrap (no simulation)."""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path[:0] = [str(TASK / 'src'), str(ROOT / 'src/task/consequence-evaluator/src')]


def arrays(path):
    with np.load(path, allow_pickle=False) as stream:
        return {key: stream[key].copy() for key in stream.files}


def expected_observation(data):
    """Independent NumPy reconstruction of the residual observation contract."""
    obj, hand, refs = (data[key] for key in ('obj', 'hand', 'refs'))
    rotation, center = obj[:, :3, :3], obj[:, :3, 3]
    local_hand = np.einsum('bpj,bjk->bpk', hand - center[:, None], rotation)
    relative_rotation = rotation.transpose(0, 2, 1)[:, None] @ refs[:, :, :3, :3]
    relative_position = np.einsum('bkj,bji->bki', refs[:, :, :3, 3] - center[:, None], rotation)
    goal = np.concatenate((relative_position, relative_rotation[:, :, :, 0], relative_rotation[:, :, :, 1]), -1)
    n = len(obj)
    return np.concatenate((data['q'], data['dq'], data['previous'], center,
        rotation[:, :, 0], rotation[:, :, 1], data['velocity'], local_hand.reshape(n, -1),
        data['tau'][:, :8].reshape(n, -1), goal.reshape(n, -1)), -1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    run = args.run.resolve()
    manifest = json.loads((run / 'manifest.json').read_text())
    assert manifest['status'] == 'COMPLETED' and manifest['schema'] == 'ref2dex.wam-residual-ppo.v2'
    assert manifest['gae_lambda'] == 1.
    prefix = arrays(run / 'reset-prefix.npz')
    reset_count = 0
    initialized_held = 0
    for path in sorted(run.glob('reset-*.npz')):
        if path.name == 'reset-prefix.npz':
            continue
        data = arrays(path)
        starts = data['starts']
        for env, phase in enumerate(starts):
            for key, field in (('obj', 'object_history'), ('hand', 'hand_history')):
                expected = np.stack([prefix[key][max(0, int(phase) + offset), env]
                    if phase else data[field][-1, env] for offset in (-3, -2, -1)])
                np.testing.assert_array_equal(data[field][:3, env], expected)
            expected_previous = prefix['applied'][phase - 1, env] if phase else np.zeros(18)
            np.testing.assert_array_equal(data['previous'][env], expected_previous)
        np.testing.assert_array_equal(data['z0'], prefix['obj'][0, :, 2, 3])
        initialized_held += int(data['previous_held'].sum())
        reset_count += 1
    phases = set()
    for path in run.glob('diagnostic*.npz'):
        data = arrays(path)
        phases.update(data['phases'].tolist())
        from trajectory_policy.retargeter_data import ACTIVE
        np.testing.assert_array_equal(data['delta'], np.zeros_like(data['delta']))
        np.testing.assert_array_equal(data['applied'][..., list(ACTIVE)],
                                      np.clip(data['base'][..., list(ACTIVE)], -1, 1))
        for env, phase in enumerate(data['phases']):
            # FK on restored q must match the measured snapshot up to float noise.
            np.testing.assert_allclose(data['obj'][0, env], prefix['obj'][phase, env], atol=1e-6)
            np.testing.assert_allclose(data['hand'][0, env], prefix['hand'][phase, env], atol=1e-4)
            supported = data['table_footprint'][0, env] and abs(data['support_gap'][0, env]) <= .02
            held = (data['surface_gap'][0, env] <= .01 and not supported
                    and data['obj'][0, env, 2, 3] - prefix['obj'][0, env, 2, 3] >= .03)
            # Locate the reset for this diagnostic via its phase tuple.
            reset = next(arrays(p) for p in run.glob('reset-*.npz') if p.name != 'reset-prefix.npz'
                         and np.array_equal(arrays(p)['starts'], data['phases']))
            assert bool(reset['previous_held'][env]) == held
    assert phases == {0, 32, 48, 62}
    import torch
    from trajectory_policy.residual_ppo import ResidualValue
    torch.set_num_threads(2)
    value = ResidualValue().to(args.device)
    maximum_value_error = 0.
    bootstrap_count = 0
    for path in sorted(run.glob('bootstrap[0-9]*.npz')):
        data = arrays(path)
        np.testing.assert_allclose(data['observation'], expected_observation(data), rtol=1e-5, atol=1e-5)
        state = torch.load(run / (path.stem.replace('bootstrap', 'bootstrap-value') + '.pt'), map_location=args.device)
        value.load_state_dict(state)
        with torch.no_grad():
            actual = value(torch.as_tensor(data['observation'], device=args.device)).cpu().numpy()
        expected = np.where(data['done'], 0., actual)
        error = float(np.max(np.abs(expected - data['bootstrap'])))
        maximum_value_error = max(maximum_value_error, error)
        np.testing.assert_allclose(data['bootstrap'], expected, rtol=1e-5, atol=1e-5)
        assert not data['done'].any(), '96-step probe must be a truncation'
        bootstrap_count += 1
    assert bootstrap_count == manifest['updates']
    packet = torch.load(run / 'final.pt', map_location='cpu', weights_only=False)
    assert packet['update'] == manifest['updates'] + manifest['resume_updates']
    for key in ('actor_optimizer', 'value_optimizer'):
        assert packet[key]['state']
    assert set(packet['rng']) == {'torch', 'cuda', 'phase', 'numpy', 'python'}
    assert packet['phases'] == manifest['phases']
    result = dict(engineering_pass=True, resets=reset_count, bootstrap_queries=bootstrap_count,
                  bootstrap_value_max_error=maximum_value_error, initialized_held=initialized_held,
                  phases=sorted(phases), device=args.device, pid=os.getpid(),
                  claim='captured context/value wiring only; no grasp efficacy conclusion')
    (run / 'audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
