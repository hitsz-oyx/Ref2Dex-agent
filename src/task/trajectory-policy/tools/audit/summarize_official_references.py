"""CPU trace/controller/reference audit and cross-reference report."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path[:0] = [str(TASK / 'src'), str(ROOT / 'src/task/consequence-evaluator/src')]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def runs(mask):
    longest = current = 0
    for value in mask:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', type=Path, required=True)
    args = p.parse_args()
    args.directory = args.directory.resolve()
    bank = json.loads((ROOT / 'outputs/consequence-evaluator/corrected-expert-inputs-20261007-r2/manifest.json').read_text())
    rows = []
    from consequence_evaluator.gate1 import episode_outcome
    from trajectory_policy.reference_actor import reference_from_raw
    import torch
    for motion in bank['motion_inputs']:
        directory = args.directory / motion['sequence']
        if motion['sequence'] == 's1_cubesmall_lift' and (args.directory / (motion['sequence'] + '-retry-r2')).exists():
            # Pre-control loader-guard failure has one declared implementation
            # retry; this is not selecting a rollout by physical performance.
            directory = args.directory / (motion['sequence'] + '-retry-r2')
        if not (directory / 'result.json').exists():
            continue
        manifest = json.loads((directory / 'manifest.json').read_text())
        result = json.loads((directory / 'result.json').read_text())
        assert manifest['status'] == 'COMPLETED' and not manifest['training_allowed']
        authenticated_updates = []
        for path, digest in manifest['input_sha256'].items():
            if sha(path) != digest:
                relative = Path(path).relative_to(ROOT)
                if relative not in (Path('src/task/trajectory-policy/tools/run/evaluate_official_reference.py'),
                    Path('src/task/trajectory-policy/docs/experiments/probes/P-20261011-official-reference-screen.md')):
                    raise AssertionError('immutable input drift: ' + path)
                original = subprocess.check_output(['git', 'show', manifest['git_commit'] + ':' + str(relative)], cwd=ROOT)
                assert hashlib.sha256(original).hexdigest() == digest
                authenticated_updates.append(str(relative))
        assert sha(directory / 'trajectory.npz') == manifest['trajectory_sha256']
        assert sha(motion['path']) == motion['sha256']
        with np.load(directory / 'trajectory.npz', allow_pickle=False) as stream:
            data = {k: stream[k] for k in stream.files}
        assert all(np.isfinite(v).all() for v in data.values())
        assert data['obj'].shape == (motion['frames'], 8, 4, 4)
        assert data['obs'].shape == (motion['frames'] - 1, 8, 1442)
        raw = torch.load(motion['path'], map_location='cpu', weights_only=False).numpy()
        ref = reference_from_raw(raw)
        errors = {k: float(np.max(np.abs(data[key] - ref[k]))) for k, key in
                  (('obj', 'reference_obj'), ('hand_pos', 'reference_hand'))}
        assert max(errors.values()) < 2e-5, errors
        assert np.array_equal(data['action'], data['applied'])
        action = data['action'].copy()
        action[..., 6:] = (1 + action[..., 6:]) / 2
        controller = manifest['native_controller']
        pd = (action * np.asarray(controller['scale'], np.float32)
              + np.asarray(controller['offset'], np.float32))
        pd[..., :6] += data['q'][:-1, ..., :6]
        for distal, parent, factor in ((7, 6, 1.05), (9, 8, 1.05), (11, 10, 1.05),
                                        (13, 12, 1.05), (16, 15, .6), (17, 15, .8)):
            pd[..., distal] = pd[..., parent] * factor
        pd_error = float(np.max(np.abs(pd - data['pd_targets'])))
        assert pd_error < 1e-6, pd_error
        lift5 = hold45 = placed = 0
        for env in range(8):
            supported = data['table_footprint'][:, env] & (np.abs(data['support_gap'][:, env]) <= .02)
            held = ((data['surface_gap'][:, env] <= .01) & ~supported
                & (data['obj'][:, env, 2, 3] - data['obj'][0, env, 2, 3] >= .03))
            held[0] = False
            maximum = runs(held)
            lift5 += maximum >= 5
            hold45 += maximum >= 45
            outcome = episode_outcome(dict(object_pose=data['obj'][:, env],
                object_velocity=data['velocity'][:, env], surface_gap=data['surface_gap'][:, env],
                support_gap=data['support_gap'][:, env], table_footprint=data['table_footprint'][:, env]))
            placed += outcome['success']
            assert maximum == result['outcomes'][env]['maximum_held_frames']
            assert outcome['success'] == result['outcomes'][env]['success']
        position = data['obj'][..., :3, 3] - ref['obj'][:, None, :3, 3]
        rmse = float(np.sqrt(np.mean(np.sum(position**2, -1))))
        assert (lift5, hold45, placed) == (result['lift5'], result['hold45'], result['controlled_place'])
        assert abs(rmse - result['reference_position_rmse_m']) < 1e-6
        rows.append(dict(sequence=motion['sequence'], frames=motion['frames'], **result,
            elapsed_s=manifest['elapsed_s'], producer_run=str(directory.relative_to(ROOT)),
            git_commit=manifest['git_commit'], audit=dict(pass_=True, pd_max_error=pd_error,
            reference_max_errors=errors, authenticated_source_updates=authenticated_updates,
            trace_sha256=manifest['trajectory_sha256'])))
    summary = dict(status='UNCLEAR', scope='fixed frame0 official teacher probe; correlated environment repeats',
        expected_references=12, completed=len(rows), audit_pass=True, rows=rows)
    (args.directory / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    table = ['| Reference | Lift5 | Hold45 | Controlled place | Object pos RMSE (cm) | Ref ends supported |',
             '|---|---:|---:|---:|---:|---|']
    for row in rows:
        table.append('| %s | %d/8 | %d/8 | %d/8 | %.2f | %s |' % (row['sequence'], row['lift5'],
            row['hold45'], row['controlled_place'], row['reference_position_rmse_m'] * 100,
            'yes' if row['reference_terminal_support'] else 'no'))
    (args.directory / 'summary.md').write_text('\n'.join(table) + '\n')
    if rows:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 7), sharey=True)
        y = np.arange(len(rows))
        axes[0].barh(y - .18, [r['hold45'] for r in rows], height=.35, label='hold45')
        axes[0].barh(y + .18, [r['controlled_place'] for r in rows], height=.35, label='controlled place')
        axes[0].set_yticks(y, [r['sequence'] for r in rows])
        axes[0].set_xlim(0, 8.5)
        axes[0].set_xlabel('Episodes / 8 fixed starts')
        axes[0].legend()
        axes[1].barh(y, [r['reference_position_rmse_m'] * 100 for r in rows])
        axes[1].set_xlabel('Object reference position RMSE (cm)')
        axes[0].invert_yaxis()
        fig.suptitle('Official DExplore actor: corrected12 reference probe')
        fig.tight_layout()
        fig.savefig(args.directory / 'comparison.png', dpi=140)
        plt.close(fig)
    print('\n'.join(table))


if __name__ == '__main__':
    main()
