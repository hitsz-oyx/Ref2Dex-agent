"""Recompute frozen PW-on/off physical gates and the predeclared pilot decision."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path.insert(0, str(ROOT / 'src/task/consequence-evaluator/src'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execution', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    execution, out = a.execution.resolve(), a.output.resolve()
    if out.exists() or ROOT / 'outputs/trajectory-policy' not in out.parents:
        raise ValueError('fresh task summary output required')
    os.environ.update(MPLCONFIGDIR=str(ROOT / 'tmp/trajectory-policy/pw-critic/matplotlib'))
    from consequence_evaluator.gate1 import episode_outcome
    m = json.loads((execution / 'manifest.json').read_text())
    original = json.loads((execution / 'result.json').read_text())
    audit = json.loads((execution / 'audit.json').read_text())
    if m['status'] != 'COMPLETED' or m.get('pw_mode') != 'evaluate' or m['seed'] != 189 or not audit['pass']:
        raise ValueError('completed audited fixed PW evaluation required')
    if m['training_allowed'] or m['exploration_m'] != 0 or m['intervention_beta'] != 0:
        raise ValueError('evaluation must have no training, noise or expert interventions')
    for path, digest in audit['evidence_sha256'].items():
        if sha(path) != digest:
            raise ValueError('execution audit evidence drift: ' + path)
    # Authenticate the pre-execution protocol from the actual producer commit.
    card = TASK / 'docs/experiments/probes/P-20261011-pw-critic-policy.md'
    relative = str(card.relative_to(ROOT))
    protocol = subprocess.check_output(['git', 'show', m['git_commit'] + ':' + relative], cwd=ROOT)
    if hashlib.sha256(protocol).hexdigest() != m['input_sha256'][str(card)]:
        raise ValueError('frozen protocol Git blob differs')
    with np.load(execution / 'trajectory.npz', allow_pickle=False) as f:
        trace = {k: f[k].copy() for k in f.files}
    with np.load(m['reference'], allow_pickle=False) as f:
        ref = {k: f[k].copy() for k in f.files}
    summaries, gates = {}, {}
    for role in ('teacher', 'gt_tau_R', 'tau_only', 'joint'):
        ids = [i for i, r in enumerate(m['roles']) if r == role]
        rows = []
        for env in ids:
            rows.append(episode_outcome(dict(object_pose=trace['obj'][:, env],
                object_velocity=trace['velocity'][:, env], surface_gap=trace['surface_gap'][:, env],
                support_gap=trace['support_gap'][:, env], table_footprint=trace['table_footprint'][:, env])))
        pos = trace['obj'][:, ids][:, :, :3, 3] - ref['obj'][:, None, :3, 3]
        relative_rotation = ref['obj'][:, None, :3, :3].swapaxes(-1, -2) @ trace['obj'][:, ids][:, :, :3, :3]
        rotation = np.arccos(np.clip((np.trace(relative_rotation, axis1=-2, axis2=-1) - 1) / 2, -1, 1))
        summaries[role] = dict(episodes=len(ids), hold45=sum(row['maximum_held_frames'] >= 45 for row in rows),
            success=sum(row['success'] for row in rows), median_held=float(np.median([row['maximum_held_frames'] for row in rows])),
            clipping_rate=float(trace['clipped'][:, ids].mean()),
            reference_position_rmse_m=float(np.sqrt(np.square(pos).sum(-1).mean())),
            reference_rotation_mean_rad=float(rotation.mean()))
        for key, value in summaries[role].items():
            if abs(value - original['summary'][role][key]) > 1e-6:
                raise ValueError('physical summary recomputation differs: ' + role + '/' + key)
    calibration = all(summaries[r]['hold45'] >= 3 and summaries[r]['success'] >= 3 for r in ('teacher', 'gt_tau_R'))
    for role in ('tau_only', 'joint'):
        s = summaries[role]
        gates[role] = dict(hold=s['hold45'] >= 6, placement=s['success'] >= 6, clip=s['clipping_rate'] < .01,
            duration=s['median_held'] >= .8 * summaries['gt_tau_R']['median_held'],
            position=s['reference_position_rmse_m'] <= 1.5 * summaries['teacher']['reference_position_rmse_m'],
            rotation=s['reference_rotation_mean_rad'] <= 1.5 * summaries['teacher']['reference_rotation_mean_rad'] + .1)
    off, on = summaries['tau_only'], summaries['joint']
    gain = 1 - on['reference_position_rmse_m'] / off['reference_position_rmse_m']
    competence = all(all(row.values()) for row in gates.values())
    benefit = competence and gain >= .05 and on['success'] >= off['success'] and on['median_held'] >= off['median_held']
    status = ('PROMISING' if benefit else 'UNPROMISING') if calibration else 'UNCLEAR'
    fit = Path(m['actor_fit'])
    learning = json.loads((fit / 'result.json').read_text())
    result = dict(status=status, calibration_pass=calibration, competence_pass=competence,
        predeclared_benefit_pass=benefit, position_relative_gain=gain, summary=summaries, gates=gates,
        probe_id='P-20261011-pw-critic-policy',
        scope='one fixed-reference/initialization pilot, not a formal PW benefit or route refutation',
        producer_commit=m['git_commit'],
        evidence_sha256={str(path): sha(path) for path in (execution / 'manifest.json', execution / 'trajectory.npz',
            execution / 'audit.json', fit / 'result.json', fit / 'audit.json')})
    out.mkdir(parents=True)
    (out / 'protocol_frozen.md').write_bytes(protocol)
    (out / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    seconds = np.arange(543) / 30
    for role, label in (('teacher', 'Teacher'), ('gt_tau_R', 'GT trajectory / R'),
                        ('tau_only', 'PW-off learned actor'), ('joint', 'PW-on learned actor')):
        ids = [i for i, r in enumerate(m['roles']) if r == role]
        rise = trace['obj'][:, ids][:, :, 2, 3] - trace['obj'][0, ids, 2, 3]
        position = np.linalg.norm(trace['obj'][:, ids][:, :, :3, 3] - ref['obj'][:, None, :3, 3], axis=-1)
        held = (trace['surface_gap'][:, ids] <= .01) & (rise >= .03) & ~(trace['table_footprint'][:, ids]
            & (np.abs(trace['support_gap'][:, ids]) <= .02))
        axes[0, 0].plot(seconds, position.mean(1), label=label)
        axes[0, 1].plot(seconds, held.mean(1), label=label)
        axes[1, 0].plot(seconds, rise.mean(1), label=label)
    for ax, label in zip((axes[0, 0], axes[0, 1], axes[1, 0]),
                         ('Reference position error (m)', 'Held fraction', 'Object rise (m)')):
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(label)
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    axes[1, 1].bar(['PW-off', 'PW-on'], [learning['reports'][name]['validation_normalized_mse'] for name in ('tau_only', 'joint')])
    axes[1, 1].set_ylabel('Held-out Q normalized MSE')
    axes[1, 1].set_title('Predictive diagnostic; physical decision above')
    fig.suptitle('Frozen PW critic policy pilot: ' + status)
    fig.tight_layout()
    fig.savefig(out / 'comparison.png', dpi=150)
    plt.close(fig)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
