"""Aggregate the predeclared frozen-model holdout panel; no new model selection."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import yaml

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path.insert(0, str(ROOT / 'src/task/consequence-evaluator/src'))
from consequence_evaluator.gate1 import episode_outcome


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gates(row, teacher, oracle):
    return dict(hold45=row['hold45'] >= 6, placement=row['success'] >= 6,
        clipping=row['clipping_rate'] < .01, duration=row['median_held'] >= .8 * oracle['median_held'],
        reference_position=row['reference_position_rmse_m'] <= 1.5 * teacher['reference_position_rmse_m'],
        reference_rotation=row['reference_rotation_mean_rad'] <= 1.5 * teacher['reference_rotation_mean_rad'] + .1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--card', type=Path, required=True)
    p.add_argument('--runs', type=Path, nargs=2, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    out = args.output.resolve()
    if out.exists() or ROOT / 'outputs/trajectory-policy' not in out.parents:
        raise ValueError('fresh owned summary output required')
    first_manifest = json.loads((args.runs[0] / 'manifest.json').read_text())
    card_path = args.card.resolve()
    if str(card_path) != first_manifest['validation_card']:
        raise ValueError('exact recorded protocol card required')
    protocol_sha = first_manifest['input_sha256'][str(card_path)]
    protocol_bytes = card_path.read_bytes()
    if hashlib.sha256(protocol_bytes).hexdigest() != protocol_sha:
        protocol_bytes = subprocess.check_output(['git', 'show', first_manifest['git_commit'] + ':'
            + str(card_path.relative_to(ROOT))], cwd=ROOT)
    if hashlib.sha256(protocol_bytes).hexdigest() != protocol_sha:
        raise ValueError('cannot authenticate original frozen protocol')
    protocol = yaml.safe_load(protocol_bytes.decode().split('---', 2)[1])
    reports, hashes, found, initial_states = [], {}, [], []
    for folder in args.runs:
        m = json.loads((folder / 'manifest.json').read_text())
        r = json.loads((folder / 'result.json').read_text())
        a = json.loads((folder / 'audit.json').read_text())
        if (m['status'] != 'COMPLETED' or not a['pass'] or m.get('dagger_rollin')
                or m.get('intervention_beta') != 0 or m['experiment_id'] != protocol['validation_id']
                or m.get('seed_pool') != 'validation.holdout' or m['actor_period'] != 1
                or m['roles'] != ['teacher'] * 4 + ['gt_tau_R'] * 4 + ['tau_only'] * 8 + ['joint'] * 8):
            raise ValueError('audited frozen autonomous holdout execution required')
        for path, digest in a['evidence_sha256'].items():
            if sha(path) != digest:
                raise ValueError('audited measurement/model drift: ' + path)
        for path, digest in m['input_sha256'].items():
            if Path(path) == card_path and digest == protocol_sha:
                continue  # Results may be appended; frozen protocol is authenticated above.
            if sha(path) != digest:
                raise ValueError('frozen execution input drift: ' + path)
        fit = Path(m['actor_fit'])
        for name, digest in protocol['checkpoint_sha256'].items():
            if sha(fit / (name + '.pt')) != digest:
                raise ValueError('frozen actor differs')
        if sha(m['retargeter']) != protocol['retargeter_sha256'] or sha(m['reference']) != protocol['reference_sha256']:
            raise ValueError('frozen R/reference differs')
        if sha(folder / 'trajectory.npz') != m['trajectory_sha256']:
            raise ValueError('trajectory identity differs')
        with np.load(folder / 'trajectory.npz', allow_pickle=False) as f:
            data = {k: f[k] for k in ('q', 'obj', 'hand', 'velocity', 'surface_gap', 'support_gap', 'table_footprint', 'clipped')}
        with np.load(m['reference'], allow_pickle=False) as f:
            reference = f['obj']
        outcomes = []
        for env in range(24):
            outcome = episode_outcome(dict(object_pose=data['obj'][:, env],
                object_velocity=data['velocity'][:, env], surface_gap=data['surface_gap'][:, env],
                support_gap=data['support_gap'][:, env], table_footprint=data['table_footprint'][:, env]))
            if any(outcome[key] != r['outcomes'][env][key] for key in outcome):
                raise ValueError('independent physical outcome differs')
            outcomes.append(outcome)
        rows = {}
        for role in ('teacher', 'gt_tau_R', 'tau_only', 'joint'):
            ids = [i for i, name in enumerate(m['roles']) if name == role]
            poses = data['obj'][:, ids]
            error = poses[..., :3, 3] - reference[:, None, :3, 3]
            relative = np.matmul(reference[:, None, :3, :3].swapaxes(-1, -2), poses[..., :3, :3])
            angle = np.arccos(np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1) / 2, -1, 1))
            row = dict(hold45=sum(outcomes[i]['maximum_held_frames'] >= 45 for i in ids),
                success=sum(outcomes[i]['success'] for i in ids),
                median_held=float(np.median([outcomes[i]['maximum_held_frames'] for i in ids])),
                clipping_rate=float(data['clipped'][:, ids].mean()),
                reference_position_rmse_m=float(np.sqrt(np.mean(np.sum(error ** 2, axis=-1)))),
                reference_rotation_mean_rad=float(angle.mean()))
            if any(abs(value - r['summary'][role][key]) > 1e-6 for key, value in row.items()):
                raise ValueError('independent group metric differs')
            rows[role] = row
        calibrated = all(rows[role]['hold45'] >= 3 and rows[role]['success'] >= 3 for role in ('teacher', 'gt_tau_R'))
        student_gates = {role: gates(rows[role], rows['teacher'], rows['gt_tau_R']) for role in ('tau_only', 'joint')}
        initial = {key: float(np.max(np.abs(data[key][0, 8:16] - data[key][0, 16:24]))) for key in ('q', 'obj', 'hand', 'velocity')}
        initial_states.append({key: data[key][0].copy() for key in ('q', 'obj', 'hand', 'velocity')})
        reports.append(dict(seed=m['seed'], run_id=m['run_id'], calibrated=calibrated,
            groups=rows, gates=student_gates, initial_offline_dagger_max_difference=initial))
        found.append(m['seed'])
        hashes.update({str((folder / name).resolve()): sha(folder / name)
                       for name in ('manifest.json', 'result.json', 'trajectory.npz', 'audit.json')})
    if sorted(found) != sorted(protocol['validation_seeds']):
        raise ValueError('exact predeclared seed panel required')
    valid = all(r['calibrated'] for r in reports)
    passed = [all(r['gates']['joint'].values()) for r in reports]
    verdict = ('SUPPORTED' if all(passed) else 'REFUTED' if not any(passed) else 'INCONCLUSIVE') if valid else 'INCONCLUSIVE'
    repeated_advantage = verdict == 'SUPPORTED' and all(
        r['groups']['joint']['reference_position_rmse_m'] <= .8 * r['groups']['tau_only']['reference_position_rmse_m']
        and r['groups']['joint']['median_held'] >= 1.25 * r['groups']['tau_only']['median_held'] for r in reports)
    out.mkdir()
    (out / 'protocol_frozen.md').write_bytes(protocol_bytes)
    hashes[str(card_path)] = protocol_sha
    result = dict(schema='ref2dex.reference-validation-summary.v1', experiment_id=protocol['validation_id'],
        status='COMPLETED', conclusion=verdict, scope='two prespecified holdout launches; fixed models and single reference',
        repeated_descriptive_advantage=bool(repeated_advantage), seeds=reports,
        input_sha256=hashes, summary_tool_sha256=sha(__file__),
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    result['cross_seed_initial_max_difference'] = {
        key: float(np.max(np.abs(initial_states[0][key] - initial_states[1][key])))
        for key in initial_states[0]}
    (out / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    os.environ['MPLCONFIGDIR'] = str(ROOT / 'tmp/trajectory-policy/competent-pd/matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    for offset, (role, label) in enumerate((('tau_only', 'Offline continuation'), ('joint', 'PD DAgger'))):
        positions = np.arange(2) + (offset - .5) * .35
        values = [r['groups'][role] for r in reports]
        for ax, metric in zip(axes, ('success', 'median_held', 'reference_position_rmse_m')):
            ax.bar(positions, [r[metric] for r in values], width=.35, label=label)
    for ax, label in zip(axes, ('Controlled placements / 8', 'Median held frames', 'Object reference RMSE (m)')):
        ax.set_ylabel(label)
        ax.set_xticks(np.arange(2))
        ax.set_xticklabels(['Seed ' + str(r['seed']) for r in reports])
        ax.grid(axis='y', alpha=.2)
    axes[0].set_ylim(0, 8.5)
    axes[0].legend(fontsize=8)
    fig.suptitle('Fixed models, independent launches, teacher intervention off')
    fig.tight_layout()
    fig.savefig(out / 'comparison.png', dpi=160)
    plt.close(fig)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
