"""Audit saved paired resets and compare execution outcomes, never pool old runs."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / 'src/task/wam-rl-policy-v3/src'))
from wam_rl_policy_v3.physical_safety import physical_safety


def death(row):
    if row['physical_abort']:
        return row['physical_abort']
    if row['final_success']:
        return 'success'
    achieved = row['stage_achieved']
    if not achieved['contact']:
        return 'contact_not_certified'
    if not achieved['lift']:
        return 'contact_without_lift'
    if not achieved['hold']:
        return 'lift_without_continuous_hold'
    if not row['stage_final']['hold']:
        return 'hold_lost_at_end'
    return 'hold_without_goal_arrival'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    reg = json.loads((run / 'registration.json').read_text())
    rows = {}
    manifests = [json.loads(p.read_text()) for p in sorted(run.glob('g[456]/manifest.json'))]
    if len(manifests) != 3 or any(m['status'] != 'COMPLETED' for m in manifests):
        raise ValueError('all three workers must complete before final comparison')
    for manifest in manifests:
        for row in manifest['runs']:
            key = (row['trial'], row['arm'])
            if key in rows:
                raise ValueError('duplicate reset/arm')
            folder = run / ('g' + str(manifest['gpu'])) / row['id']
            rows[key] = dict(row, folder=str(folder), death=death(row))
    if len(rows) != 40:
        raise ValueError('exactly forty completed episodes required')
    initial_checks = []
    stages = ['approach', 'contact', 'lift', 'hold', 'release']
    pairs = []
    for job in reg['jobs']:
        old, new = (rows[(job['id'], arm)] for arm in ['OLD', 'NEW'])
        with np.load(Path(old['folder']) / 'initial.npz') as a, np.load(Path(new['folder']) / 'initial.npz') as b:
            checks = {key: bool(np.array_equal(a[key], b[key])) for key in a.files}
            if not all(checks.values()):
                raise ValueError('paired initial mismatch: ' + job['id'])
            # Original saved reset, not just the reference approximation.
            original = ROOT / 'outputs/wam-rl-policy-v3/grasp-v11-coverage40-20261022-r1' / ('g' + str(4 + job['worker'])) / (job['id'] + '-A1') / 'initial.npz'
            with np.load(original) as prior:
                prior_checks = {key: bool(np.array_equal(a[key], prior[key])) for key in a.files}
            if not all(prior_checks.values()):
                raise ValueError('existing-panel initial mismatch: ' + job['id'])
        initial_checks.append(dict(trial=job['id'], paired=checks, original=prior_checks))
        for row in (old, new):
            folder = Path(row['folder'])
            native = json.loads((folder / 'native-audit.json').read_text())
            consumer = json.loads((folder / 'consumer.json').read_text())
            stage_log = json.loads((folder / 'stages.json').read_text())
            row['consumer_phase_controls'] = dict(Counter(c['phase'] for c in consumer))
            row['plan_mode_controls'] = dict(Counter(c['plan_mode'] for c in consumer if c['tau_valid']))
            row['first_stage_time_s'] = stage_log[-1]['first_time_s']
            source = Path(next(r['source'] for r in json.loads((ROOT / 'outputs/wam-rl-policy-v3/inspire-teacher-rollout-audit-20261020-r3/manifest.json').read_text())['records'] if r['sequence'] == job['sequence']))
            with np.load(source / 'canonical.npz') as z:
                points = z['points'].copy()
            with np.load(ROOT / 'outputs/wam-rl-policy-v3/inspire-teacher-rollout-audit-20261020-r3/sequences' / job['sequence'] / 'admission.npz') as z:
                bounds = z['table_bounds'].copy()
            with np.load(folder / 'trajectory.npz') as z:
                goal = z['objects'][0, 0, :3].astype(float) + [0., 0., .08]
                error = float(np.linalg.norm(z['objects'][-1, 0, :3] - goal))
                if abs(error - row['final_goal_error_m']) > 1e-8:
                    raise ValueError('raw final goal error differs')
                first = None
                for tick in range(len(z['q'])):
                    safe = physical_safety({k: z[k][tick] for k in ('objects', 'q', 'dq', 'table_pose', 'object_contact_force')}, 0, points, bounds, native['physical_joint_limits'])
                    if safe['abort']:
                        first = dict(tick=tick, **safe)
                        break
                if (first['reason'] if first else None) != row['physical_abort']:
                    raise ValueError('raw safety replay differs: ' + row['id'])
                if first and first['tick'] != row['controls']:
                    raise ValueError('first safety event not at termination')
                if len(z['action']) != row['controls'] or len(z['q']) != row['controls'] + 1:
                    raise ValueError('raw clock mismatch')
                row['first_limit'] = None if not first or not first['joint_violation_indices'] else dict(
                    tick=first['tick'], indices=first['joint_violation_indices'],
                    names=[native['native_dof_names'][i] for i in first['joint_violation_indices']])
                row['highest_stage'] = next((s for s in reversed(stages) if row['stage_achieved'][s]), 'none')
        pairs.append(dict(trial=job['id'], sequence=job['sequence'], object=job['object'],
                          old_success=old['final_success'], new_success=new['final_success'],
                          old_death=old['death'], new_death=new['death'],
                          old_first_limit=old['first_limit'], new_first_limit=new['first_limit'],
                          old_highest_stage=old['highest_stage'], new_highest_stage=new['highest_stage'],
                          old_first_stage_time_s=old['first_stage_time_s'], new_first_stage_time_s=new['first_stage_time_s'],
                          old_goal_error_m=old['final_goal_error_m'], new_goal_error_m=new['final_goal_error_m'],
                          delta_goal_error_m=new['final_goal_error_m']-old['final_goal_error_m']))
        if old['final_success'] != new['final_success']:
            with np.load(Path(old['folder']) / 'trajectory.npz') as a, np.load(Path(new['folder']) / 'trajectory.npz') as b:
                length = min(len(a['tau']), len(b['tau']))
                differences = np.max(np.abs(a['tau'][:length] - b['tau'][:length]), axis=(1, 2, 3))
                where = np.flatnonzero(differences > 0)
                tick = int(where[0]) if len(where) else None
                if tick is not None:
                    pairs[-1]['first_tau_difference'] = dict(
                        tick=tick, tau8_max_abs_m=float(differences[tick]),
                        state_max_abs={k:float(np.max(np.abs(a[k][tick]-b[k][tick]))) for k in ('q','dq','hand','objects')},
                        R_action_delta=(b['adapter_raw_action'][tick] - a['adapter_raw_action'][tick]).tolist())
    summary = {}
    for arm in ['OLD', 'NEW']:
        selected = [r for (trial, a), r in rows.items() if a == arm]
        summary[arm] = dict(n=len(selected), successes=sum(r['final_success'] for r in selected),
                            safety_aborts=sum(r['physical_abort'] is not None for r in selected),
                            deaths=dict(Counter(r['death'] for r in selected)),
                            first_limit_sets=dict(Counter(','.join(map(str, r['first_limit']['indices'])) for r in selected if r['first_limit'])),
                            first_limit_joints=dict(Counter(name for r in selected if r['first_limit'] for name in r['first_limit']['names'])),
                            stage_achieved={s: sum(r['stage_achieved'][s] for r in selected) for s in stages},
                            highest_stage=dict(Counter(r['highest_stage'] for r in selected)),
                            consumer_phase_controls=dict(sum((Counter(r['consumer_phase_controls']) for r in selected), Counter())),
                            plan_mode_controls=dict(sum((Counter(r['plan_mode_controls']) for r in selected), Counter())),
                            final_goal_error_m_quantiles=dict(zip(['min','p25','median','p75','p90','max'], np.quantile([r['final_goal_error_m'] for r in selected], [0,.25,.5,.75,.9,1]).tolist())),
                            controls=sum(r['controls'] for r in selected), wam_calls=sum(r['wam_calls'] for r in selected))
    result = dict(status='COMPLETED', experiment_id=reg['experiment_id'], arms=summary,
                  gained=sum(not p['old_success'] and p['new_success'] for p in pairs),
                  lost=sum(p['old_success'] and not p['new_success'] for p in pairs),
                  unchanged_success=sum(p['old_success'] and p['new_success'] for p in pairs),
                  pairs=pairs, initial_checks=initial_checks,
                  condition_boundaries={str(m['gpu']): {r['arm']:r['condition_boundary'] for r in m['runs']} for m in manifests},
                  raw_safety_replays=40, learning_updates=sum(m['learning_updates'] for m in manifests))
    discordant = result['gained'] + result['lost']
    result['paired_exact_p_two_sided'] = (min(1., 2 * sum(math.comb(discordant, k) for k in range(min(result['gained'], result['lost']) + 1)) / 2**discordant)
                                        if discordant else 1.)
    result['paired_goal_error_delta_quantiles_m'] = np.quantile([p['delta_goal_error_m'] for p in pairs], [0, .25, .5, .75, 1]).tolist()
    (run / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('pairs','initial_checks')}, indent=2))


if __name__ == '__main__':
    main()
