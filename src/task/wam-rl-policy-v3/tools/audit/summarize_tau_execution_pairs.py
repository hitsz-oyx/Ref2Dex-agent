"""Read-only paired20 execution summary, with raw first-limit and initial checks."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[5]
FIELDS = ('q', 'dq', 'hand', 'objects', 'physical', 'tick', 'table_pose',
          'contact_force_valid', 'dof_force_valid')
STAGES = ('approach', 'contact', 'lift', 'hold', 'release')


def read(path):
    return json.loads(path.read_text())


def quantiles(values):
    return dict(zip(('min', 'q25', 'median', 'q75', 'max'),
                    np.quantile(values, [0, .25, .5, .75, 1]).tolist()))


def initial_difference(left, right):
    with np.load(left) as a, np.load(right) as b:
        if set(a.files) != set(b.files) or set(a.files) != set(FIELDS):
            raise ValueError('initial field set mismatch')
        errors = {}
        for name in FIELDS:
            if a[name].shape != b[name].shape or a[name].dtype != b[name].dtype:
                raise ValueError('initial shape/dtype mismatch: ' + name)
            errors[name] = float(np.abs(a[name].astype(np.float64)-b[name]).max())
        return errors


def classify(row):
    if row['final_success']:
        return 'success'
    if row['physical_abort'] == 'physical_joint_limit':
        return 'joint_abort'
    if row['physical_abort'] == 'physical_table_penetration':
        return 'table_penetration'
    if row['physical_abort']:
        return row['physical_abort']
    if not row['stage_achieved']['contact']:
        return 'contact_not_certified'
    if not row['stage_final']['hold']:
        return 'hold_not_retained'
    return 'goal_not_reached'


def raw_row(folder, row):
    audit = read(folder/'native-audit.json')
    limits = audit['physical_joint_limits']
    with np.load(folder/'trajectory.npz') as data:
        q = data['q'][:, 0]
        hits = np.asarray(limits['has_limits'], bool) & (
            (q < np.asarray(limits['lower'])-.10) |
            (q > np.asarray(limits['upper'])+.10))
        ticks = np.flatnonzero(hits.any(axis=1))
        first_tick = None if not len(ticks) else int(data['tick'][ticks[0], 0])
        indices = [] if not len(ticks) else np.flatnonzero(hits[ticks[0]]).tolist()
        names = [audit['native_dof_names'][i] for i in indices]
        controls = len(data['action'])
        if controls != row['controls'] or len(q) != controls+1:
            raise ValueError('raw control clock differs: ' + str(folder))
        if row['physical_abort']=='physical_joint_limit' and first_tick!=controls:
            raise ValueError('joint abort was not first limited state')
        if not np.array_equal(data['tick'][:,0],np.arange(controls+1)):
            raise ValueError('raw state clock is not consecutive')
        tau_valid = np.asarray(data['tau_valid'], bool)
        tau_norms = np.linalg.norm(data['tau'][tau_valid],axis=-1)
        tau_mean = None if not len(tau_norms) else float(tau_norms.mean())
    consumer = read(folder/'consumer.json')
    phases = Counter(x['phase'] for x in consumer)
    reached_close = bool(phases['close'] or phases['lift'] or phases['hold'])
    return dict(row, death=classify(row), folder=str(folder), first_limit_tick=first_tick,
                first_limit_indices=indices, first_limit_names=names,
                phase_controls=dict(phases), entered_close=reached_close,
                valid_tau_states=int(tau_valid.sum()), mean_consumed_first8_tau_norm_m=tau_mean)


def summarize(run, source):
    registration = read(run/'registration.json')
    jobs = registration['jobs']
    if len(jobs)!=20 or len({j['id'] for j in jobs})!=20:
        raise ValueError('expected exactly 20 frozen reset identities')
    source_jobs = read(source/'registration.json')['jobs'][:20]
    source_by_id = {job['id']: job for job in source_jobs}
    for a,b in zip(jobs,source_jobs):
        for key in ('id','sequence','reference','native_seed','model_seed'):
            if a[key]!=b[key]:
                raise ValueError('source identity changed: '+key)
    rows={};runtimes=[];environments=[]
    for manifest_path in sorted(run.glob('g*/manifest.json')):
        manifest=read(manifest_path)
        if manifest['status']!='COMPLETED':
            raise ValueError('worker incomplete: '+str(manifest_path))
        runtimes.append(manifest_path.parent/'runtime-code')
        environments.append({k:manifest.get(k) for k in ('worker','gpu','python','torch','cuda','git_commit','elapsed_s','environment_controls','maximum_pd_reconstruction_error','wam_seconds')})
        for row in manifest['runs']:
            key=(row['trial'],row['arm'])
            if key in rows:
                raise ValueError('duplicate pair arm')
            rows[key]=raw_row(manifest_path.parent/row['id'],row)
    if len(rows)!=40 or {arm for _,arm in rows}!={'OLD','NEW'}:
        raise ValueError('expected 20 completed pairs, OLD and NEW only')
    # Every executed controller / scorer / host is identical across GPU workers.
    runtime_mismatches=[]
    for first in runtimes[0].rglob('*.py'):
        rel=first.relative_to(runtimes[0])
        for other in runtimes[1:]:
            if not (other/rel).is_file() or first.read_bytes()!=(other/rel).read_bytes():
                runtime_mismatches.append(str(rel))
    if runtime_mismatches:
        raise ValueError('runtime byte mismatch: '+str(runtime_mismatches))
    paired=[]
    for job in jobs:
        old,new=rows[(job['id'],'OLD')],rows[(job['id'],'NEW')]
        ab=initial_difference(Path(old['folder'])/'initial.npz',Path(new['folder'])/'initial.npz')
        old_initial=source/f"g{4+source_by_id[job['id']]['worker']}"/(job['id']+'-A1')/'initial.npz'
        historical=initial_difference(Path(old['folder'])/'initial.npz',old_initial)
        if any(ab.values()) or any(historical.values()):
            raise ValueError('initial pair/history mismatch: '+job['id'])
        paired.append(dict(id=job['id'],sequence=job['sequence'],object=job['object'],native_seed=job['native_seed'],model_seed=job['model_seed'],initial_ab=ab,initial_vs_history=historical,
                           old=old,new=new,goal_error_delta_m=new['final_goal_error_m']-old['final_goal_error_m']))
    groups={}
    for arm in ('OLD','NEW'):
        group=[rows[(job['id'],arm)] for job in jobs]
        condition=[r['condition_boundary'] for r in group]
        if any(c['goal_valid'] or c['event_valid'] for c in condition):
            raise ValueError('unknown condition boundary violated')
        groups[arm]=dict(n=len(group),success=sum(r['final_success'] for r in group),
            physical_aborts=sum(r['physical_abort'] is not None for r in group),
            death_distribution=dict(Counter(r['death'] for r in group)),
            first_limit_joints=dict(Counter(name for r in group for name in r['first_limit_names'])),
            first_limit_sets=dict(Counter(','.join(r['first_limit_names']) or 'none' for r in group)),
            stage_achieved={s:sum(r['stage_achieved'][s] for r in group) for s in STAGES},
            highest_stage_distribution=dict(Counter(next((s for s in reversed(STAGES) if r['stage_achieved'][s]),'none') for r in group)),
            entered_close=sum(r['entered_close'] for r in group),
            final_goal_error_m=quantiles([r['final_goal_error_m'] for r in group]),
            controls=sum(r['controls'] for r in group),
            tau_steps=sorted(set(r['tau_step'] for r in group)),
            condition_goal_valid=0,condition_event_valid=0)
    net=groups['NEW']['success']-groups['OLD']['success']
    safety=groups['NEW']['physical_aborts']-groups['OLD']['physical_aborts']
    if net>0 and safety<=0:
        decision='PROMOTE_NEW_AND_SCHEDULE_COMPONENT_DECOMPOSITION'
    elif net<0:
        decision='STOP_EPE_AS_ACCEPTANCE_AND_INSPECT_CONSUMED_SEGMENTS'
    elif net>0:
        decision='SUCCESS_GAIN_WITH_SAFETY_REGRESSION_DO_NOT_PROMOTE'
    else:
        decision='NO_NET_EXECUTION_GAIN_LOCATE_STAGE_AND_DEATH_CHANGES'
    return dict(status='PASS',run_id=run.name,experiment_id=registration['experiment_id'],paired_n=20,episode_n=40,source_registration=str(source/'registration.json'),
        exact_initial_pairs=20,exact_historical_initials=20,identical_runtime_workers=len(runtimes),environments=environments,
        groups=groups,net_successes=net,safety_abort_delta=safety,
        paired_new_only_success=sum(p['new']['final_success'] and not p['old']['final_success'] for p in paired),
        paired_old_only_success=sum(p['old']['final_success'] and not p['new']['final_success'] for p in paired),
        paired_goal_error_delta_m=quantiles([p['goal_error_delta_m'] for p in paired]),
        decision=decision,pairs=paired,
        limitations=['exposed fixed20 Probe, no independent heldout or repeat robustness claim',
                     'table penetration is unchanged CAD-footprint safety proxy, not pair-contact proof',
                     'contact not certified is end-of-horizon failure, not a new instantaneous abort',
                     'exact stored initial arrays do not certify hidden PhysX state'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--source',type=Path,default=ROOT/'outputs/wam-rl-policy-v3/grasp-v11-coverage40-20261022-r1')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=summarize(args.run.resolve(),args.source.resolve())
    with args.output.open('x') as output:
        json.dump(result,output,indent=2);output.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='pairs'},indent=2))


if __name__=='__main__':
    main()
