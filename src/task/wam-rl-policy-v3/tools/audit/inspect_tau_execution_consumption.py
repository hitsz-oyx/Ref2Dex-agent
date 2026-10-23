"""Inspect saved discordant pairs at their first live tau consumption; no inference."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[5]


def read(path):
    return json.loads(path.read_text())


def first_difference(a,b):
    n=min(len(a),len(b))
    difference=np.abs(a[:n].astype(float)-b[:n].astype(float)).reshape(n,-1).max(axis=1)
    hits=np.flatnonzero(difference>1e-7)
    return None if not len(hits) else int(hits[0])


def inspect(run):
    manifests=[(p.parent,read(p)) for p in sorted(run.glob('g*/manifest.json'))]
    rows={(r['trial'],r['arm']):(folder/r['id'],r) for folder,m in manifests for r in m['runs']}
    frozen=manifests[0][0]/'runtime-code'
    sys.path.insert(0,str(frozen/'src/task/wam-rl-policy-v3/src'))
    from wam_rl_policy_v3.pose_consumer import pose7,reduce_to_wrist_plan
    catalog=read(ROOT/'outputs/wam-rl-policy-v3/inspire-teacher-rollout-audit-20261020-r3/manifest.json')['records']
    results=[]
    for job in read(run/'registration.json')['jobs']:
        keys=[(job['id'],arm) for arm in ('OLD','NEW')]
        if not all(key in rows for key in keys):
            continue
        if rows[keys[0]][1]['final_success']==rows[keys[1]][1]['final_success']:
            continue
        oldfolder,old=rows[keys[0]];newfolder,new=rows[keys[1]]
        source=Path(next(c['source'] for c in catalog if c['sequence']==job['sequence']))
        with np.load(source/'canonical.npz') as z:
            points=z['points'].copy()
        with np.load(oldfolder/'trajectory.npz') as za,np.load(newfolder/'trajectory.npz') as zb:
            first_valid=[np.flatnonzero(z['tau_valid']) for z in (za,zb)]
            if any(not len(t) for t in first_valid):
                results.append(dict(trial=job['id'],reason='one arm has no valid tau'));continue
            t=max(int(t[0]) for t in first_valid)
            history_errors={key:float(np.abs(za[key][max(t-3,0):t+1].astype(float)-zb[key][max(t-3,0):t+1].astype(float)).max()) for key in ('q','dq','objects','hand')}
            differences={key:first_difference(za[key],zb[key]) for key in ('tau','adapter_raw_action','action','q','hand','objects')}
            arms={}
            for arm,z,folder,row in [('OLD',za,oldfolder,old),('NEW',zb,newfolder,new)]:
                plan=reduce_to_wrist_plan(z['hand'][t,0],pose7(z['palm_pose'][t,0]),z['forecast_hand'][t],pose7(z['objects'][t,0]),points)
                consumers=read(folder/'consumer.json');stages=read(folder/'stages.json')
                delta=z['forecast_hand'][t]-z['hand'][t,0]
                arms[arm]=dict(first_valid_tick=int(first_valid[0 if arm=='OLD' else 1][0]),
                    first_consumed_first8_mean_displacement_m=float(np.linalg.norm(z['tau'][t],axis=-1).mean()),
                    first_forecast_mean_displacement_m_by_segment={label:float(np.linalg.norm(delta[start:end],axis=-1).mean()) for label,start,end in [('frames1_4',0,4),('frames5_8',4,8),('frames9_24',8,24)]},
                    first_plan=dict(mode=plan.mode,contact_index=plan.contact_index,eta_s=plan.eta_s,grasp_pose_object=plan.local_poses[-1].tolist()),
                    phase_controls=dict(Counter(c['phase'] for c in consumers)),first_stage_times_s=stages[-1]['first_time_s'],
                    reach_filter_controls=row['reach_filter_controls'],final_success=row['final_success'],physical_abort=row['physical_abort'],final_goal_error_m=row['final_goal_error_m'])
            results.append(dict(trial=job['id'],sequence=job['sequence'],first_common_tau_tick=t,
                measured_history_max_abs_differences=history_errors,identical_measured_input=not any(history_errors.values()),
                first_nontrivial_difference_tick=differences,
                first_R_action_max_abs_difference=float(np.abs(za['adapter_raw_action'][t]-zb['adapter_raw_action'][t]).max()),
                first_executed_action_max_abs_difference=float(np.abs(za['action'][t]-zb['action'][t]).max()),arms=arms))
    return dict(scope='saved discordant-pair descriptive inspection; no model call, physics, or component intervention',
        consumption=dict(retargeter='frames1_8 sparse11 displacement -> 686D observation -> fixed R',
                         wrist_plan='full24 sparse11 -> rigid-base Kabsch poses -> last predicted precontact grasp pose',
                         update_phases=['approach','close'],later_phases='measured lift/hold servo; no new tau query'),
        completed_pairs_inspected=sum(all((j['id'],a) in rows for a in ('OLD','NEW')) for j in read(run/'registration.json')['jobs']),
        discordant_pairs=results,
        limitations=['displacement norms are not prediction errors','post-divergence trajectories are not matched-state model ablations',
                     'this inspection cannot separate first8/R and full24/keypose causal contributions'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    result=inspect(args.run.resolve())
    with args.output.open('x') as output:json.dump(result,output,indent=2);output.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
