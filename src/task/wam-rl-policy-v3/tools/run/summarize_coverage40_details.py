"""Saved-data geometry coverage metrics; do not relabel old fixed-15mm experiments."""
import argparse,collections,json
from pathlib import Path
import numpy as np


def limits(infos):
    result=dict(native_target_rejections=0,linear_rate_boundary=0,angular_rate_boundary=0,linear_rate_violations=0,angular_rate_violations=0)
    for info in infos:
        result['native_target_rejections']+=int(not info['native_target_admitted'])
        pose=info.get('relative_pose_servo')
        if pose:
            for name,step,rate in (('linear','linear_step_m','linear_rate_limit_mps'),('angular','angular_step_rad','angular_rate_limit_radps')):
                ceiling=pose[rate]/30.
                result[name+'_rate_boundary']+=int(pose[step]>=ceiling-1e-8)
                result[name+'_rate_violations']+=int(pose[step]>ceiling+1e-8)
    return result


def cause(row):
    if row['final_success']:return 'success'
    if row['physical_abort']:return row['physical_abort']
    if row['final_hold']:return 'held_but_off_goal'
    if row['stage_achieved']['hold']:return 'held_then_lost'
    if row['stage_achieved']['lift']:return 'lift_without_continuous_hold'
    if row['stage_achieved']['contact']:return 'contact_without_lift'
    return 'approach_without_contact'


def quantiles(values):return np.quantile(values,[0,.25,.5,.75,.9,1]).tolist() if values else None


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();reg=json.loads((a.run/'registration.json').read_text());rows=[]
    if reg['control_hz']!=30:raise ValueError('this frozen Inspire protocol requires30Hz')
    for g in (4,5,6):
        folder=a.run/('g%d'%g);m=json.loads((folder/'manifest.json').read_text())
        if m['status']!='COMPLETED' or m['protocol']!=reg['protocol']:raise ValueError('incomplete or mixed protocol')
        for row in m['runs']:
            trial=folder/row['id'];infos=json.loads((trial/'consumer.json').read_text());audit=json.loads((trial/'native-audit.json').read_text());bad=row['physical_safety_final']['joint_violation_indices'];last=infos[-1] if infos else {}
            if len(infos)!=row['controls']:raise ValueError('consumer/control clock mismatch')
            rows.append(dict(row,cause=cause(row),limit_counts=limits(infos),final_consumer_phase=last.get('phase'),joint_names=[audit['native_dof_names'][i] for i in bad],raw=str(trial/'trajectory.npz')))
    planned={j['id']+'-'+arm for j in reg['jobs'] for arm in j['order']}
    if len(rows)!=len(planned) or {r['id'] for r in rows}!=planned:raise ValueError('missing/duplicate episode')
    groups={}
    for subset in reg['subsets']:
        groups[subset]={}
        for arm in reg['arms']:
            records=[r for r in rows if r['subset']==subset and r['arm']==arm]
            if len(records)!=40:raise ValueError('n40 per arm/subset required')
            survivors=[r for r in records if r['controls']==reg['horizon_controls'] and r['physical_abort'] is None]
            joint=[r for r in records if r['physical_abort']=='physical_joint_limit']
            groups[subset][arm]=dict(n=40,success=sum(r['final_success'] for r in records),deaths=dict(collections.Counter(r['cause'] for r in records)),
                controls=sum(r['controls'] for r in records),actor_calls=sum(r['actor_calls'] for r in records),upstream_adapter_clip=sum(r['adapter_clipped_controls'] for r in records),final_canonical_clip=sum(r['final_clipped_controls'] for r in records),
                candidate_filter_controls=sum(r['reach_filter_controls'] for r in records),source_admission_rejections=sum(r.get('source_admission_failures',0) for r in records),
                success_with_source_rejection=sum(r['final_success'] and r.get('source_admission_failures',0)>0 for r in records),
                exact_native_projection_controls=None,
                **{k:sum(r['limit_counts'][k] for r in records) for k in records[0]['limit_counts']},
                hold_quantiles_s=quantiles([r['longest_hold_s'] for r in records]),all_terminal_error_quantiles_mm=quantiles([r['final_goal_error_m']*1000 for r in records]),
                full_horizon_survivors=len(survivors),full_horizon_error_quantiles_mm=quantiles([r['final_goal_error_m']*1000 for r in survivors]),
                error_over_width_quantiles=quantiles([r['final_goal_error_m']/r['geometry']['minimum_extent_m'] for r in records]),tolerance_range_mm=[min(r['geometry']['position_tolerance_m'] for r in records)*1000,max(r['geometry']['position_tolerance_m'] for r in records)*1000],
                geometry_stages_and15mm_success=sum(r['final_hold'] and r['goal_within15mm'] for r in records),joint_abort_phases=dict(collections.Counter(r['final_consumer_phase'] for r in joint)),joint_abort_names=dict(collections.Counter(n for r in joint for n in r['joint_names'])))
    violations=sum(sum(r['limit_counts'][k] for k in ('native_target_rejections','linear_rate_violations','angular_rate_violations')) for r in rows)
    report=dict(status='PASS' if not violations else 'FAIL',groups=groups,quantiles_order=[0,.25,.5,.75,.9,1],rows=rows,
        definitions=dict(upstream_adapter_clip='R/source proposed normalized action outside [-1,1], before intent reduction',final_canonical_clip='raw final controller output vs np.clip, >1e-6',candidate_filter_controls='geometry reachability path active; separate from clipping',native_target_rejections='final admitted command checked against all18 nominal PD target limits; not actual q guarantee',rate_boundary='servo intent increment norm on registered limit; NOT physical velocity or exact pre-limit clipping count',geometry_stages_and15mm_success='secondary 15mm endpoint with NEW geometry stages; NOT old v10 protocol'),
        telemetry_gaps=['exact native projection count is not saved, null does not mean zero','pre-norm-limiter command is not saved; rate-boundary count is not an exact clipping count','legacy goal_servo_saturated_controls component threshold is not used to measure the new SE3 norm limiter'],
        scope='Saved states and control logs only; no relabeling, no training. Raw independent replay/initial parity stored separately. A0 hybrid source is not GT-tau/R upper bound.')
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(status=report['status'],groups=groups),indent=2));return int(report['status']!='PASS')


if __name__=='__main__':raise SystemExit(main())
