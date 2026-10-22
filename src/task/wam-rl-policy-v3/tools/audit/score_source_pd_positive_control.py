"""Recompute production source-PD physical evidence; keep the three chains distinct."""
import argparse,importlib.util,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT/'src/task/wam-rl-policy-v3/src'))


def longest_seconds(mask,times):
    start=None;longest=0.
    for valid,now in zip(mask,times):
        if valid:
            if start is None:start=now
            longest=max(longest,float(now-start))
        else:start=None
    return longest


def score(run,output):
    from scipy.spatial.transform import Rotation
    from wam_rl_policy_v3.task_feedback import FeedbackObserver
    from wam_rl_policy_v3.task_scheduler import TaskScheduler,SCHEDULER_SCHEMA
    from wam_rl_policy_v3.task_spec import TaskSpec
    from wam_rl_policy_v3.robot_adapter import InspireAdapter
    from wam_rl_policy_v3.teacher_quality import table_support
    manifest=json.loads((run/'manifest.json').read_text());declarations=Path(manifest['source_manifest'])
    rows={r['sequence']:r for r in json.loads(declarations.read_text())['records']}
    path=run/'before-task-scheduler.py'
    if not path.is_file():raise ValueError('explicit old scheduler snapshot required')
    loader=importlib.util.spec_from_file_location('wam_rl_policy_v3._source_pd_before',path)
    old=importlib.util.module_from_spec(loader);sys.modules[loader.name]=old;loader.loader.exec_module(old)
    results=[];details={}
    with np.load(run/'trajectory.npz') as capture,np.load(run/'controls.npz') as controls:
        for i,name in enumerate(manifest['sequence_order']):
            row=rows[name];side=declarations.parent/row['sidecar']
            spec=TaskSpec.from_packet(json.loads((side/'raw-task.json').read_text()))
            with np.load(Path(row['source'])/'canonical.npz') as z:points=z['points'].copy()
            with np.load(side/'admission.npz') as z:bounds=z['table_bounds'].copy()
            obj=capture['objects'][:,i];poses=np.repeat(np.eye(4)[None],len(obj),0)
            poses[:,:3,:3]=Rotation.from_quat(obj[:,3:7]).as_matrix();poses[:,:3,3]=obj[:,:3]
            observer=FeedbackObserver(points);schedulers={'before':old.TaskScheduler(spec,points),'after':TaskScheduler(spec,points)}
            traces={key:[] for key in schedulers};terminal={key:None for key in schedulers};last_pointer={key:0 for key in schedulers};retractions={key:0 for key in schedulers}
            lift_hold=[];feedback_rows=[];physical_abort=manifest['aborts'][i];abort_tick=None if physical_abort is None else physical_abort['tick']
            for t in range(len(obj)):
                now=float(capture['tick'][t,i])/30
                contact=capture['contact_forces'][t,i];valid=bool(capture['contact_force_valid'][t,i]);hand=InspireAdapter.canonical_hand(capture['hand'][t,i])
                supported=bool(table_support(poses[t:t+1],points,capture['table_pose'][t,i],bounds,capture['object_contact_force'][t:t+1,i])[0][0])
                safe=abort_tick is None or t<abort_tick
                fb=observer.update(poses[t],hand,obj[t,7:13],now,contact_forces=contact,force_valid=valid,supported=supported,workspace_ok=safe)
                held=bool(safe and obj[t,2]-obj[0,2]>=.05 and fb.grasp_established and not fb.grasp_lost and valid and not supported and (np.linalg.norm(contact,axis=-1)>=.1).sum()>=2)
                lift_hold.append(held);feedback_rows.append(dict(tick=t,grasp=fb.grasp_established,contact=fb.contact_observed,released=fb.released,supported=supported,stable=fb.stable,held=held))
                for key,scheduler in schedulers.items():
                    if terminal[key] is not None:continue
                    d=scheduler.update(poses[t],now,fb,hand_points=hand)
                    retractions[key]+=int(scheduler.path.pointer<last_pointer[key]);last_pointer[key]=scheduler.path.pointer
                    traces[key].append(dict(tick=t,phase=int(d.phase),pointer=int(scheduler.path.pointer),progress=float(d.path_progress),task_success=bool(d.task_complete),path_complete=bool(d.path_complete),failure=d.failure_reason,transitions=list(d.transitions)))
                    if d.terminated:terminal[key]=traces[key][-1]
            times=capture['tick'][:,i]/30;duration=longest_seconds(lift_hold,times)
            before=terminal['before'];after=terminal['after']
            result=dict(sequence=name,terminal_mode=spec.terminal_mode.value,source_endpoint_frame=spec.source_length-1,observed_last_tick=int(capture['tick'][-1,i]),
                endpoint_plus_one_second_observed=bool(capture['tick'][-1,i]>=spec.source_length-1+30),physical_or_control_abort=physical_abort,
                before_full_task_success=bool(before and before['task_success']),after_full_task_success=bool(after and after['task_success']),
                after_terminal=after,observed_path_retractions=retractions,maximum_path_progress={key:max(r['progress'] for r in trace) for key,trace in traces.items()},
                source_pd_active_controls=int(controls['active'][:,i].sum()),full_source_commands_executed=bool(controls['active'][:manifest['source_controls'][i],i].all()),
                longest_lift5cm_force_coupled_unsupported_s=duration,narrow_lift_hold_success=bool(duration>=1.-1e-10),
                independent_terminal_check='No GT-tau/WAM-tau executor success inherited from source-PD; full result is named TaskScheduler contract, narrow physical mask is separate.')
            results.append(result);details[name]=dict(feedback=feedback_rows,**traces)
    summary=dict(status='COMPLETED',scheduler_schema=SCHEDULER_SCHEMA,sample_unit='six fixed sequences; one production execution each, not held-out method evaluation',
        count=len(results),before_full_task_successes=sum(r['before_full_task_success'] for r in results),after_full_task_successes=sum(r['after_full_task_success'] for r in results),
        narrow_lift_hold_successes=sum(r['narrow_lift_hold_success'] for r in results),full_source_executed=sum(r['full_source_commands_executed'] for r in results),
        endpoint_plus_one_second_observed=sum(r['endpoint_plus_one_second_observed'] and r['physical_or_control_abort'] is None for r in results),
        full_task_qualification_pass=all(r['after_full_task_success'] and r['physical_or_control_abort'] is None for r in results),
        source_pd_saturation_rejections=manifest['source_pd_saturation_rejections'],applied_saturation_controls=manifest['clipped_envcontrols'],maximum_pd_error=manifest['maximum_pd_error'],
        records=results,limitations=['Original force-derived planned intent remains unverified.','Synthetic narrow lift does not certify full RELEASE or exact8cm.','Old/new share the same actual trace; no dynamic counterfactual policy claim.','FeedbackObserver shared with production; not independent contact ground truth.'])
    output.mkdir(parents=True,exist_ok=True);(output/'scores.json').write_text(json.dumps(summary,indent=2)+'\n');(output/'task-traces.json').write_text(json.dumps(details)+'\n')
    descriptor=dict(status='PREPARED_ONLY_NO_TRAINING',run=str(run),states=str(run/'trajectory.npz'),actions=str(run/'controls.npz'),chains_separate=True,
        sequence_order=manifest['sequence_order'],source_pd_complete_eligible=[r['sequence'] for r in results if r['physical_or_control_abort'] is None and r['full_source_commands_executed']],
        tau_contract='native11 world displacement; tau[0]=first future at1/30s,8steps=8/30s; first3/last8 invalid; never use live future at deployment',
        calibration_budget='S2 owns scaffolding; at most2000updates or1800s, no training launched by this tool',task_contract='Must be specified by S2 and match qualification; source-PD success cannot be inherited by GT-tau->R')
    (output/'calibration-inputs.json').write_text(json.dumps(descriptor,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('records','limitations')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();score(args.run.resolve(),args.output.resolve())
