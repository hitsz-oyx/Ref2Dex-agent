"""CPU replay physical captures through one explicitly frozen task package."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[5]
def longest(mask):
 x=np.r_[False,np.asarray(mask,bool),False];edges=np.flatnonzero(x[1:]!=x[:-1]);return int(max((b-a for a,b in zip(edges[::2],edges[1::2])),default=0))
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--package',type=Path,required=True);p.add_argument('--label',required=True);p.add_argument('--teacher',type=Path,required=True);p.add_argument('--source-pd',type=Path,required=True);p.add_argument('--declarations',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();sys.path.insert(0,str(a.package.resolve().parent))
 from wam_rl_policy_v3.task_runtime import TaskController
 from wam_rl_policy_v3.task_spec import TaskSpec
 from wam_rl_policy_v3.robot_adapter import InspireAdapter
 from wam_rl_policy_v3.task_feedback import FeedbackObserver
 from wam_rl_policy_v3.teacher_quality import table_support,measured_failures
 from wam_rl_policy_v3 import task_scheduler
 from scipy.spatial.transform import Rotation
 teacher=json.loads((a.teacher/'manifest.json').read_text());native=json.loads((a.source_pd/'manifest.json').read_text());names=native['sequence_order'];controls=np.load(a.source_pd/'controls.npz');pd=np.load(a.source_pd/'trajectory.npz');rows=[];scoring_traces={}
 for arm in ['teacher','source_pd']:
  for index,name in enumerate(names):
   side=a.declarations/'sequences'/name;packet=json.loads((side/'raw-task.json').read_text());spec=TaskSpec.from_packet(packet)
   with np.load(side/'admission.npz') as z:table=z['table_pose'].copy();bounds=z['table_bounds'].copy()
   record=next(r for r in teacher['records'] if r['sequence']==name)
   with np.load(a.teacher/record['path']/'canonical.npz') as z:points=z['points'].copy()
   if arm=='teacher':
    with np.load(a.teacher/record['path']/'trajectory.npz') as z:
     pose=z['obj'].copy();hand=z['hand'].copy();vel=z['velocity'].copy();contact=z['body_contact_forces'][:,teacher['observation_contract']['contact_body_ids']].copy();valid=z['contact_force_valid'].copy();object_force=z['object_contact_force'].copy();ticks=z['frame_tick'].copy()
    support,gap,footprint=table_support(pose,points,table,bounds,object_force);support_origin='geometric fixed-base initial table pose from original audit, fresh object force; fresh teacher table pose was not saved'
   else:
    obj=pd['objects'][:,index];pose=np.repeat(np.eye(4)[None],len(obj),0);pose[:,:3,:3]=Rotation.from_quat(obj[:,3:7]).as_matrix();pose[:,:3,3]=obj[:,:3];hand=pd['hand'][:,index];vel=obj[:,7:13];contact=pd['contact_forces'][:,index];valid=pd['contact_force_valid'][:,index];ticks=pd['tick'][:,index];object_force=pd['object_contact_force'][:,index];tables=pd['table_pose'][:,index]
    values=[table_support(pose[t:t+1],points,tables[t],bounds,object_force[t:t+1]) for t in range(len(pose))];support=np.array([v[0][0] for v in values]);gap=np.array([v[1][0] for v in values]);footprint=np.array([v[2][0] for v in values]);support_origin='fresh measured table pose and object net contact force, static table verified separately'
   support=support&valid;controller=TaskController(spec,points);observer=FeedbackObserver(points);task_rows=[];feedback_rows=[];last=None;transitions=[];terminal_tick=None
   for t in range(len(pose)):
    canonical=InspireAdapter.canonical_hand(hand[t]);fb=observer.update(pose[t],canonical,vel[t],float(ticks[t])/30,contact_forces=contact[t],force_valid=bool(valid[t]),supported=bool(support[t]));feedback_rows.append(fb)
    if terminal_tick is None:
     result=controller.observe(pose[t],canonical,vel[t],float(ticks[t])/30,contact_forces=contact[t],force_valid=bool(valid[t]),supported=bool(support[t]));last=result['decision'];transitions.extend(last.transitions);task_rows.append(dict(tick=int(ticks[t]),phase=int(last.phase),progress=float(last.path_progress),path_complete=bool(last.path_complete),task_complete=bool(last.task_complete),failure_reason=last.failure_reason,grasp=bool(fb.grasp_established),released=bool(fb.released),supported=bool(support[t])))
     if last.terminated:terminal_tick=int(ticks[t])
   coupled=np.array([f.grasp_established for f in feedback_rows]);lost=np.array([f.grasp_lost for f in feedback_rows]);failures=measured_failures(pose,coupled,lost,support,gap,footprint);lift=pose[:,2,3]-pose[0,2,3];palm=np.linalg.norm(hand[:,0]-pose[:,:3,3],axis=-1);multicontact=(np.linalg.norm(contact,axis=-1)>.1).sum(-1)>=2;hold=(lift>=.05)&~support&(palm<=.2)&multicontact&valid;seen=int(ticks[-1]);source_endpoint=spec.source_length-1;coverage=seen>=source_endpoint+30
   row=dict(arm=arm,sequence=name,declared_task=str(side/'raw-task.json'),terminal_mode=spec.terminal_mode.value,goal_type=spec.goal_type,source_length=spec.source_length,observed_controls=seen,source_endpoint_covered=bool(seen>=source_endpoint),source_endpoint_plus_1s_covered=bool(coverage),full_task_success=bool(last.task_complete),protocol_full_positive_eligible=bool(last.task_complete or coverage),censored=bool(not last.task_complete and terminal_tick is None and not coverage),terminal_tick=terminal_tick,terminal_failure=last.failure_reason,final_phase=int(last.phase),phases_visited=sorted(set(r['phase'] for r in task_rows)),maximum_path_progress=max(r['progress'] for r in task_rows),path_complete_ever=any(r['path_complete'] for r in task_rows),transitions=list(dict.fromkeys(transitions)),maximum_lift_m=float(lift.max()),longest_force_hold_controls=longest(hold[1:]),longest_force_hold_elapsed_s=max(0.,(longest(hold[1:])-1)/30),lift_hold_elapsed1s_success=bool(longest(hold[1:])>=31),lift_hold30_success=bool(longest(hold[1:])>=30),grasp_established_ever=bool(coupled.any()),terminal_endpoint_position_error_m=float(np.linalg.norm(pose[-1,:3,3]-spec.object_path[-1,:3,3])),final_supported=bool(support[-1]),support_origin=support_origin,measured_failure_frames={k:int(v.sum()) for k,v in failures.items()},result_reason='observed_success' if last.task_complete else last.failure_reason if last.failure_reason else 'right_censored_no_endpoint_plus_settle_observation' if not coverage else 'not_complete_within_registered_budget',declared_task_intent_verified=False)
   if arm=='source_pd':row.update(pd_clipped_envcontrols=int(controls['clipped'][:,index].sum()),pd_clipped_within_reference=int((controls['clipped'][:,index]&controls['within_reference'][:,index]).sum()),maximum_pd_target_error=float(np.abs(controls['target'][:,index]-controls['actual_pd'][:,index]).max()))
   rows.append(row);scoring_traces[arm+'/'+name]=task_rows
 summary={arm:dict(count=len(names),full_task_successes=sum(r['full_task_success'] for r in rows if r['arm']==arm),registered_success_rate=sum(r['full_task_success'] for r in rows if r['arm']==arm)/len(names),censored=sum(r['censored'] for r in rows if r['arm']==arm),lift_hold30_successes=sum(r['lift_hold30_success'] for r in rows if r['arm']==arm),lift_hold_elapsed1s_successes=sum(r['lift_hold_elapsed1s_success'] for r in rows if r['arm']==arm),gate_95_pass=sum(r['full_task_success'] for r in rows if r['arm']==arm)/len(names)>=.95) for arm in ['teacher','source_pd']}
 out=dict(status='COMPLETED',label=a.label,package=str(a.package.resolve()),scheduler_schema=getattr(task_scheduler,'SCHEDULER_SCHEMA','undeclared_pre_v2'),declarations='unchanged original full raw-task.json; no fresh-trajectory relabeling',thresholds=dict(force_n=.1,contact_bodies=2,lift_m=.05,palm_distance_m=.2,hold_controls=30,elapsed1s_minimum_positive_states=31,elapsed_seconds='max(0,(n_positive_states-1)/30)',full_task='package defaults unchanged'),summary=summary,records=rows,limitations=['raw declared RELEASE may encode loss, not intended release','teacher missing saved settling suffix','source-PD cubemedium clipping prevents exact upper-control certification','not a sample-efficiency or WAM decoding comparison']);a.output.mkdir(parents=True,exist_ok=True);(a.output/(a.label+'-summary.json')).write_text(json.dumps(out,indent=2)+'\n');(a.output/(a.label+'-task-traces.json')).write_text(json.dumps(scoring_traces,indent=2)+'\n');print(json.dumps(summary))
if __name__=='__main__':main()
