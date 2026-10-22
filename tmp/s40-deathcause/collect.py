"""brief40: read frozen trajectories; replay actual safety gates, never simulate."""
import argparse, collections, csv, importlib, json, os, sys, types
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[2]
ROOTS=[('S2e',ROOT/'tmp/s2e-geometry-20261022/outputs/wam-rl-policy-v3/s2e-geometry-20261022-r1'),('S2d',ROOT/'tmp/s2d-controls-20261022/outputs/wam-rl-policy-v3/s2d-controls-20261022-r4'),('BC',ROOT/'outputs/wam-rl-policy-v3/bc-ppo-usefulness-20261023-r1/assessment-bc'),('PPO',ROOT/'outputs/wam-rl-policy-v3/bc-ppo-usefulness-20261023-r1/assessment-ppo')]

def read(p):return json.loads(p.read_text())
def local(p):
 try:return str(p.relative_to(ROOT))
 except ValueError:return str(p)
def pose(v):
 p=np.eye(4);p[:3,:3]=Rotation.from_quat(v[3:7]).as_matrix();p[:3,3]=v[:3];return p

def package(root,name):
 p=root/'runtime-code/src/task/wam-rl-policy-v3/src/wam_rl_policy_v3'
 module=types.ModuleType(name);module.__path__=[str(p)];sys.modules[name]=module
 return importlib.import_module(name+'.physical_safety'),importlib.import_module(name+'.tau_replay_contract')

def tablespace(s,hand,table_bounds):
 table=s['table_pose'][0];rot=Rotation.from_quat(table[3:7]).as_matrix();a=int(np.argmin(table_bounds[1]-table_bounds[0]));sign=1 if rot[2,a]>=0 else -1;top=table_bounds[1,a] if sign>0 else -table_bounds[0,a]
 pts=(hand-table[:3])@rot;axes=[i for i in range(3) if i!=a];inside=((pts[:,axes]>=table_bounds[0,axes])&(pts[:,axes]<=table_bounds[1,axes])).all(-1);gaps=sign*pts[:,a]-top
 return dict(minimum_sample_gap_m=float(gaps.min()),sample_inside_footprint=int(inside.sum()),minimum_inside_gap_m=float(gaps[inside].min()) if inside.any() else None)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'tmp/s40-deathcause/evidence.json');args=parser.parse_args()
 if ROOT/'tmp' not in args.output.resolve().parents or not args.output.resolve().parent.name.startswith('s40-'):raise ValueError('owned s40 output required')
 catalog=read(ROOT/'outputs/wam-rl-policy-v3/inspire-teacher-rollout-audit-20261020-r3/manifest.json')['records'];cache={};records=[];mutants=collections.Counter();replayed_states=0
 fallback_names=read(ROOT/'outputs/wam-rl-policy-v3/bc-ppo-usefulness-20261023-r1/assessment-bc/p0000-A1/native-audit.json')['native_dof_names']
 for batch,run in ROOTS:
  manifest=read(run/'manifest.json');assert manifest['status']=='COMPLETED'
  gate,contract=package(run,'s40_'+batch.lower())
  for file in sorted(run.glob('*/result.json')):
   folder=file.parent;r=read(file);audit=manifest.get('native_initial_audit') or read(folder/'native-audit.json');sequence=r.get('sequence',manifest.get('source_sequence'));limits=audit['physical_joint_limits'];lo=np.asarray(limits['lower']);hi=np.asarray(limits['upper']);mask=np.asarray(limits['has_limits'],bool);names=audit.get('native_dof_names',fallback_names)
   if sequence not in cache:
    c=next(c for c in catalog if c['sequence']==sequence)
    with np.load(Path(c['source'])/'canonical.npz') as z:points=z['points'].copy()
    with np.load(ROOT/'outputs/wam-rl-policy-v3/inspire-teacher-rollout-audit-20261020-r3/sequences'/sequence/'admission.npz') as z:bounds=z['table_bounds'].copy()
    cache[sequence]=(points,bounds)
   points,bounds=cache[sequence]
   with np.load(folder/'trajectory.npz') as z:data={k:z[k].copy() for k in z.files}
   n=len(data['action']);assert len(data['q'])==n+1 and np.array_equal(data['tick'][:,0],np.arange(n+1))
   states=[{k:v[t] for k,v in data.items() if k in ('q','dq','objects','table_pose','object_contact_force')} for t in range(n+1)]
   safeties=[gate.physical_safety(s,0,points,bounds,limits) for s in states];replayed_states+=len(states)
   first_abort=next((i for i,s in enumerate(safeties) if s['abort']),None);published=r.get('physical_safety_abort',r.get('physical_abort'))
   assert safeties[-1]['reason']==published and (first_abort is None or first_abort==n),(batch,folder.name,'gate/clock mismatch')
   q=data['q'][:,0];pd=data['pd_targets_actual'];excess=np.maximum(lo-q,q-hi);excess[:,~mask]=0;pd_excess=np.maximum(lo-pd,pd-hi);pd_excess[:,~mask]=0
   wrist_delta=np.diff(q[:,3:6],axis=0);branch=(np.abs(wrist_delta)>2*np.pi)&(np.abs(q[:-1,3:6])>2*np.pi-.25);branch_ticks=(np.flatnonzero(branch.any(-1))+1).tolist()
   independently=np.flatnonzero((excess>.1).any(-1));assert independently.tolist()==[i for i,s in enumerate(safeties) if s['joint_violation_indices']],(batch,folder.name,'independent joint replay')
   expected=contract.pd_targets(data['action'],q[:-1],np.asarray(audit['controller']['offset']),np.asarray(audit['controller']['scale']));pd_error=float(np.max(np.abs(expected-pd)));assert pd_error<2e-5
   # Sensitivity checks on the original numerical gate, not a claim of physical repair.
   if published=='physical_joint_limit':
    assert (excess[-1]>.1).any();mutants['silently_remove_joint_gate']+=1
    corrupted=dict(states[-1]);corrupted['q']=np.clip(q[-1],lo,hi)[None];assert not gate.physical_safety(corrupted,0,points,bounds,limits)['joint_violation_indices'];mutants['score_precommand_instead_of_postcommand']+=int(not safeties[-2]['abort'])
   infos=read(folder/'consumer.json') if (folder/'consumer.json').exists() else None
   goal=np.asarray(infos[-1]['postgrasp_object_pose'])[:3,3] if infos else data['objects'][0,0,:3]+[0,0,.08]
   goal_source='consumer.postgrasp_object_pose' if infos else 'runner task: initial object +8cm world Z (confirmed separately)'
   final_error=float(np.linalg.norm(data['objects'][-1,0,:3]-goal));reported=r.get('final_goal_error_m',r.get('final_object_goal_error_m'));assert reported is None or abs(final_error-reported)<1e-6
   success=bool(r['final_success']);stage=r.get('stage_achieved',r.get('final_stage',{}).get('achieved'))
   stop='physical_abort' if published else ('legacy_success_early_stop' if success and n<600 else 'horizon_truncation')
   signature=published or ('legacy_witness_success' if success else ('stage_contact_not_certified' if stage is not None and not stage['contact'] else 'stage_unrecorded' if stage is None else 'other_stage_failure'))
   clip=(np.abs(data['raw_action']-data['action'])>1e-6).any(-1);adapter=data.get('adapter_raw_action');adapter_clips=int((np.abs(adapter)>1+1e-6).any(-1).sum()) if adapter is not None else r.get('adapter_clipped_controls')
   joints=[]
   for j in safeties[-1]['joint_violation_indices']:
    joints.append(dict(index=int(j),name=names[j],q=float(q[-1,j]),previous_q=float(q[-2,j]),delta_q=float(q[-1,j]-q[-2,j]),dq=float(data['dq'][-1,0,j]),lower=float(lo[j]),upper=float(hi[j]),nominal_excess=float(excess[-1,j]),last_PD_target=float(pd[-1,j]),target_nominal_excess=float(max(pd_excess[-1,j],0))))
   first_pd=np.flatnonzero((pd_excess>1e-6).any(-1));first_unsafe=np.flatnonzero((pd_excess>.1).any(-1))
   forces=np.linalg.norm(data['contact_forces'][:,0],axis=-1);hand=data['hand'][-1,0];obj=data['objects'][-1,0];world=points@pose(obj)[:3,:3].T+obj[:3];tipd=np.linalg.norm(hand[[2,4,6,8,10],None]-world[None],axis=-1).min(-1)
   lastpd=pd[-1];prevpd=pd[-2] if n>1 else q[0]
   last_bad=np.flatnonzero(mask&((lastpd<lo-1e-6)|(lastpd>hi+1e-6))).tolist()
   row=dict(batch=batch,label=folder.name,sequence=sequence,chain=r.get('chain',batch),controller='R actor + v7 measured consumer' if batch=='S2e' else 'R actor direct per-tick action' if batch=='S2d' else 'R + frozen residual-BC + geometry consumer' if batch=='BC' else 'R + PPO-trained residual + geometry consumer',tau_source='geometry position reference THROUGH neural R' if r.get('chain')=='geometry' else 'object-anchored GT position reference THROUGH neural R' if r.get('chain')=='c-live-gt' else 'WAM prediction',controls=n,terminal_tick=int(data['tick'][-1,0]),first_abort_tick=first_abort,termination=stop,signature=signature,legacy_final_success=success,stage_achieved=stage,final_phase=infos[-1]['phase'] if infos else None,goal_error_m=final_error,goal_source=goal_source,last_raw_action_max=float(np.abs(data['raw_action'][-1]).max()),last_action_max=float(np.abs(data['action'][-1]).max()),final_clip_controls=int(clip.sum()),adapter_clip_controls=adapter_clips,last_control_clipped=bool(clip[-1]),last_PD_outside_nominal_indices=last_bad,terminal_PD_translation_error_m=float(np.linalg.norm(lastpd[:3]-q[-1,:3])),terminal_measured_wrist_orientation_step_rad=float((Rotation.from_quat(data['palm_pose'][-1,0,3:7])*Rotation.from_quat(data['palm_pose'][-2,0,3:7]).inv()).magnitude()),last_command_delta_position_m=float(np.linalg.norm(lastpd[:3]-q[-2,:3])),last_command_delta_wrist_rad=float(np.linalg.norm(lastpd[3:6]-q[-2,3:6])),last_PD_target_step_position_m=float(np.linalg.norm(lastpd[:3]-prevpd[:3])),last_PD_target_step_wrist_rad=float(np.linalg.norm(lastpd[3:6]-prevpd[3:6])),last_measured_step_wrist_rad=float(np.linalg.norm(q[-1,3:6]-q[-2,3:6])),max_PD_reconstruction_error=pd_error,pd_target_nominal_outside_controls=int((pd_excess>1e-6).any(-1).sum()),pd_target_beyond_abort_margin_controls=int((pd_excess>.1).any(-1).sum()),first_PD_outside_nominal_tick=int(first_pd[0]) if len(first_pd) else None,first_PD_beyond_abort_margin_tick=int(first_unsafe[0]) if len(first_unsafe) else None,maximum_PD_nominal_excess=float(max(pd_excess.max(),0)),final_safety=safeties[-1],joints=joints,force_valid_final=bool(data['contact_force_valid'][-1,0]),force_norm_final_n=forces[-1].tolist(),force_max_previous_n=float(forces[-2].max()),force_max_final_n=float(forces[-1].max()),final_tip_surface_distances_m=tipd.tolist(),final_hand_table=tablespace(states[-1],hand,bounds),previous_hand_table=tablespace(states[-2],data['hand'][-2,0],bounds),initial_root_translation=data['q'][0,0,:3].tolist(),wrist_branch_transition_state_ticks=branch_ticks,terminal_wrist_branch_transition=bool(n in branch_ticks),source_files=dict(trajectory=local(folder/'trajectory.npz'),result=local(file),safety_code=local(Path(gate.__file__)),consumer=local(folder/'consumer.json') if infos else None),last_window=[])
   for t in range(max(0,n-4),n+1):
    row['last_window'].append(dict(state_tick=t,q=q[t].tolist(),dq=data['dq'][t,0].tolist(),force_valid=bool(data['contact_force_valid'][t,0]),force_norm_n=forces[t].tolist(),safety=safeties[t],PD_command_to_next_state=pd[t].tolist() if t<n else None))
   records.append(row)
 assert len(records)==32
 joint=[r for r in records if r['signature']=='physical_joint_limit'];aborts=[r for r in records if r['first_abort_tick'] is not None]
 summary=dict(status='PASS_EVIDENCE_REPLAY_NOT_CAUSAL_CERTIFICATION',n=32,states_replayed=replayed_states,controls_read=sum(r['controls'] for r in records),physical_abort_episodes=len(aborts),joint_abort_episodes=len(joint),joint_events=sum(len(r['joints']) for r in joint),failures_in_original_protocol=sum(not r['legacy_final_success'] for r in records),stops=dict(collections.Counter(r['termination'] for r in records)),signatures=dict(collections.Counter(r['signature'] for r in records)),joint_abort_with_any_prior_PD_outside_nominal=sum(r['pd_target_nominal_outside_controls']>0 for r in joint),joint_abort_triggering_joint_last_PD_outside_nominal=sum(any(j['target_nominal_excess']>1e-6 for j in r['joints']) for r in joint),joint_abort_any_joint_last_PD_outside_nominal=sum(bool(r['last_PD_outside_nominal_indices']) for r in joint),joint_names=dict(collections.Counter(j['name'] for r in joint for j in r['joints'])),gate_sensitivity_checks=dict(mutants),terminal_branch_episodes=sum(r['terminal_wrist_branch_transition'] for r in records),terminal_branch_joint_abort_episodes=sum(r['terminal_wrist_branch_transition'] for r in joint),branch_transition_events=sum(len(r['wrist_branch_transition_state_ticks']) for r in records),nonterminal_branch_events=sum(len([t for t in r['wrist_branch_transition_state_ticks'] if t!=r['terminal_tick']]) for r in records),records=records,scope='Read-only per-state safety and PD reconstruction. Same gate is not proof of a common physical/controller cause. No physics/model calls, no relabeling success to a new protocol.')
 args.output.parent.mkdir(exist_ok=True,parents=True);args.output.write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
 fields=['batch','label','sequence','chain','controller','tau_source','terminal_tick','first_abort_tick','termination','signature','final_phase','goal_error_m','last_action_max','last_command_delta_position_m','last_command_delta_wrist_rad','last_measured_step_wrist_rad','terminal_PD_translation_error_m','last_PD_outside_nominal_indices','terminal_wrist_branch_transition','final_clip_controls','adapter_clip_controls','pd_target_nominal_outside_controls','maximum_PD_nominal_excess']
 with args.output.with_suffix('.csv').open('w') as f:
  writer=csv.DictWriter(f,fieldnames=fields,lineterminator="\n");writer.writeheader();writer.writerows({k:r[k] for k in fields} for r in records)
 print(json.dumps({k:v for k,v in summary.items() if k!='records'},indent=2))
if __name__=='__main__':main()
