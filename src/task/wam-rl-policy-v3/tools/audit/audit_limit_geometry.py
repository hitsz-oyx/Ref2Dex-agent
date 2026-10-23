"""Audit frozen geometry/R admission episodes, including rejected intent values."""
import argparse,json,sys
from collections import Counter
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[5]

def main(run):
 m=json.loads((run/'manifest.json').read_text());assert m['status']=='COMPLETED' and len(m['runs'])==5
 frozen=run/'runtime-code';sys.path[:0]=[str(frozen/'src/task/wam-rl-policy-v3/src'),str(frozen/'src/task/trajectory-policy/src')]
 from wam_rl_policy_v3.measured_policy import MeasuredInspirePolicy
 from wam_rl_policy_v3.pose_consumer import grasp_intent_from_retargeter,pose7
 from wam_rl_policy_v3.reference_admission import JointEnvelope,limit_categories
 model=MeasuredInspirePolicy(with_tau=True).eval();model.load_state_dict(torch.load(m['base_checkpoint'],map_location='cpu',weights_only=False)['model']);torch.set_num_threads(2)
 audit=m['native_initial_audit'];lim=audit['physical_joint_limits'];lo=np.asarray(lim['lower']);hi=np.asarray(lim['upper']);mask=np.asarray(lim['has_limits'],bool)
 offset=np.asarray(audit['controller']['offset']);scale=np.asarray(audit['controller']['scale']);env=JointEnvelope(lim,offset,scale)
 totals=Counter();ledger=[];pd_error=0.;max_initial_error=0
 old=ROOT/'tmp/s2e-geometry-20261022/outputs/wam-rl-policy-v3/s2e-geometry-20261022-r1'
 for row in m['runs']:
  folder=run/row['label'];consumer=json.loads((folder/'consumer.json').read_text());n=row['controls']
  with np.load(folder/'trajectory.npz') as z:d={k:z[k].copy() for k in z.files}
  assert len(d['q'])==n+1 and len(consumer)==n
  q=d['q'][:,0];pd=d['pd_targets_actual'];bad=[]
  # Independent reconstruction, not the envelope helper.
  full=np.zeros((n,18));full[:,[0,1,2,3,4,5,6,8,10,12,14,15]]=np.clip(d['action'],-1,1)
  expected=offset+scale*(1+full)/2;expected[:,:6]=q[:-1,:6]+offset[:6]+scale[:6]*full[:,:6]
  for dst,src,ratio in ((7,6,1.05),(9,8,1.05),(11,10,1.05),(13,12,1.05),(16,15,.6),(17,15,.8)):expected[:,dst]=expected[:,src]*ratio
  delta=float(np.abs(expected-pd).max());pd_error=max(pd_error,delta);assert delta<2e-6
  for t in range(n):
   cats=limit_categories(q[t+1],pd[t],lim)
   for key in ('true_outside','near_inside','command_outside_state_inside','physical_gate_outside'):totals[key]+=bool(cats[key])
   assert env.check(pd[t])==[]
  for field in ('q','dq','hand','objects','physical','tick','table_pose','contact_force_valid','dof_force_valid'):
   with np.load(old/row['label']/'trajectory.npz') as z:max_initial_error=max(max_initial_error,float(np.abs(d[field][0].astype(float)-z[field][0].astype(float)).max()))
  with torch.inference_mode():raw=model.actor(torch.tensor(d['observation'][-1:])).numpy()[0]
  qlast=q[-1];palm=pose7(d['palm_pose'][-1,0]);obj=pose7(d['objects'][-1,0]);goal=obj.copy();goal[:3,3]=d['objects'][0,0,:3]+[0,0,.08]
  intent=grasp_intent_from_retargeter(qlast,palm,obj,goal,np.clip(raw,-1,1),offset,scale)
  target=obj@intent.approach_pose_object;root=qlast[:3]+target[:3,3]-palm[:3,3]
  original=json.loads((old/row['label']/'result.json').read_text())
  with np.load(old/row['label']/'trajectory.npz') as previous:
   prefix={k:float(np.abs(d[k].astype(float)-previous[k][:len(d[k])].astype(float)).max()) for k in ('q','dq','objects','action','forecast_hand')}
  finalforces=np.linalg.norm(d['contact_forces'][-1,0],axis=-1)
  ledger.append(dict(block=row['block'],old_controls=original['controls'],old_joint_abort=original['physical_safety_abort']=='physical_joint_limit',
   controls=n,new_joint_abort=row['physical_safety_abort']=='physical_joint_limit',new_abort=row['physical_safety_abort'],protective_stop=row['protective_stop'],
   old_observed_first_300=True,new_observed_s=n/30,common_prefix_max_differences=prefix,final_success=row['final_success'],ever_success=row['ever_success'],longest_hold_s=row['longest_hold_elapsed_s'],
   stages=row['final_stage']['achieved'],final_target_error_mm=row['final_object_goal_error_m']*1000,final_lift_mm=float((d['objects'][-1,0,2]-d['objects'][0,0,2])*1000),
   final_two_net_force_fingers=bool(d['contact_force_valid'][-1,0] and (finalforces>=.1).sum()>=2),
   rejected_R_intent_root_xyz_m=root.tolist(),measured_root_xyz_m=qlast[:3].tolist(),native_root_lower_m=lo[:3].tolist(),native_root_upper_m=hi[:3].tolist(),
   reference_finger_alternatives=sum(x['reference_finger_alternative'] for x in consumer),braking_controls=sum(x['admission']['braking'] for x in consumer),
   native_pd_projections=sum(x['admission']['target_projected'] for x in consumer)))
 result=dict(status='COMPLETE_NOT_QUALIFIED',controls=sum(r['controls'] for r in ledger),episodes=len(ledger),old_joint_abort=sum(r['old_joint_abort'] for r in ledger),
   new_joint_abort=sum(r['new_joint_abort'] for r in ledger),protective_stops=sum(r['protective_stop'] is not None for r in ledger),success=sum(r['final_success'] for r in ledger),
   counts=dict(totals),maximum_PD_reconstruction_error=pd_error,maximum_initial_error_vs_frozen=max_initial_error,rows=ledger,
   scope='Earlier protective rejection is failure, not repaired grasp; raw R intent is before bounded pose consumer, unlike last native PD in brief40.')
 assert max_initial_error==0
 (run/'geometry-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();main(a.run.resolve())
