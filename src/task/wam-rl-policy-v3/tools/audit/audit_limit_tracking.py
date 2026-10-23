"""Independent state/PD/limit and object tracking audit; no physics or GPU."""
import argparse,csv,json,sys
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation,RotationSpline
ROOT=Path(__file__).resolve().parents[5]

def angle(r):return np.arccos(np.clip((np.trace(r,axis1=-2,axis2=-1)-1)/2,-1,1))
def matrix(x):
 y=np.repeat(np.eye(4)[None],len(x),axis=0);y[:,:3,3]=x[:,:3];y[:,:3,:3]=Rotation.from_quat(x[:,3:7]).as_matrix();return y

def lag_gain(clock,measured,reference):
 # Descriptive one-trajectory position fit, not an identified transfer function.
 duration=clock[-1];delay_max=min(.5,duration/4);lead_max=min(.25,duration/4)
 use=(clock>=delay_max)&(clock<=duration-lead_max)
 t=clock[use];y=measured[use]
 if len(t)<5:return dict(delay_s=None,gain=None,scope='too few common samples')
 y=y-y.mean(0);best=None
 delays=np.arange(-int(lead_max*30),int(delay_max*30)+1)/30
 for d in delays:
  x=reference(t-d);x=x-x.mean(0);den=float((x*x).sum())
  if den<1e-12:continue
  gain=float((x*y).sum()/den);residual=float(((y-gain*x)**2).sum())
  candidate=(residual,abs(d),float(d),gain)
  if best is None or candidate<best:best=candidate
 if best is None:return dict(delay_s=None,gain=None,scope='zero reference variation')
 return dict(delay_s=best[2],gain=best[3],scope='30Hz common-window least squares with free position intercept')

def audit(run):
 m=json.loads((run/'manifest.json').read_text());assert m['status'] in ('COMPLETED','STOPPED_P0'),m['status']
 native=m['native_audit'];bounds=native['physical_joint_limits'];lo=np.asarray(bounds['lower']);hi=np.asarray(bounds['upper']);mask=np.asarray(bounds['has_limits'],bool)
 offset=np.asarray(native['controller']['offset']);scale=np.asarray(native['controller']['scale']);names=native['native_dof_names']
 active=[0,1,2,3,4,5,6,8,10,12,14,15];pairs=((7,6,1.05),(9,8,1.05),(11,10,1.05),(13,12,1.05),(16,15,.6),(17,15,.8))
 totals=Counter();phase=defaultdict(Counter);events=defaultdict(Counter);case_events=defaultdict(set);max_pd=0.;bad_pd=0
 for line in (run/'command-admission.jsonl').open():
  z=json.loads(line);q=np.asarray(z['q_after']);q0=np.asarray(z['q_before']);target=np.asarray(z['pd_target']);action=np.asarray(z['executed_raw'])
  full=np.zeros(18);full[active]=np.clip(action,-1,1);expected=offset+scale*(1+full)/2;expected[:6]=q0[:6]+offset[:6]+scale[:6]*full[:6]
  for dst,src,ratio in pairs:expected[dst]=expected[src]*ratio
  max_pd=max(max_pd,float(np.abs(expected-target).max()))
  assert np.allclose(expected,target,atol=2e-6,rtol=0),'PD inverse mismatch'
  outside=mask&((q<lo)|(q>hi));near=mask&~outside&(np.minimum(q-lo,hi-q)<=.10)
  command=mask&((target<lo)|(target>hi));gate=mask&((q<lo-.10)|(q>hi+.10))
  margin=np.minimum(.02,.10*(hi-lo));inner=mask&((target<lo+margin-2e-6)|(target>hi-margin+2e-6));bad_pd+=int(inner.any())
  categories=dict(true_outside=outside,near_inside=near,command_outside_state_inside=command&~outside,physical_gate_outside=gate)
  for key,value in categories.items():
   ids=np.flatnonzero(value).tolist();assert ids==z['categories'][key]
   totals[key]+=int(value.any());phase[z['phase']][key]+=int(value.any())
   for j in ids:events[key][names[j]]+=1
   if ids:case_events[key].add(z['case'])
  totals['controls']+=1;phase[z['phase']]['controls']+=1
  for key in ('braking','target_projected','canonical_proposal_outside'):
   totals[key]+=int(z['protection'][key]);phase[z['phase']][key]+=int(z['protection'][key])
 assert totals['controls']==m['controls'];assert bad_pd==0
 rows=[];max_error=0.;ref_mismatch=[];reset_mismatch=[];group=defaultdict(list)
 for r in m['results']:
  index=r['index'];folder=run/('case%03d'%index);row={k:r[k] for k in ('index','shape','scale_m','requested_mean_speed_m_s','stage','feedforward','status','tracked_controls')}
  row['abort']=None if r.get('abort') is None else r['abort']['kind']
  row['pass_tracking']=False
  if not (folder/'trajectory.npz').exists():rows.append(row);group[(r['stage'],r['feedforward'])].append(row);continue
  with np.load(folder/'trajectory.npz') as z:obj=matrix(z['objects'][:,0]);palm=matrix(z['palm_pose'][:,0]);q=z['q'][:,0]
  with np.load(folder/'reference.npz') as z:times=z['times'];poses=z['poses'];relative=z['hand_object']
  ticks=json.loads((folder/'ticks.json').read_text());clock=np.asarray([x['reference_time_s'] for x in ticks]);pos=CubicSpline(times,poses[:,:3,3],bc_type='clamped');rot=RotationSpline(times,Rotation.from_matrix(poses[:,:3,:3]))
  e=np.linalg.norm(obj[:,:3,3]-pos(clock),axis=-1);a=angle(rot(clock).as_matrix().transpose(0,2,1)@obj[:,:3,:3]);hand_obj=np.linalg.inv(palm)@obj
  slip=np.linalg.norm(hand_obj[:,:3,3]-relative[:3,3],axis=-1);turn=angle(relative[:3,:3].T@hand_obj[:,:3,:3]);margin=np.minimum(q-lo,hi-q);margin[:,~mask]=np.inf
  complete=r['status']=='COMPLETED';closed=bool(r.get('closed'));closure=float(np.linalg.norm(obj[-1,:3,3]-poses[0,:3,3])) if closed and complete else None
  close_angle=float(angle(poses[0,:3,:3].T@obj[-1,:3,:3])) if closed and complete else None
  passed=bool(complete and np.quantile(e,.95)<=.015 and np.quantile(a,.95)<=np.deg2rad(5) and slip.max()<=.01 and turn.max()<=np.deg2rad(10) and margin.min()>=.02 and not any(x['collision_pairs'] for x in ticks) and (not closed or closure<=.015 and close_angle<=np.deg2rad(5)))
  assert passed==r['pass_tracking'],('success disagreement',index)
  assert abs(np.quantile(e,.95)-r['position_quantiles_m'][3])<1e-9
  max_error=max(max_error,float(np.abs(a-np.asarray([x['orientation_error_rad'] for x in ticks])).max()))
  worst=np.unravel_index(np.argmin(margin),margin.shape)
  row.update(pass_tracking=passed,position_p95_mm=float(np.quantile(e,.95)*1000),orientation_p95_deg=float(np.rad2deg(np.quantile(a,.95))),
             drift_max_mm=float(slip.max()*1000),drift_max_deg=float(np.rad2deg(turn.max())),minimum_margin=float(margin.min()),minimum_joint=names[worst[1]],
             closure_mm=None if closure is None else closure*1000,closure_deg=None if close_angle is None else float(np.rad2deg(close_angle)),**lag_gain(clock,obj[:,:3,3],pos))
  base=ROOT/'outputs/wam-rl-policy-v3'/('object-trajectory-tracking-20261023-r2' if r['stage']=='A' else 'object-trajectory-tracking-20261023-r4')
  old_index=(index//4)*4+index%2 if r['stage']=='A' else (index//4)*2+index%2
  old_folder=base/('case%03d'%old_index)
  with np.load(old_folder/'reference.npz') as z:
   if not np.array_equal(times,z['times']) or not np.array_equal(poses,z['poses']):ref_mismatch.append(index)
  with np.load(old_folder/'initial.npz') as z:old_q=z['q'];old_obj=z['objects']
  with np.load(folder/'initial.npz') as z:
   if not np.array_equal(old_q,z['q']) or not np.array_equal(old_obj,z['objects']):reset_mismatch.append(index)
  rows.append(row);group[(r['stage'],r['feedforward'])].append(row)
 summary={}
 for key,rs in group.items():
  valid=[r for r in rs if 'position_p95_mm' in r];fit=[r for r in valid if r['delay_s'] is not None]
  summary[str(key)]=dict(cases=len(rs),complete=sum(x['status']=='COMPLETED' for x in rs),passed=sum(x['pass_tracking'] for x in rs),
    aborts=dict(Counter(x['abort'] or x['status'] for x in rs if x['status']!='COMPLETED')),
    position_p95_median_mm=float(np.median([x['position_p95_mm'] for x in valid])) if valid else None,
    orientation_p95_median_deg=float(np.median([x['orientation_p95_deg'] for x in valid])) if valid else None,
    lag_fit_cases=len(fit),lag_median_s=float(np.median([x['delay_s'] for x in fit])) if fit else None,
    gain_median=float(np.median([x['gain'] for x in fit])) if fit else None)
 assert not ref_mismatch and not reset_mismatch,(ref_mismatch,reset_mismatch)
 result=dict(status=m['status'],rows=len(rows),planned=180,unmeasured=len(m['unmeasured']),groups=summary,controls=m['controls'],
    state_categories=dict(totals),phases={k:dict(v) for k,v in phase.items()},joint_events={k:dict(v) for k,v in events.items()},
    cases_with_categories={k:len(v) for k,v in case_events.items()},maximum_pd_reconstruction_error=max_pd,
    inner_pd_violations=bad_pd,maximum_orientation_reconstruction_error=max_error,reference_mismatch=ref_mismatch,reset_mismatch=reset_mismatch)
 (run/'limit-audit.json').write_text(json.dumps(result,indent=2)+'\n')
 fields=list(dict.fromkeys(k for row in rows for k in row))
 with (run/'limit-matrix.csv').open('w') as f:
  writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
 print(json.dumps(result,indent=2));return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args();audit(args.run.resolve())
