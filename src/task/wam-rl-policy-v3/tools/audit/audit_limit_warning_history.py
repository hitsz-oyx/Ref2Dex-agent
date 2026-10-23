"""New guard warnings on frozen brief40 states; NOT counterfactual rollouts."""
import json,sys,argparse
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[5];sys.path.insert(0,str(ROOT/'src/task/wam-rl-policy-v3/src'))
from wam_rl_policy_v3.reference_admission import JointEnvelope

def main(out):
 x=json.loads((ROOT/'tmp/s40-deathcause/evidence.json').read_text());rows=[]
 for row in x['records']:
  if row['final_safety']['reason']!='physical_joint_limit':continue
  folder=(ROOT/row['source_files']['trajectory']).parent;manifest=json.loads((folder.parent/'manifest.json').read_text())
  audit=manifest.get('native_audit') or manifest.get('native_initial_audit')
  if audit is None:audit=json.loads((folder/'native-audit.json').read_text())
  env=JointEnvelope(audit['physical_joint_limits'],audit['controller']['offset'],audit['controller']['scale'])
  with np.load(folder/'trajectory.npz') as f:q=f['q'][:,0];dq=f['dq'][:,0];actions=f['action']
  warnings=[];state_bad=[];modified=[]
  for t,action in enumerate(actions):
   _,info=env.protect(action,q[t],dq[t])
   if info['braking']:warnings.append(t)
   if info['target_projected']:modified.append(t)
   if ((env.mask)&((q[t]<env.nominal_lower)|(q[t]>env.nominal_upper))).any():state_bad.append(t)
  first=None if not warnings else warnings[0]
  rows.append(dict(label=row['batch']+'/'+row['label'],controls=len(actions),first_warning_state_tick=first,warning_controls=len(warnings),
    first_preterminal_nominal_outside_state_tick=None if not state_bad else state_bad[0],command_projection_controls=len(modified),
    warning_before_physical_abort=first is not None and first<len(actions),warning_before_first_nominal_outside=first is not None and (not state_bad or first<state_bad[0]),
    source_trajectory=row['source_files']['trajectory'],hypothetical_only=True))
 result=dict(episodes=len(rows),warning_before_physical_abort=sum(r['warning_before_physical_abort'] for r in rows),
  warning_before_first_nominal_outside=sum(r['warning_before_first_nominal_outside'] for r in rows),rows=rows,
  scope='Offline warning timing only. Altered actions would change subsequent physics; no prevented-abort claim.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.output)
