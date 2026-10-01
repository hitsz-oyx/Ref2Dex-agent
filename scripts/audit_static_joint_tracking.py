#!/usr/bin/env python3
"""Saved-data joint semantics and realized tracking diagnostic; no physics."""
import argparse,json,sys,xml.etree.ElementTree as ET
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from scripts.run_contact_response_probe import sha
from src.task.CmResidual.v118_planner import NATIVE_TO_URDF
from src.task.CmResidual.dexplore_cm_geometry import native_joint_limits

def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2)
    root=a.directory;m=json.loads((root/'run_manifest.json').read_text())
    if m['run_status']!='COMPLETED':raise ValueError('nonterminal input')
    if any(sha(Path(p))!=h for p,h in m['input_sha256'].items()):raise ValueError('protected input drift')
    out=root/'joint_tracking_audit_r1';out.mkdir(exist_ok=False)
    u=ROOT/'third_party/DExplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf'
    joints=[j for j in ET.parse(u).getroot().findall('joint') if j.get('type')!='fixed']
    lo,hi=native_joint_limits(u,'cpu');bounded=torch.tensor([joints[i].get('type')!='continuous' for i in NATIVE_TO_URDF])
    semantics=[dict(native_index=k,name=joints[i].get('name'),type=joints[i].get('type'),
        lower=joints[i].find('limit').get('lower'),upper=joints[i].find('limit').get('upper')) for k,i in enumerate(NATIVE_TO_URDF)]
    inputs={str(u.resolve()):sha(u),str(Path(__file__).resolve()):sha(Path(__file__)),str(root/'run_manifest.json'):sha(root/'run_manifest.json')}
    rows=[]
    for seed in (502,503):
        d=root/f's{seed}';r=json.loads((d/'results.json').read_text())
        for name in ('initial','trace'):
            file=d/(name+'.pt');inputs[str(file)]=sha(file)
            if inputs[str(file)]!=r[name+'_sha256']:raise ValueError('trace drift')
        a=torch.load(d/'initial.pt',map_location='cpu',weights_only=False);b=torch.load(d/'trace.pt',map_location='cpu',weights_only=False)
        for motion in range(3):
            mask=a['motion']==motion;goal=a['goal_q'][mask];q=b['native_q'][:,mask];delta=q-goal[None]
            violation=torch.maximum(lo-goal,goal-hi).clamp_min(0)
            rows.append(dict(seed=seed,motion=motion,bounded_goal_max_violation=float(violation[:,bounded].max()),
                helper_fallback_max_violation=float(violation.max()),continuous_joint5_goal_rad=float(goal[0,5]),
                realized_joint5_range_rad=[float(q[...,5].min()),float(q[...,5].max())],
                max_wrist_translation_error_mm=float(delta[...,:3].norm(dim=-1).max()*1000),
                max_wrist_rotation_coordinate_error_rad=float(delta[...,3:6].abs().max()),
                max_finger_error_rad=float(delta[...,6:].abs().max()),
                worst_finger_native_joint=int(delta[...,6:].abs().amax((0,1)).argmax())+6,
                max_proxy_conjunction_ticks=int(b['contact'][:,mask].bool().all(-1).sum(0).max())))
    result=dict(run_status='COMPLETED',status='POSTHOC_DIAGNOSTIC',no_new_physics=True,semantics=semantics,records=rows,input_sha256=inputs,
        boundary='continuous wrist axes have no URDF position bounds; geometry-helper +/-pi fallback is not a physical limit; realized tracking errors do not identify cause or intrinsic grasp infeasibility')
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(status=result['status'],records=rows)))
if __name__=='__main__':main()
