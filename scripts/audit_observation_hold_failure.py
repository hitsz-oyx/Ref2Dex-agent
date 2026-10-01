#!/usr/bin/env python3
"""Post-hoc state/action diagnostics, with no fitting or test-data reuse."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.run_contact_response_probe import sha

def expert_targets(initial,steps=202):
    motion=initial['motion'].numpy();stops=initial['phase_stop'].numpy()
    progress=np.minimum(np.arange(1,steps+1)[:,None],stops[motion][None])
    goal=initial['native_reference_q'].numpy()[motion[None],progress].copy()
    lower=initial['native_lower'].numpy();upper=initial['native_upper'].numpy()
    goal[...,14]=np.clip(goal[...,14],lower[14],upper[14])
    dose=.30*np.clip((progress-(initial['lift_start'].numpy()[motion][None]-15))/15,0,1)
    for parent,children in {6:((7,1.05),),8:((9,1.05),),10:((11,1.05),),12:((13,1.05),),15:((16,.6),(17,.8))}.items():
        lo=max(lower[parent],*(lower[j]/r for j,r in children));hi=min(upper[parent],*(upper[j]/r for j,r in children))
        goal[...,parent]=np.clip(goal[...,parent]+dose,lo,hi)
        for child,ratio in children:goal[...,child]=goal[...,parent]*ratio
    return goal

def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2)
    root=a.directory.resolve();out=a.output.resolve()
    if out.exists() or ROOT not in out.parents:raise ValueError('new isolated audit path required')
    m=json.loads((root/'run_manifest.json').read_text())
    if m['run_status']!='COMPLETED' or m['label']!='UNPROMISING':raise ValueError('closed baseline required')
    mismatches=[f for f,h in m['input_sha256'].items() if sha(Path(f))!=h]
    if mismatches:raise ValueError('protected input drift: '+repr(mismatches))
    checkpoint=torch.load(root/'fit/policy.pt',map_location='cpu',weights_only=False)
    mean=checkpoint['observation_mean'].numpy();std=checkpoint['observation_std'].numpy();rows=[];inputs={str(root/'run_manifest.json'):sha(root/'run_manifest.json'),str(root/'fit/policy.pt'):sha(root/'fit/policy.pt')}
    for panel in ('teacher','s511','s512'):
        d=root/panel;r=json.loads((d/'results.json').read_text())
        for name in ('initial','trace'):
            f=d/(name+'.pt')
            if sha(f)!=r[name+'_sha256']:raise ValueError('panel trace drift')
            inputs[str(f)]=sha(f)
        initial=torch.load(d/'initial.pt',map_location='cpu',weights_only=False);data=torch.load(d/'trace.pt',map_location='cpu',weights_only=False)
        goal=expert_targets(initial);context=data['context'].numpy();applied=data['target'].numpy();action=data['action'].numpy();motion=initial['motion'].numpy()
        offset=initial['pd_offset'].numpy();scale=initial['pd_scale'].numpy()
        expert=(goal-offset)/scale;expert[...,:6]=(goal[...,:6]-offset[:6]-context[...,:6])/scale[:6];expert[...,6:]=2*expert[...,6:]-1;expert[...,[7,9,11,13,16,17]]=0
        if panel=='teacher' and (np.max(np.abs(goal-applied))>1e-5 or np.max(np.abs(expert-action))>1e-5):raise ValueError('independent teacher reconstruction')
        z=(context-mean)/std
        for mo in range(3):
            ids=motion==mo
            for window,(start,stop) in {'all':(0,202),'early':(0,30),'late':(100,202)}.items():
                error=(applied-goal)[start:stop,ids];shift=z[start:stop,ids];qerror=(data['native_q'].numpy()-goal)[start:stop,ids]
                row=dict(panel=panel,motion=mo,window=window,rows=(stop-start)*int(ids.sum()),
                    wrist_target_rmse_mm=float(np.sqrt(np.mean(error[...,:3]**2))*1000),wrist_target_signed_bias_mm=np.mean(error[...,:3],axis=(0,1)).tolist(),
                    wrist_rotation_target_rmse_rad=float(np.sqrt(np.mean(error[...,3:6]**2))),finger_target_rmse_rad=float(np.sqrt(np.mean(error[...,6:]**2))),
                    realized_wrist_tracking_rmse_mm=float(np.sqrt(np.mean(qerror[...,:3]**2))*1000),
                    clipped_feature_fraction=float(np.mean(np.abs(shift)>10)),contexts_with_any_clip_fraction=float(np.mean((np.abs(shift)>10).any(-1))),
                    expert_queries_outside_action_bounds=int((np.abs(expert[start:stop,ids])>1+1e-6).any(-1).sum()),
                    largest_shift_features=[int(i) for i in np.argsort(np.mean(np.abs(shift),axis=(0,1)))[-5:][::-1]])
                row['wrist_target_signed_bias_mm']=[v*1000 for v in row['wrist_target_signed_bias_mm']];rows.append(row)
    result=dict(run_status='COMPLETED',posthoc=True,no_training=True,no_model_inference=True,no_test_reuse=True,
        protected_input_count=len(m['input_sha256']),protected_inputs_unchanged=True,baseline_wall_seconds=m['wall_seconds'],
        baseline_storage_bytes=sum(f.stat().st_size for f in root.rglob('*') if f.is_file()),rows=rows,
        boundary='descriptive diagnostic; errors and distribution shift do not establish causation',input_sha256=inputs,script_sha256=sha(Path(__file__)))
    out.mkdir(parents=True);(out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','input_sha256')}))
    print(json.dumps([x for x in rows if x['motion']==1]))
if __name__=='__main__':main()
