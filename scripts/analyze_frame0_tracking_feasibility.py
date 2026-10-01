#!/usr/bin/env python3
"""Independent saved-world support, coupling and label audit for fixed frame0 tracking arms."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.run_contact_response_probe import sha
from scripts.analyze_static_hold_feasibility import rotation,longest

def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2)
    root=a.directory;m=json.loads((root/'run_manifest.json').read_text())
    if m['run_status']!='COLLECTION_COMPLETED' or m['evaluation_seeds']!=[506,507]:raise ValueError('cohort drift')
    asset=ROOT/'third_party/DExplore/dexplore/data/assets/mjcf'
    def vertices(file):return np.array([list(map(float,l.split()[1:4])) for l in file.read_text().splitlines() if l.startswith('v ')],dtype=np.float64)
    ov=vertices(asset/'objects/airplane/airplane.obj');tv=vertices(asset/'objects/table/table.obj')
    if int(np.ptp(tv,axis=0).argmin())!=1:raise ValueError('table mesh normal drift')
    rows=[];max_error=0.
    for seed in (506,507):
        d=root/f's{seed}';r=json.loads((d/'results.json').read_text())
        if r['run_status']!='COMPLETED' or not r['no_policy_calls'] or r['arm_counts']!=[48]*2 or not r['frame0_initialization']:raise ValueError('panel drift')
        for name in ('initial','trace'):
            if sha(d/(name+'.pt'))!=r[name+'_sha256']:raise ValueError('trace hash drift')
        initial=torch.load(d/'initial.pt',map_location='cpu',weights_only=False);data=torch.load(d/'trace.pt',map_location='cpu',weights_only=False)
        motion=initial['motion'].numpy();arms=initial['arm_assignment'].numpy();base=initial['base_q'].numpy();targets=data['target'].numpy();frames=initial['native_reference_q'].numpy();stops=initial['phase_stop'].numpy();starts=initial['anchors'].numpy();lift_start=initial['lift_start'].numpy();progress=data['progress'].numpy()
        if data['object_root'].shape!=(162,96,13):raise ValueError('coverage')
        if not np.array_equal(progress,np.broadcast_to(np.arange(1,163)[:,None],(162,96))):raise ValueError('progress')
        lo=initial['native_lower'].numpy();hi=initial['native_upper'].numpy()
        expected_targets=[]
        coupling={6:((7,1.05),),8:((9,1.05),),10:((11,1.05),),12:((13,1.05),),15:((16,.6),(17,.8))}
        for tick in range(162):
            target_progress=np.minimum(tick+1,stops[motion]);ref=frames[motion,target_progress].copy()
            dose=.30*arms*np.clip((target_progress-(lift_start[motion]-15))/15,0,1)
            ref[:,14]=np.clip(ref[:,14],lo[14],hi[14])
            for parent,children in coupling.items():
                minimum=max(float(lo[parent]),*[float(lo[j])/ratio for j,ratio in children]);maximum=min(float(hi[parent]),*[float(hi[j])/ratio for j,ratio in children])
                ref[:,parent]=np.clip(ref[:,parent]+dose,minimum,maximum)
                for child,ratio in children:ref[:,child]=ref[:,parent]*ratio
            expected_targets.append(ref)
        expected_targets=np.stack(expected_targets)
        if not np.allclose(expected_targets,targets,atol=1e-6,rtol=0):raise ValueError('reference/ramp/coupling target mismatch')
        if np.any(targets[...,6:]<lo[6:]-1e-6) or np.any(targets[...,6:]>hi[6:]+1e-6):raise ValueError('finger bounds')
        previous_q=np.concatenate((base[None],data['native_q'][:-1].numpy()),0)
        action=data['action'].numpy().copy();normalized=action.copy();normalized[...,6:]=(1+normalized[...,6:])/2
        reconstructed=initial['pd_offset'].numpy()+initial['pd_scale'].numpy()*normalized
        reconstructed[...,:6]+=previous_q[...,:6]
        for parent,children in coupling.items():
            for child,ratio in children:reconstructed[...,child]=reconstructed[...,parent]*ratio
        if not np.allclose(reconstructed,targets,atol=1e-5,rtol=0):raise ValueError('native PD action reconstruction')
        physical=json.loads((d/'physical_metadata.json').read_text())
        if sha(d/'physical_metadata.json')!=r['physical_metadata_sha256'] or physical['gravity']!=[0.,0.,-9.8100004196167]:raise ValueError('physical metadata drift')
        if np.any(np.abs(data['action'].numpy())>1+1e-6):raise ValueError('clipping')
        hand=np.linalg.norm(data['hand_force'].numpy(),axis=-1)>.1;objforce=np.linalg.norm(data['object_force'].numpy(),axis=-1)>.1
        pair=np.stack((hand.any(-1),objforce),-1)
        if not np.array_equal(pair,data['contact'].numpy().astype(bool)):raise ValueError('force reconstruction')
        table=initial['table_root'].numpy();table_rot=rotation(table[:,3:7])
        if np.any(np.abs(table_rot[:,2,1])<.999):raise ValueError('table not horizontal')
        support=(table_rot[:,2,:]@tv.T+table[:,2,None]).max(-1)
        clear=np.stack([(rotation(o[:,3:7])[:,2,:]@ov.T+o[:,2,None]).min(-1)-support for o in data['object_root'].numpy()])
        max_error=max(max_error,float(np.abs(clear-data['clearance'].numpy()).max()))
        if max_error>1e-5:raise ValueError('independent clearance')
        z=data['object_root'][:,:,2].numpy();start=initial['initial_height'].numpy()
        phase=(progress>=starts[motion][None])&(progress<=stops[motion][None])
        if not np.all(phase.sum(0)==90):raise ValueError('all90phase coverage')
        geometry_held=phase&(z-start[None]>=np.float32(.03))&(clear.astype(np.float32)>=np.float32(.02))
        valid=geometry_held&pair.all(-1)
        for env in range(96):
            steps=longest(valid[:,env]);physical_steps=longest(geometry_held[:,env])
            if steps!=int(data['max_hold_steps'][env]):raise ValueError('independent labels')
            rows.append(dict(seed=seed,environment=env,motion=int(motion[env]),arm=int(arms[env]),retained75=steps>=75,max_hold_steps=steps,geometry_only75=physical_steps>=75,geometry_only_max_steps=physical_steps))
    def summarize(group):return dict(n=len(group),retained75_count=sum(x['retained75'] for x in group),retained75_rate=float(np.mean([x['retained75'] for x in group])),mean_max_hold_steps=float(np.mean([x['max_hold_steps'] for x in group])),geometry_only75_count=sum(x['geometry_only75'] for x in group))
    arms={str(k):dict(pooled=summarize([x for x in rows if x['arm']==k]),motions={str(j):summarize([x for x in rows if x['arm']==k and x['motion']==j]) for j in range(3)}) for k in range(2)}
    if any(v['pooled']['n']!=96 or any(u['n']!=32 for u in v['motions'].values()) for v in arms.values()):raise ValueError('all arm coverage')
    gates=dict(pooled10=arms['1']['pooled']['retained75_rate']>=.10,all_motion5=all(v['retained75_rate']>=.05 for v in arms['1']['motions'].values()),gain5pp=arms['1']['pooled']['retained75_rate']-arms['0']['pooled']['retained75_rate']>=.05)
    result=dict(run_status='COMPLETED',label='PROMISING' if all(gates.values()) else 'UNPROMISING',arms=arms,gates=gates,trajectories=192,
        boundary='prospective fixed frame0 approach/lift/holding controller; no learned policy, attributed pair forces, force closure or Validation; geometric secondary cannot replace primary',independent_clearance_max_abs_error_m=max_error)
    torch.save(dict(rows=rows),root/'analysis.pt');(root/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
