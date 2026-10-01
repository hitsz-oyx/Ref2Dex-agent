#!/usr/bin/env python3
"""Independent saved-world support, coupling and label audit for scratch observation-policy holding."""
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
    if m['run_status']!='COLLECTION_COMPLETED' or m['evaluation_seeds']!=[517,518]:raise ValueError('cohort drift')
    asset=ROOT/'third_party/DExplore/dexplore/data/assets/mjcf'
    def vertices(file):return np.array([list(map(float,l.split()[1:4])) for l in file.read_text().splitlines() if l.startswith('v ')],dtype=np.float64)
    ov=vertices(asset/'objects/airplane/airplane.obj');tv=vertices(asset/'objects/table/table.obj')
    if int(np.ptp(tv,axis=0).argmin())!=1:raise ValueError('table mesh normal drift')
    rows=[];max_error=0.
    for seed in (517,518):
        d=root/f's{seed}';r=json.loads((d/'results.json').read_text())
        if r['run_status']!='COMPLETED' or not r['no_source_actor_calls'] or r['learned_policy_calls']!=202 or r['mode']!='policy' or not r['policy_model_fingerprint'] or r['policy_checkpoint_sha256']!=m['policy_sha256'] or not r['frame0_initialization']:raise ValueError('panel drift')
        for name in ('initial','trace'):
            if sha(d/(name+'.pt'))!=r[name+'_sha256']:raise ValueError('trace hash drift')
        initial=torch.load(d/'initial.pt',map_location='cpu',weights_only=False);data=torch.load(d/'trace.pt',map_location='cpu',weights_only=False)
        motion=initial['motion'].numpy();base=initial['base_q'].numpy();targets=data['target'].numpy();frames=initial['native_reference_q'].numpy();stops=initial['phase_stop'].numpy();starts=initial['anchors'].numpy();lift_start=initial['lift_start'].numpy();progress=data['progress'].numpy()
        if data['object_root'].shape!=(202,96,13):raise ValueError('coverage')
        if not np.array_equal(progress,np.broadcast_to(np.arange(1,203)[:,None],(202,96))):raise ValueError('progress')
        lo=initial['native_lower'].numpy();hi=initial['native_upper'].numpy()
        if not np.isfinite(targets).all():raise ValueError('finite policy targets')
        coupling={6:((7,1.05),),8:((9,1.05),),10:((11,1.05),),12:((13,1.05),),15:((16,.6),(17,.8))}
        if np.any(targets[...,6:]<lo[6:]-1e-6) or np.any(targets[...,6:]>hi[6:]+1e-6):raise ValueError('native finger targets')
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
        previous_q=np.concatenate((base[None],data['native_q'][:-1].numpy()),0)
        previous_dq=np.concatenate((initial['initial_dof_vel'].numpy()[None],data['native_dq'][:-1].numpy()),0)
        previous_object=np.concatenate((initial['object_root'].numpy()[None],data['object_root'][:-1].numpy()),0)
        previous_pair=np.concatenate((initial['initial_contact'].numpy()[None],data['contact'][:-1].numpy()),0)
        independent_context=[]
        for tick in range(202):
            planned_progress=np.minimum(tick+1,stops[motion])
            value=np.concatenate((previous_q[tick],previous_dq[tick],previous_object[tick],previous_pair[tick],frames[motion,planned_progress],(planned_progress/stops[motion])[:,None]),-1)
            independent_context.append(value)
        if not np.allclose(np.stack(independent_context),data['context'].numpy(),atol=1e-6,rtol=0):raise ValueError('current/planned context causal reconstruction')
        scales=np.array([.02]*3+[.10]*3+[.40]*12,dtype=np.float32)
        residual=data['model_residual'].numpy()
        if residual.shape!=(202,96,18) or np.any(np.abs(residual)>1+1e-6) or np.any(residual[...,[7,9,11,13,16,17]]!=0):raise ValueError('learned residual contract')
        decoded=np.stack(independent_context)[...,51:69]+residual*scales
        decoded[...,14]=np.clip(decoded[...,14],lo[14],hi[14])
        for parent,children in coupling.items():
            bottom=max(lo[parent],*(lo[j]/ratio for j,ratio in children));top=min(hi[parent],*(hi[j]/ratio for j,ratio in children))
            decoded[...,parent]=np.clip(decoded[...,parent],bottom,top)
            for child,ratio in children:decoded[...,child]=decoded[...,parent]*ratio
        if not np.allclose(decoded,targets,atol=1e-5,rtol=0):raise ValueError('independent reference-target decoder')

        geometric=(z-start[None]>=np.float32(.03))&(clear.astype(np.float32)>=np.float32(.02))
        for env in range(96):
            steps=longest(valid[:,env]);stop=int(stops[motion[env]])
            if steps!=int(data['max_hold_steps'][env]):raise ValueError('historical secondary labels')
            window=(progress[:,env]>=stop-74)&(progress[:,env]<=stop+30)
            if int(window.sum())!=105:raise ValueError('full105coverage')
            held=bool(geometric[window,env].all())
            rows.append(dict(seed=seed,environment=env,motion=int(motion[env]),physical105=held,strictforce75=steps>=75,
                minimum_105_root_rise_mm=float((z[window,env]-start[env]).min()*1000),minimum_105_clearance_mm=float(clear[window,env].min()*1000)))
    def summarize(group):return dict(n=len(group),physical105_count=sum(x['physical105'] for x in group),physical105_rate=float(np.mean([x['physical105'] for x in group])),strictforce75_count=sum(x['strictforce75'] for x in group))
    pooled=summarize(rows);motions={str(m):summarize([x for x in rows if x['motion']==m]) for m in range(3)}
    seeds={str(s):summarize([x for x in rows if x['seed']==s and x['motion']==1]) for s in (517,518)}
    if pooled['n']!=192 or any(v['n']!=64 for v in motions.values()) or any(v['n']!=32 for v in seeds.values()):raise ValueError('complete fixed cohorts')
    gates=dict(known_motion1_pooled50=motions['1']['physical105_rate']>=.5,each_seed_motion1_25=all(v['physical105_rate']>=.25 for v in seeds.values()))
    result=dict(run_status='COMPLETED',label='PROMISING' if all(gates.values()) else 'UNPROMISING',pooled=pooled,motions=motions,seeds_motion1=seeds,gates=gates,primary_motion=1,
        boundary='scratch absolute-target initializer feasibility; all motions reported; no Cm, matched learning utility, generalization or formal Validation',independent_clearance_max_abs_error_m=max_error,causal_context_reconstruction_pass=True)
    torch.save(dict(rows=rows),root/'analysis.pt');(root/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
