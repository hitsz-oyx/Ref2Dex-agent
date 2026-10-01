#!/usr/bin/env python3
"""Independent full-mesh/force/trajectory label audit; no model inference."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.run_contact_response_probe import sha


def rotation(quat):
    q=quat.astype(np.float64);q=q/np.linalg.norm(q,axis=-1,keepdims=True)
    x,y,z,w=np.moveaxis(q,-1,0)
    return np.stack((1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y),
        2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x),
        2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)),-1).reshape(*q.shape[:-1],3,3)


def longest(flags):
    run=maximum=0
    for flag in flags:
        run=run+1 if flag else 0;maximum=max(maximum,run)
    return maximum


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2)
    root=a.directory;m=json.loads((root/'run_manifest.json').read_text())
    if m['run_status']!='COLLECTION_COMPLETED' or m['evaluation_seeds']!=[502,503]:raise ValueError('cohort/status drift')
    asset=ROOT/'third_party/DExplore/dexplore/data/assets/mjcf'
    def vertices(file):
        return np.array([list(map(float,l.split()[1:4])) for l in file.read_text().splitlines() if l.startswith('v ')],dtype=np.float32)
    object_vertices=vertices(asset/'objects/airplane/airplane.obj');table_top=float(vertices(asset/'objects/table/table.obj')[:,2].max())
    rows=[];maximum_error=0.;boundary_samples=0
    for seed in m['evaluation_seeds']:
        directory=root/f's{seed}';r=json.loads((directory/'results.json').read_text())
        if r['run_status']!='COMPLETED' or r['physical_steps_each']!=90 or r['mechanical_trajectories']!=96 or not r['no_policy_calls']:
            raise ValueError('incomplete mechanical panel')
        for name in ('initial','trace'):
            if sha(directory/(name+'.pt'))!=r[name+'_sha256']:raise ValueError('saved output drift')
        initial=torch.load(directory/'initial.pt',map_location='cpu',weights_only=False)
        data=torch.load(directory/'trace.pt',map_location='cpu',weights_only=False)
        obj=data['object_root'].numpy();table=initial['table_root'].numpy();motion=initial['motion'].numpy()
        if obj.shape!=(90,96,13) or len(object_vertices)!=initial['object_vertex_count'] or table_top!=initial['table_top_local_z']:
            raise ValueError('mesh/trace identity drift')
        expected=initial['anchors'].numpy()[motion][None]+np.arange(1,91)[:,None]
        if not np.array_equal(data['progress'].numpy(),expected):raise ValueError('actual progress drift')
        hand=np.linalg.norm(data['hand_force'].numpy(),axis=-1)>.1
        object_contact=np.linalg.norm(data['object_force'].numpy(),axis=-1)>.1
        pair=np.stack((hand.any(-1),object_contact),-1)
        if not np.array_equal(pair,data['contact'].numpy().astype(bool)):raise ValueError('force proxy reconstruction drift')
        inv=rotation(table[:,3:7]).transpose(0,2,1)
        independent=[]
        for state in obj:
            matrix=inv@rotation(state[:,3:7]);trans=(inv@(state[:,:3]-table[:,:3])[...,None])[...,0]
            support=matrix[:,2,:]@object_vertices.T+trans[:,2,None]
            independent.append(support.min(-1)-table_top)
        independent=np.stack(independent);saved=data['clearance'].numpy()
        error=float(np.abs(independent-saved).max());maximum_error=max(error,maximum_error)
        if error>1e-5:raise ValueError('full-mesh table-plane clearance reconstruction mismatch')
        boundary_samples+=int((np.abs(saved-.02)<1e-5).sum())
        height=obj[:,:,2];rest=initial['rest_height'].numpy();start=initial['initial_height'].numpy()
        height_proxy=(height-rest[None]>=np.float32(.03))&(height-start[None]>=np.float32(-.01))&pair.all(-1)
        valid=height_proxy&(saved>=np.float32(.02))
        actual=height_proxy&(independent.astype(np.float32)>=np.float32(.02))
        for env in range(96):
            steps=longest(valid[:,env]);reconstructed=longest(actual[:,env])
            if steps!=int(data['max_hold_steps'][env]) or steps!=reconstructed:raise ValueError('independent longest hold mismatch')
            rows.append(dict(evaluation_seed=seed,environment=env,motion=int(motion[env]),max_hold_steps=steps,
                retained75=steps>=75,height_proxy75=longest(height_proxy[:,env])>=75,
                minimum_clearance_mm=float(saved[:,env].min()*1000),mean_height_loss_mm=float((start[env]-height[:,env]).mean()*1000)))
    def summarize(group):
        return dict(trajectories=len(group),retained75_count=sum(r['retained75'] for r in group),
            retained75_rate=float(np.mean([r['retained75'] for r in group])),
            height_proxy75_count=sum(r['height_proxy75'] for r in group),
            mean_max_hold_steps=float(np.mean([r['max_hold_steps'] for r in group])),
            mean_minimum_clearance_mm=float(np.mean([r['minimum_clearance_mm'] for r in group])))
    pooled=summarize(rows);motions={str(i):summarize([r for r in rows if r['motion']==i]) for i in range(3)}
    gates=dict(pooled75_at_least50percent=pooled['retained75_rate']>=.5,
        **{f'motion{i}_75_at_least25percent':motions[str(i)]['retained75_rate']>=.25 for i in range(3)})
    result=dict(run_status='COMPLETED',label='PROMISING' if all(gates.values()) else 'UNPROMISING',gates=gates,
        pooled=pooled,motions=motions,panels={str(seed):summarize([r for r in rows if r['evaluation_seed']==seed]) for seed in (502,503)},
        no_learned_policy=True,elevated_zero_velocity_initialization=True,
        boundary='fixed static reference-pose mechanical feasibility; no frame0 pickup, learned actor, force closure, policy utility or Validation claim')
    torch.save(dict(rows=rows),root/'analysis.pt');(root/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    (root/'trajectory_audit.json').write_text(json.dumps(dict(status='PASS',trajectories=192,full_mesh_clearance_max_abs_error_m=maximum_error,
        within10micrometers_of_clearance_threshold_samples=boundary_samples,force_proxies_independently_reconstructed=True,
        actual_progress_and_75step_labels_verified=True),indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
