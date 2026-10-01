#!/usr/bin/env python3
"""Independent saved-world support, coupling and label audit for all fixed doses."""
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
    if m['run_status']!='COLLECTION_COMPLETED' or m['evaluation_seeds']!=[504,505]:raise ValueError('cohort drift')
    asset=ROOT/'third_party/DExplore/dexplore/data/assets/mjcf'
    def vertices(file):return np.array([list(map(float,l.split()[1:4])) for l in file.read_text().splitlines() if l.startswith('v ')],dtype=np.float64)
    ov=vertices(asset/'objects/airplane/airplane.obj');tv=vertices(asset/'objects/table/table.obj')
    if int(np.ptp(tv,axis=0).argmin())!=1:raise ValueError('table mesh normal drift')
    rows=[];max_error=0.
    for seed in (504,505):
        d=root/f's{seed}';r=json.loads((d/'results.json').read_text())
        if r['run_status']!='COMPLETED' or not r['no_policy_calls'] or r['dose_counts']!=[24]*4:raise ValueError('panel drift')
        for name in ('initial','trace'):
            if sha(d/(name+'.pt'))!=r[name+'_sha256']:raise ValueError('trace hash drift')
        initial=torch.load(d/'initial.pt',map_location='cpu',weights_only=False);data=torch.load(d/'trace.pt',map_location='cpu',weights_only=False)
        motion=initial['motion'].numpy();arms=initial['dose_assignment'].numpy();goal=initial['goal_q'].numpy();base=initial['base_q'].numpy()
        if data['object_root'].shape!=(90,96,13):raise ValueError('coverage')
        if not np.array_equal(data['progress'].numpy(),initial['anchors'].numpy()[motion][None]+np.arange(1,91)[:,None]):raise ValueError('progress')
        if not np.array_equal(goal[:,:6],base[:,:6]) or not np.array_equal(goal[:,14],base[:,14]):raise ValueError('fixed wrist/thumbyaw changed')
        if not np.array_equal(data['target'].numpy(),np.broadcast_to(goal,(90,96,18))):raise ValueError('constant target drift')
        lo=initial['native_lower'].numpy();hi=initial['native_upper'].numpy();doses=initial['dose_rad'].numpy()
        for parent,children in {6:((7,1.05),),8:((9,1.05),),10:((11,1.05),),12:((13,1.05),),15:((16,.6),(17,.8))}.items():
            minimum=max(float(lo[parent]),*[float(lo[j])/ratio for j,ratio in children]);maximum=min(float(hi[parent]),*[float(hi[j])/ratio for j,ratio in children])
            if not np.allclose(goal[:,parent],np.clip(base[:,parent]+doses,minimum,maximum),atol=1e-7,rtol=0):raise ValueError('dose mapping')
            for child,ratio in children:
                if not np.allclose(goal[:,child],goal[:,parent]*ratio,atol=1e-7,rtol=0):raise ValueError('coupling')
        if np.any(goal[:,6:]<lo[6:]-1e-6) or np.any(goal[:,6:]>hi[6:]+1e-6):raise ValueError('finger bounds')
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
        z=data['object_root'][:,:,2].numpy();rest=initial['rest_height'].numpy();start=initial['initial_height'].numpy()
        valid=(z-rest[None]>=np.float32(.03))&(z-start[None]>=np.float32(-.01))&pair.all(-1)&(clear.astype(np.float32)>=np.float32(.02))
        for env in range(96):
            steps=longest(valid[:,env])
            if steps!=int(data['max_hold_steps'][env]):raise ValueError('independent labels')
            rows.append(dict(seed=seed,environment=env,motion=int(motion[env]),arm=int(arms[env]),dose_rad=float(doses[env]),retained75=steps>=75,max_hold_steps=steps))
    def summarize(group):return dict(n=len(group),retained75_count=sum(x['retained75'] for x in group),retained75_rate=float(np.mean([x['retained75'] for x in group])),mean_max_hold_steps=float(np.mean([x['max_hold_steps'] for x in group])))
    arms={str(k):dict(pooled=summarize([x for x in rows if x['arm']==k]),motions={str(j):summarize([x for x in rows if x['arm']==k and x['motion']==j]) for j in range(3)}) for k in range(4)}
    if any(v['pooled']['n']!=48 or any(u['n']!=16 for u in v['motions'].values()) for v in arms.values()):raise ValueError('all arm coverage')
    gates={str(k):dict(pooled50=arms[str(k)]['pooled']['retained75_rate']>=.5,all_motion25=all(v['retained75_rate']>=.25 for v in arms[str(k)]['motions'].values()),gain25pp=arms[str(k)]['pooled']['retained75_rate']-arms['0']['pooled']['retained75_rate']>=.25) for k in (1,2,3)}
    result=dict(run_status='COMPLETED',label='PROMISING' if any(all(v.values()) for v in gates.values()) else 'UNPROMISING',arms=arms,gates=gates,trajectories=192,
        boundary='prospective fixed mechanical dose candidate screen; no frame0 pickup, learned policy, attributed pair forces, force closure or Validation',independent_clearance_max_abs_error_m=max_error)
    torch.save(dict(rows=rows),root/'analysis.pt');(root/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
