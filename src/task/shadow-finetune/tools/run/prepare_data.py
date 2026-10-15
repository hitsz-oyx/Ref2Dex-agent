"""Extract Shadow-native supervision from existing raw episodes, preserving clocks."""
import argparse,json,sys,time,shutil
from pathlib import Path
import numpy as np
TASK=Path(__file__).resolve().parents[2];sys.path.insert(0,str(TASK/'src'))
from shadow_finetune import paths
from video_data_preparation.unidexgrasp_data import object_poses


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--pilot-only',action='store_true');p.add_argument('--train-objects',type=int,default=0)
    p.add_argument('--dev-objects',type=int,default=0);p.add_argument('--seed',type=int,default=211)
    p.add_argument('--budget-seconds',type=int,default=1800);a=p.parse_args()
    out=a.output.resolve()
    if out.exists() or paths.ROOT/'outputs/shadow-finetune' not in out.parents:raise ValueError('fresh Task output required')
    out.mkdir(parents=True);(out/'sequences').mkdir();started=time.monotonic()
    roots=[paths.PILOT]+([] if a.pilot_only else sorted(paths.SOURCE.glob('batch*/replay')))
    manifests={root:json.loads((root/'trajectory-manifest.json').read_text()) for root in roots}
    rng=np.random.default_rng(a.seed);selected=set()
    for split,limit in [('train',a.train_objects),('dev',a.dev_objects)]:
        names=sorted({o['object_code'] for m in manifests.values() for o in m['objects'] if o['split']==split})
        if limit:names=list(np.asarray(names)[rng.permutation(len(names))[:limit]])
        selected.update(names)
    records=[];rejected=[]
    for root,m in manifests.items():
        candidates=[r for r in m['trajectories'] if r['object_code'] in selected and r['endpoint_within_5cm']]
        for episode in sorted({r['episode'] for r in candidates}):
            if time.monotonic()-started>a.budget_seconds:raise TimeoutError('bounded data preparation')
            if shutil.disk_usage(out).free<20*2**30:raise RuntimeError('20GiB free floor')
            with np.load(root/('episode-%02d.npz'%episode),allow_pickle=False) as f:
                data={k:f[k] for k in ('hand','object','observation','action','applied','goal','time','reset')}
            if not np.allclose(np.diff(data['time']),1/60,atol=1e-7):raise ValueError('60Hz clock')
            for row in [r for r in candidates if r['episode']==episode]:
                env=row['env'];hand=data['hand'][:,env];poses=object_poses(data['object'][:,env])
                if data['reset'][:-1,env].any():raise ValueError('internal reset')
                action=data['action'][:,env]
                if np.abs(action).max()>1.00001:raise ValueError('canonical action range')
                if not np.allclose(action[:,6:],data['applied'][:,env,6:],atol=1e-6):raise ValueError('finger command mismatch')
                observation=data['observation'][:,env]
                if any(not np.isfinite(x).all() for x in (hand,poses,action,observation)):raise ValueError('nonfinite source')
                speed=np.linalg.norm(np.diff(poses[:,:3,3],axis=0),axis=-1)*60
                hand_speed=np.linalg.norm(np.diff(hand,axis=0),axis=-1).max(-1)*60
                rel=poses[:-1,:3,:3].transpose(0,2,1)@poses[1:,:3,:3]
                angle=np.arccos(np.clip((np.trace(rel,axis1=1,axis2=2)-1)/2,-1,1))*60
                bad=np.flatnonzero((speed>5)|(hand_speed>5)|(angle>40))+1
                onset=int(bad[0]) if len(bad) else len(hand)
                ticks=np.arange(max(0,min(len(hand)-48,onset-48)))
                local=np.einsum('tpi,tij->tpj',hand-poses[:,None,:3,3],poses[:,:3,:3])
                ticks=ticks[(local[ticks]>=-2).all((1,2))]
                if not len(ticks):rejected.append(dict(root=str(root),episode=episode,env=env,reason='no complete safe window'));continue
                objid=next(i for i,o in enumerate(m['objects']) if o['object_code']==row['object_code'])
                target=out/'sequences'/('%06d.npz'%len(records))
                np.savez_compressed(target,hand=hand,poses=poses,goal=data['goal'][:,env],observation=observation,action=action,time=data['time'])
                records.append(dict(path=str(target.relative_to(out)),geometry=str(root/'geometry'/('object-%03d.npz'%objid)),
                    object_code=row['object_code'],scale=row['scale'],split=row['split'],ticks=ticks.tolist(),
                    raw_episode=str(root/('episode-%02d.npz'%episode)),env=env,unsafe_onset=onset,
                    selection='terminal object-goal distance <=5cm; not stable-grasp certification'))
            print(json.dumps(dict(root=str(root),episode=episode,sequences=len(records))),flush=True)
    if not all(any(r['split']==s for r in records) for s in ('train','dev')):raise ValueError('both splits need successful candidates')
    manifest=dict(schema=paths.SCHEMA,status='COMPLETED',seed=a.seed,records=records,rejected=rejected,
        control_hz=60,history_offsets=[-6,-4,-2,0],forecast_offsets=list(range(2,49,2)),action_offsets=list(range(8)),
        goal='known task goal translation; preserve current object orientation, no future observed object goal',
        object_selection=sorted(selected),training_allowed=True,elapsed_s=time.monotonic()-started)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(status='COMPLETED',records=len(records),windows=sum(len(r['ticks']) for r in records),elapsed_s=manifest['elapsed_s'])))

if __name__=='__main__':main()
