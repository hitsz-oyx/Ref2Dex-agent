"""Publish an explicit immutable v5 manifest and TRAIN-only joint21 moments."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

TASK=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(TASK/'src'))
from ptv3_shared_wam_v5.contracts import DATA_SCHEMA,INPUT_SEMANTICS,LABEL_SEMANTICS,STATS_SCHEMA
from ptv3_shared_wam_v5.data import V5Windows
from ptv3_shared_wam_v5.storage import read_json,load_array


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','hands','contacts','geometry','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--max-sequences-per-source',type=int)
    p.add_argument('--require-complete',action='store_true')
    p.add_argument('--training-admitted',action='store_true')
    p.add_argument('--moments-per-source',type=int,default=128)
    p.add_argument('--git-commit',required=True);a=p.parse_args()
    plan=read_json(a.plan);geo=read_json(a.geometry/'manifest.json')
    keys={r['key'] for r in geo['objects']};sequences=[];counts={};indices={'train':[],'dev':[]};pending=[]
    a.output.mkdir(parents=True,exist_ok=False)
    for job in plan['jobs']:
        sid=job['source_id'];key=job['key'];folder=a.hands/'sequences'/key;cf=a.contacts/key
        if a.max_sequences_per_source and counts.get(sid,0)>=a.max_sequences_per_source:continue
        if not (folder/'meta.json').exists() or not (cf/'meta.json').exists():pending.append(key);continue
        jv=load_array(folder/'joint_valid.npy',mmap_mode='r');usable=jv.any((1,2))
        row_index=len(sequences);selected={}
        for split,relative in job['windows'].items():
            rows=load_array(a.plan.parent/relative)
            valid=np.ones(len(rows),bool)
            for slot in np.unique(rows[:,0]):
                name=job['objects'][int(slot)];gkey=f's{sid}:{name}' if sid<3 else name
                if gkey not in keys:valid[rows[:,0]==slot]=False
            ticks=rows[:,1]
            for lag in (0,1,2,3):valid &= usable[ticks-lag]
            rows=rows[valid]
            if len(rows):selected[split]=np.column_stack((np.full(len(rows),row_index),rows)).astype(np.int32)
        if not selected:pending.append(key);continue
        sequences.append(dict(key=key,source_id=sid,path=str(folder.resolve()),contact_path=str(cf.resolve()),objects=job['objects']))
        counts[sid]=counts.get(sid,0)+1
        for split,rows in selected.items():indices[split].append(rows)
    if a.require_complete and pending:raise ValueError(f'{len(pending)} sequences still unavailable; no full manifest published')
    if not sequences:raise ValueError('no ready sequences')
    paths={}
    for split,rows in indices.items():
        value=np.concatenate(rows) if rows else np.empty((0,3),np.int32)
        dest=a.output/(split+'.npy');np.save(dest,value);dest.chmod(0o664);paths[split]=str(dest.resolve())
    manifest=dict(schema=DATA_SCHEMA,status='FROZEN',input_semantics=INPUT_SEMANTICS,label_semantics=LABEL_SEMANTICS,
        training_allowed=bool(a.training_admitted),engineering_subset=bool(a.max_sequences_per_source or pending),
        git_commit=a.git_commit,sequences=sequences,indices=paths,geometry_manifest=str((a.geometry/'manifest.json').resolve()),
        source_ids=sorted(counts),source_probabilities={str(s):plan['source_probabilities'][str(s)] for s in counts},
        pending_sequences=len(pending),base_plan=str(a.plan.resolve()),
        limitation='Surface proximity is not simulator/measured physical contact. Synthetic midpoint contact labels are masked.')
    manifest_path=a.output/'manifest.json';manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    manifest_path.chmod(0o664)
    ds=V5Windows(manifest_path,'train');rng=np.random.default_rng(20261019)
    moments={name:dict(sum=np.zeros((24,3)),sq=np.zeros((24,3)),count=np.zeros((24,1))) for name in ('hand','scene')}
    panel=[]
    for sid,pool in ds.pools.items():
        chosen=rng.choice(pool,min(len(pool),a.moments_per_source),replace=False);panel.extend(map(int,chosen))
        for index in chosen:
            row=ds[int(index)]
            for name,target_key,valid_key in (('hand','trajectory_target','trajectory_valid'),('scene','scene_target','scene_valid')):
                values=row[target_key].astype(np.float64);valid=row[valid_key]
                if name=='scene':values=values.transpose(1,0,2,3);valid=valid.transpose(1,0,2)
                values=values.reshape(24,-1,3);mask=valid.reshape(24,-1,1)
                m=moments[name];m['sum']+=np.where(mask,values,0).sum(1);m['sq']+=np.where(mask,values**2,0).sum(1);m['count']+=mask.sum(1)
    stats=dict(schema=STATS_SCHEMA,split='train',units='m',fps=30,history=4,horizon=24,hand_points=21,
        source_ids=sorted(counts),panel_indices=panel,std_floor_m=.001,
        method='TRAIN-only, uniform within each available source, bounded fixed panel; new joint21 moments, not inherited v4.')
    for name,m in moments.items():
        mean=m['sum']/m['count'].clip(1);std=np.sqrt(np.maximum(m['sq']/m['count'].clip(1)-mean**2,1e-6))
        stats[name]=dict(mean=mean.tolist(),std=std.tolist(),count=m['count'].astype(int).tolist())
    (a.output/'stats.json').write_text(json.dumps(stats,indent=2)+'\n')
    (a.output/'stats.json').chmod(0o664)
    fingerprint=hashlib.sha256(manifest_path.read_bytes()+(a.output/'stats.json').read_bytes()).hexdigest()
    (a.output/'identity.json').write_text(json.dumps(dict(schema=DATA_SCHEMA,identity=fingerprint),indent=2)+'\n')
    print(json.dumps(dict(sequences=len(sequences),source_counts=counts,windows={s:len(load_array(p)) for s,p in paths.items()},pending=len(pending))),flush=True)


if __name__=='__main__':main()
