"""Expand GRAB TRAIN windows only; keep the existing DEV admission unchanged."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys
import numpy as np

TASK=Path(__file__).resolve().parents[2];sys.path.insert(0,str(TASK/'src'))
from ptv3_shared_wam_v6 import paths
from ptv3_shared_wam_v6.local_subset import EXPANDED_SCOPE,index_identity
from ptv3_shared_wam_v6.recipe import validate_recipe
from ptv3_shared_wam_v6.data import V6Windows
from ptv3_shared_wam_v5.cache_contract import metadata_identity
from ptv3_shared_wam_v5.window_contract import history_admitted


def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('previous-recipe','prepared-root','main3-processed','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--git-commit',required=True);a=p.parse_args()
    old=validate_recipe(a.previous_recipe);parent=json.loads(Path(old['parent_manifest']).read_text())
    if shutil.disk_usage(a.output.parent).free<22*2**30:raise ValueError('insufficient preparation peak headroom')
    a.output.mkdir(exist_ok=False);out=a.output.resolve();(out/'parent').mkdir()
    jobs={row['key']:row for row in json.loads((a.prepared_root/'data/plan.json').read_text())['jobs']}
    source=json.loads((a.main3_processed/'manifest.json').read_text())['sources'][1]
    raw=np.load(a.main3_processed/Path(source['indices']['train']['path']).name,mmap_mode='r')
    order=np.argsort(raw[:,0],kind='stable');ordered=raw[order]
    cuts=np.r_[0,np.flatnonzero(np.diff(ordered[:,0]))+1,len(ordered)]
    by_id={int(ordered[start,0]):ordered[start:end] for start,end in zip(cuts[:-1],cuts[1:]) if end>start}
    original=np.load(parent['indices']['train']);categories=np.load(parent['strata']['train'])
    arrays=[];strata=[];rejected=0;grab_counts={}
    for seq_id,row in enumerate(parent['sequences']):
        selected=original[:,0]==seq_id
        if row['source_id']!=1 or not selected.any():
            arrays.append(original[selected]);strata.append(categories[selected]);continue
        proposed=by_id[jobs[row['key']]['sequence_id']]
        ticks=proposed[:,2];slots=proposed[:,1]
        folder=Path(row['path']);valid=np.load(folder/'joint_valid.npy',mmap_mode='r')
        known=np.load(folder/'object_pose_valid.npy',mmap_mode='r')
        keep=(ticks>=3)&(ticks+24<len(valid))
        keep &= history_admitted(valid,np.clip(ticks,3,len(valid)-1))
        # Prefix sums test every H4+K24 pose without an N*28 temporary.
        for slot in np.unique(slots):
            mask=slots==slot;prefix=np.r_[0,np.cumsum(~known[:,int(slot)],dtype=np.int64)]
            start=np.clip(ticks-3,0,len(known));end=np.clip(ticks+25,0,len(known))
            keep[mask] &= (prefix[end]-prefix[start])[mask]==0
        accepted=proposed[keep];rejected+=int((~keep).sum())
        arrays.append(np.column_stack((np.full(len(accepted),seq_id),accepted[:,1:3])).astype(np.int32))
        group=accepted[:,3].astype(np.int8)
        if source['kind']=='native':group-=1
        strata.append(group);grab_counts[row['key']]=len(accepted)
    newtrain=np.concatenate(arrays);newstrata=np.concatenate(strata)
    original_dev=Path(parent['indices']['dev']);original_dev_strata=Path(parent['strata']['dev'])
    for split in ('train','dev'):
        dest=out/'parent'/(split+'.npy');sdest=out/'parent'/(split+'-strata.npy')
        if split=='train':np.save(dest,newtrain);np.save(sdest,newstrata)
        else:shutil.copyfile(original_dev,dest);shutil.copyfile(original_dev_strata,sdest)
        parent['indices'][split]=str(dest);parent['strata'][split]=str(sdest)
    source_ids=np.array([r['source_id'] for r in parent['sequences']])[newtrain[:,0]]
    counts={str(s):int((source_ids==s).sum()) for s in parent['source_ids']}
    parent.update(local_subset_scope=EXPANDED_SCOPE,
                  source_probabilities={s:n/len(newtrain) for s,n in counts.items()},
                  window_expansion='all inherited GRAB TRAIN windows with H4/K24 admission; other sources and DEV unchanged',
                  previous_local_parent=old['parent_manifest'],git_commit=a.git_commit)
    write(out/'parent/manifest.json',parent)
    events=json.loads(Path(old['event_manifest']).read_text())
    events.update(local_subset_scope=EXPANDED_SCOPE,parent_manifest_identity=metadata_identity(out/'parent/manifest.json'))
    write(out/'events.json',events);shutil.copyfile(old['stats'],out/'stats.json')
    recipe=copy.deepcopy(old);recipe.update(local_subset_scope=EXPANDED_SCOPE,git_commit=a.git_commit,
        previous_recipe=str(a.previous_recipe.resolve()),previous_recipe_identity=metadata_identity(a.previous_recipe),
        scope='user-requested expanded GRAB TRAIN, three-GPU scale-up; DEV and normalizers unchanged',
        windows=dict(train=len(newtrain),dev=len(np.load(original_dev))),
        index_identities={s:index_identity(parent['indices'][s]) for s in ('train','dev')},
        strata_identities={s:index_identity(parent['strata'][s]) for s in ('train','dev')})
    for key,path in [('parent_manifest',out/'parent/manifest.json'),('event_manifest',out/'events.json'),('stats',out/'stats.json')]:
        recipe[key]=str(path);recipe[key+'_identity']=metadata_identity(path)
    write(out/'manifest.json',recipe);validate_recipe(out/'manifest.json',out/'stats.json')
    ds=V6Windows(out/'manifest.json','train');checked=0
    for seq_id in np.unique(newtrain[:,0]):
        indices=np.flatnonzero(newtrain[:,0]==seq_id)
        for index in set([int(indices[0]),int(indices[-1])]):
            sample=ds[index]
            if any(not np.isfinite(v).all() for v in sample.values() if np.asarray(v).dtype.kind=='f'):
                raise ValueError('nonfinite expanded actual reader window')
            checked+=1
    report=dict(status='PASS',previous_train_windows=len(original),train_windows=len(newtrain),
        dev_windows=len(np.load(original_dev)),grab_train_windows=counts['1'],
        rejected_unusable_grab_windows=rejected,checked_reader_windows=checked,
        train_sequences=len(np.unique(newtrain[:,0])),source_train_windows=counts,
        sequence_count=len(parent['sequences']),new_sequence_caches=0,
        disk_free_gib=shutil.disk_usage(out).free/2**30,git_commit=a.git_commit)
    write(out/'preflight.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
