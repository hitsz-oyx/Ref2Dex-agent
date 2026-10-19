"""Cache per-frame surface contact separately from overlapping training windows."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time
import numpy as np

TASK=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(TASK/'src'))
from ptv3_shared_wam_v5.preparation import surface_contact
from ptv3_shared_wam_v5.contracts import LABEL_SEMANTICS
from ptv3_shared_wam_v5.storage import read_json,load_array
from ptv3_shared_wam_v5.cache_contract import bind_cache,metadata_identity,array_identity,validate_binding,identity


def write(path,value):
    tmp=path.with_suffix('.json.part');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)
    path.chmod(0o664)


def prepare(folder,meta,geometries,output,contract):
    joints=load_array(folder/'joints.npy',mmap_mode='r')
    offsets=load_array(folder/'surface_offsets.npy',mmap_mode='r')
    valid=load_array(folder/'surface_valid.npy',mmap_mode='r')
    expected=load_array(folder/'expected_hand_presence.npy',mmap_mode='r')
    parts=load_array(folder/'surface_parts.npy')
    poses=load_array(folder/'object_poses.npy',mmap_mode='r')
    known=load_array(folder/'object_pose_valid.npy',mmap_mode='r')
    observed=load_array(folder/'observed_frame.npy')
    n=len(joints);objects=len(meta['objects'])
    confidence=np.zeros((n,objects,2048),np.float16)
    labels=np.zeros((n,objects,2048),np.uint8)
    cv=np.zeros((n,objects),bool)
    coverage=(~expected|valid).all(-1)&valid.any(-1)&observed
    surfaces={}
    for slot,name in enumerate(meta['objects']):
        key=f"s{meta['source_id']}:{name}" if meta['source_id']<3 else name
        if key in geometries:surfaces[slot]=geometries[key]['surface']
    from scipy.spatial import cKDTree
    for frame in np.flatnonzero(coverage):
        sides=valid[frame]
        vertices=(offsets[frame].astype(np.float32)+joints[frame,:,None,0])[sides].reshape(-1,3)
        vertex_parts=parts[sides].reshape(-1)
        tree=cKDTree(vertices);bounds=(vertices.min(0),vertices.max(0))
        for slot,points in surfaces.items():
            if not known[frame,slot]:continue
            pose=poses[frame,slot]
            queries=points@pose[:3,:3].T+pose[:3,3]
            confidence[frame,slot],labels[frame,slot]=surface_contact(vertices,vertex_parts,queries,tree=tree,bounds=bounds)
            cv[frame,slot]=True
    pending=output/(meta['key']+'.partial')
    if pending.exists():pending=output/(meta['key']+f'.partial-{os.getpid()}')
    pending.mkdir()
    for name,value in [('contact',confidence),('part',labels),('valid',cv)]:
        dest=pending/(name+'.npy');np.save(dest,value);dest.chmod(0o664)
    result=dict(status='COMPLETED',key=meta['key'],source_id=meta['source_id'],label_semantics=LABEL_SEMANTICS,
        cache_contract=contract['identity'],hand_identity=metadata_identity(folder/'meta.json'),
        geometry_identities={name:geometries[f"s{meta['source_id']}:{name}" if meta['source_id']<3 else name]['identity'] for name in meta['objects']},
        payload_identity=array_identity(dict(contact=confidence,part=labels,valid=cv)),
        valid_object_frames=int(cv.sum()),total_object_frames=int(cv.size),
        positive_queries=int((confidence[cv]>.25).sum()),
        description='Nearest reconstructed hand-surface vertex proximity, radius3mm, exponential10mm falloff, cutoff33mm. Not measured physical contact.',
        missing_geometry='Mask whole contact field when an expected hand lacks geometry; never invent negative labels.',
        interpolated_frames='Contact supervision masked on synthetic HOI4D midpoints.')
    write(pending/'meta.json',result);pending.rename(output/meta['key'])
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hands',type=Path,required=True);p.add_argument('--geometry',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--seconds',type=int,default=21600)
    p.add_argument('--git-commit',required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);start=time.monotonic();done=set();records={};geometries=None
    hc=read_json(a.hands/'cache-contract.json');geometry=read_json(a.geometry/'manifest.json')
    if geometry.get('status')!='COMPLETED' or geometry.get('failures'):raise ValueError('complete object geometry registry required')
    geometries={}
    for row in geometry['objects']:
        with load_array(a.geometry/row['path'],allow_pickle=False) as z:values={k:z[k] for k in z.files}
        values['identity']=array_identity(values);geometries[row['key']]=values
    contract=bind_cache(a.output,'contact',dict(protocol=LABEL_SEMANTICS,
        hands=str(a.hands.resolve()),hand_cache=hc['identity'],
        geometry=str(a.geometry.resolve()),geometry_manifest=metadata_identity(a.geometry/'manifest.json'),
        geometry_payloads=identity({k:v['identity'] for k,v in geometries.items()})))
    legacy_hands=hc['inputs'].get('inherited_human_root')
    legacy_contacts=Path(legacy_hands).parent/'contacts' if legacy_hands else None
    write(a.output/'run-manifest.json',dict(git_commit=a.git_commit,run_id=a.output.name,
        work_version=LABEL_SEMANTICS,hands=str(a.hands),geometry=str(a.geometry),device='CPU offline KD-tree; absent from model',
        seconds=a.seconds,min_free_gib=20,pid=os.getpid(),python=sys.version))
    while time.monotonic()-start<a.seconds and not (a.output/'STOP').exists():
        if shutil.disk_usage(a.output).free<20*2**30:break
        changed=False
        folders=sorted((a.hands/'sequences').iterdir())
        priority=[]
        for sid in (0,1,2,4,5,6,8,9,10):
            first=next((f for f in folders if f.name.startswith(f's{sid}-') and '.partial' not in f.name),None)
            if first is not None:priority.append(first)
        ordered=priority+[f for f in folders if f not in priority]
        for folder in ordered:
            if folder.name in done or '.partial' in folder.name or not (folder/'meta.json').exists():continue
            key=folder.name;meta=read_json(folder/'meta.json')
            validate_binding(folder,meta,hc,Path(legacy_hands)/'sequences' if legacy_hands else None)
            if legacy_contacts and meta['source_id']!=4 and not (a.output/key).exists() and (legacy_contacts/key/'meta.json').exists():
                # Explicit unchanged human-only inheritance. Never reuse Shadow labels.
                old=read_json(legacy_contacts/key/'meta.json')
                if old['label_semantics']!=LABEL_SEMANTICS:raise ValueError('legacy human label semantics differ')
                (a.output/key).symlink_to((legacy_contacts/key).resolve(),target_is_directory=True)
            existing=a.output/key/'meta.json'
            result=read_json(existing) if existing.exists() else prepare(folder,meta,geometries,a.output,contract)
            validate_binding(existing.parent,result,contract,legacy_contacts)
            if result.get('hand_identity') and result['hand_identity']!=metadata_identity(folder/'meta.json'):
                raise ValueError('contact cache refers to a different hand surface')
            records[key]=result;done.add(key);changed=True
            progress=dict(status='RUNNING',pid=os.getpid(),sequences=len(done),
                valid_object_frames=sum(r['valid_object_frames'] for r in records.values()),
                elapsed_s=time.monotonic()-start,checked_at=time.time(),last_key=key)
            write(a.output/'progress.json',progress)
            if len(done)<=9 or len(done)%20==0:print(json.dumps(progress),flush=True)
            if time.monotonic()-start>a.seconds or (a.output/'STOP').exists():break
        hp=a.hands/'progress.json'
        state=read_json(hp) if hp.exists() else {}
        if not changed and state.get('status') in ('COMPLETED','PARTIAL','STOPPED','BOUNDED_COMPLETE'):
            write(a.output/'progress.json',dict(status='COMPLETED_FOR_AVAILABLE_HANDS',sequences=len(done),
                valid_object_frames=sum(r['valid_object_frames'] for r in records.values()),elapsed_s=time.monotonic()-start));return
        if not changed:time.sleep(10)
    write(a.output/'progress.json',dict(status='STOPPED',sequences=len(done),elapsed_s=time.monotonic()-start))


if __name__=='__main__':main()
