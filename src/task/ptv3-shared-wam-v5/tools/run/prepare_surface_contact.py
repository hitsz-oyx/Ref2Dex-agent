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


def write(path,value):
    tmp=path.with_suffix('.json.part');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def prepare(folder,meta,geometries,output):
    joints=np.load(folder/'joints.npy',mmap_mode='r')
    offsets=np.load(folder/'surface_offsets.npy',mmap_mode='r')
    valid=np.load(folder/'surface_valid.npy',mmap_mode='r')
    expected=np.load(folder/'expected_hand_presence.npy',mmap_mode='r')
    parts=np.load(folder/'surface_parts.npy')
    poses=np.load(folder/'object_poses.npy',mmap_mode='r')
    known=np.load(folder/'object_pose_valid.npy',mmap_mode='r')
    observed=np.load(folder/'observed_frame.npy')
    n=len(joints);objects=len(meta['objects'])
    confidence=np.zeros((n,objects,2048),np.float16)
    labels=np.zeros((n,objects,2048),np.uint8)
    cv=np.zeros((n,objects),bool)
    coverage=(~expected|valid).all(-1)&valid.any(-1)&observed
    for slot,name in enumerate(meta['objects']):
        key=f"s{meta['source_id']}:{name}" if meta['source_id']<3 else name
        if key not in geometries:continue
        points=geometries[key]['surface']
        for frame in np.flatnonzero(coverage&known[:,slot]):
            sides=valid[frame]
            vertices=(offsets[frame].astype(np.float32)+joints[frame,:,None,0])[sides].reshape(-1,3)
            vertex_parts=parts[sides].reshape(-1)
            pose=poses[frame,slot]
            queries=points@pose[:3,:3].T+pose[:3,3]
            confidence[frame,slot],labels[frame,slot]=surface_contact(vertices,vertex_parts,queries)
            cv[frame,slot]=True
    pending=output/(meta['key']+'.partial')
    if pending.exists():pending=output/(meta['key']+f'.partial-{os.getpid()}')
    pending.mkdir()
    for name,value in [('contact',confidence),('part',labels),('valid',cv)]:np.save(pending/(name+'.npy'),value)
    result=dict(status='COMPLETED',key=meta['key'],source_id=meta['source_id'],label_semantics=LABEL_SEMANTICS,
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
    write(a.output/'run-manifest.json',dict(git_commit=a.git_commit,run_id=a.output.name,
        work_version=LABEL_SEMANTICS,hands=str(a.hands),geometry=str(a.geometry),device='CPU offline KD-tree; absent from model',
        seconds=a.seconds,min_free_gib=20,pid=os.getpid(),python=sys.version))
    while time.monotonic()-start<a.seconds and not (a.output/'STOP').exists():
        if shutil.disk_usage(a.output).free<20*2**30:break
        if geometries is None:
            if not (a.geometry/'manifest.json').exists():
                write(a.output/'progress.json',dict(status='WAITING_GEOMETRY',pid=os.getpid(),elapsed_s=time.monotonic()-start))
                time.sleep(10);continue
            manifest=json.loads((a.geometry/'manifest.json').read_text());geometries={}
            for row in manifest['objects']:
                with np.load(a.geometry/row['path'],allow_pickle=False) as z:geometries[row['key']]={k:z[k] for k in z.files}
        changed=False
        folders=sorted((a.hands/'sequences').iterdir())
        priority=[]
        for sid in (0,1,2,4,5,6,8,9,10):
            first=next((f for f in folders if f.name.startswith(f's{sid}-') and '.partial' not in f.name),None)
            if first is not None:priority.append(first)
        ordered=priority+[f for f in folders if f not in priority]
        for folder in ordered:
            if folder.name in done or '.partial' in folder.name or not (folder/'meta.json').exists():continue
            key=folder.name;meta=json.loads((folder/'meta.json').read_text())
            existing=a.output/key/'meta.json'
            result=json.loads(existing.read_text()) if existing.exists() else prepare(folder,meta,geometries,a.output)
            records[key]=result;done.add(key);changed=True
            progress=dict(status='RUNNING',pid=os.getpid(),sequences=len(done),
                valid_object_frames=sum(r['valid_object_frames'] for r in records.values()),
                elapsed_s=time.monotonic()-start,checked_at=time.time(),last_key=key)
            write(a.output/'progress.json',progress)
            if len(done)<=9 or len(done)%20==0:print(json.dumps(progress),flush=True)
            if time.monotonic()-start>a.seconds or (a.output/'STOP').exists():break
        hp=a.hands/'progress.json'
        state=json.loads(hp.read_text()) if hp.exists() else {}
        if not changed and state.get('status') in ('COMPLETED','PARTIAL','STOPPED','BOUNDED_COMPLETE'):
            write(a.output/'progress.json',dict(status='COMPLETED_FOR_AVAILABLE_HANDS',sequences=len(done),
                valid_object_frames=sum(r['valid_object_frames'] for r in records.values()),elapsed_s=time.monotonic()-start));return
        if not changed:time.sleep(10)
    write(a.output/'progress.json',dict(status='STOPPED',sequences=len(done),elapsed_s=time.monotonic()-start))


if __name__=='__main__':main()
