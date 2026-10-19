"""Consume streamed annotation packets into audited, resumable remote hand caches."""
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
from ptv3_shared_wam_v5.preparation import reconstruct,load_base,validate_alignment
from ptv3_shared_wam_v5.contracts import DATA_SCHEMA,JOINT_NAMES,PART_NAMES


def write(path,value):
    pending=path.with_suffix('.json.part');pending.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');pending.replace(path)
    path.chmod(0o664)


def prepare(job,packet,a):
    base=load_base(job)
    if 'robot' in packet:
        from ptv3_shared_wam_v5.shadow import ShadowSurface
        geometry=ShadowSurface(a.shadow_asset)
        joints,jvalid,vertices,vvalid,parts=geometry.reconstruct(packet['q'],packet['hand_root'])
    else:
        joints,jvalid,vertices,vvalid,parts=reconstruct(packet,a.mano_root,a.device)
    if len(joints)!=len(base['hand']):raise ValueError('reconstruction/native clock length mismatch')
    # Never turn missing MANO parameters into present joints or contact negatives.
    presence=base['hand_valid'].any(-1)
    jvalid &= presence[...,None]
    vvalid &= presence
    error=validate_alignment(joints,jvalid,base)
    if not (jvalid & base['hand_valid'].any(-1)[...,None]).any():raise ValueError('no usable joint21 labels')
    pending=a.output/'sequences'/(job['key']+'.partial')
    if pending.exists():
        # Interrupted files remain evidence. A new staging directory is used.
        pending=a.output/'sequences'/(job['key']+f'.partial-{os.getpid()}')
    pending.mkdir()
    origin=joints[:,:,0]
    relative=np.where(vvalid[...,None,None],vertices-origin[:,:,None],0.)
    quantized=relative.astype(np.float16)
    quantization_error=float(np.abs(relative-quantized.astype(np.float32)).max())
    if not np.isfinite(quantized).all() or quantization_error>1e-4:raise ValueError('surface quantization exceeds0.1mm')
    arrays=dict(joints=np.where(jvalid[...,None],joints,0.).astype(np.float32),joint_valid=jvalid,
        surface_offsets=quantized,surface_valid=vvalid,surface_parts=parts,
        object_poses=np.asarray(base['object_poses'],np.float32),object_pose_valid=base['object_pose_valid'],
        observed_frame=base['observed_frame'],expected_hand_presence=presence)
    for key,value in arrays.items():
        dest=pending/(key+'.npy');np.save(dest,value);dest.chmod(0o664)
    meta=dict(schema=DATA_SCHEMA,status='COMPLETED',key=job['key'],source_id=job['source_id'],
        frames=len(joints),surface_frames=int(vvalid.sum()),joint_frames=int(jvalid.all(-1).sum()),
        alignment_max_error_m=error,surface_quantization_max_error_m=quantization_error,
        objects=job['objects'],windows=job['windows'],window_counts=job['window_counts'],
        joint_semantics=list(JOINT_NAMES),part_semantics=list(PART_NAMES),
        contact_status='PENDING_GEOMETRY',source_sequence=job['source_sequence'],
        limitation='Joint-only native sources retain masked unavailable surfaces. HOI4D midpoints are explicitly synthetic.')
    write(pending/'meta.json',meta)
    pending.rename(a.output/'sequences'/job['key'])
    return meta


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--packets',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--mano-root',type=Path,required=True)
    p.add_argument('--shadow-asset',type=Path,required=True)
    p.add_argument('--device',default='cpu');p.add_argument('--seconds',type=int,default=21600)
    p.add_argument('--threads',type=int,default=2);p.add_argument('--limit',type=int)
    p.add_argument('--retry-failed',action='store_true')
    p.add_argument('--git-commit',required=True)
    a=p.parse_args()
    if not 1<=a.seconds<=86400 or not 1<=a.threads<=8:p.error('bounded runtime/threads required')
    import torch
    torch.set_num_threads(a.threads)
    a.output.mkdir(parents=True,exist_ok=True)
    for sub in ('sequences','records'):(a.output/sub).mkdir(exist_ok=True)
    jobs=json.loads(a.plan.read_text())['jobs'];started=time.monotonic();processed=0
    records={p.stem:json.loads(p.read_text()) for p in (a.output/'records').glob('*.json')}
    if a.retry_failed:records={k:r for k,r in records.items() if r['status']=='COMPLETED'}
    write(a.output/'run-manifest.json',dict(schema='ref2dex.wam-v5-preparation-run.v1',git_commit=a.git_commit,
        run_id=a.output.name,work_version=DATA_SCHEMA,plan=str(a.plan),packets=str(a.packets),
        seconds=a.seconds,threads=a.threads,device=a.device,python=sys.version,torch=torch.__version__,
        cuda=torch.version.cuda,pid=os.getpid(),min_free_gib=20,source_read_only=True))
    status='RUNNING'
    while True:
        if time.monotonic()-started>a.seconds or (a.output/'STOP').exists() or shutil.disk_usage(a.output).free<20*2**30:
            status='STOPPED';break
        found=False
        for job in jobs:
            key=job['key']
            if key in records:continue
            source_record=a.packets/'records'/(key+'.json')
            if not source_record.exists():continue
            record=json.loads(source_record.read_text())
            if job['source_id'] in (8,9) and record.get('surface_adapter')!='native-mano-v1':continue
            if record['status']=='READY' and not (a.packets/record['packet']).exists():continue
            found=True
            try:
                if record['status']!='READY':raise ValueError(record.get('reason','native parameters unavailable'))
                with np.load(a.packets/record['packet'],allow_pickle=False) as z:packet={k:z[k] for k in z.files}
                existing=a.output/'sequences'/key/'meta.json'
                result=json.loads(existing.read_text()) if existing.exists() else prepare(job,packet,a)
                record=dict(status='COMPLETED',**{k:result[k] for k in ('source_id','frames','joint_frames','surface_frames','alignment_max_error_m')})
            except Exception as e:
                record=dict(status='QUARANTINED',source_id=job['source_id'],reason=type(e).__name__+': '+str(e))
            records[key]=record;processed+=1;write(a.output/'records'/(key+'.json'),record)
            if processed<=10 or processed%10==0:
                progress=dict(status='RUNNING',pid=os.getpid(),processed=len(records),total=len(jobs),
                    completed=sum(r['status']=='COMPLETED' for r in records.values()),
                    quarantined=sum(r['status']!='COMPLETED' for r in records.values()),
                    elapsed_s=time.monotonic()-started,checked_at=time.time(),last_key=key,last_record=record)
                write(a.output/'progress.json',progress);print(json.dumps(progress),flush=True)
            if (a.limit and processed>=a.limit) or time.monotonic()-started>a.seconds or (a.output/'STOP').exists():break
        if a.limit and processed>=a.limit:status='BOUNDED_COMPLETE';break
        if len(records)==len(jobs):status='COMPLETED' if all(r['status']=='COMPLETED' for r in records.values()) else 'PARTIAL';break
        if not found:
            if (a.packets/'export-finished.json').exists():status='PARTIAL';break
            write(a.output/'progress.json',dict(status='WAITING_PACKETS',pid=os.getpid(),processed=len(records),
                total=len(jobs),elapsed_s=time.monotonic()-started,checked_at=time.time()))
            time.sleep(10)
    write(a.output/'progress.json',dict(status=status,pid=os.getpid(),processed=len(records),total=len(jobs),
        completed=sum(r['status']=='COMPLETED' for r in records.values()),
        quarantined=sum(r['status']!='COMPLETED' for r in records.values()),elapsed_s=time.monotonic()-started,checked_at=time.time()))


if __name__=='__main__':main()
