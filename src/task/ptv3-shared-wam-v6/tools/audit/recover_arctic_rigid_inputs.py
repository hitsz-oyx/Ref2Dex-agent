"""Restore deleted native input arrays from still-local official raw annotations.

Keep the inherited two rigid pose slots; emit no articulation or relation field.
Use the existing native reader and check every still-cached source2 sequence
before admitting a recovery. Original/raw inputs and old receipts stay read-only.
"""
import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
import prepare_full_rigid_data as prep


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--device',default='cuda:7')
    p.add_argument('--execute',action='store_true');a=p.parse_args();run=a.run.resolve()
    tools=prep.TASK.parent/'cm-pointflow-effect-pretrain/tools/run/ref5_data_expansion'
    sys.path.insert(0,str(tools));sys.path.insert(0,str(prep.TASK.parent/'cm-pointflow-effect-pretrain/src'))
    import prepare_native as native
    from smplx import MANO
    from ptv3_shared_wam_v5.contracts import SPARSE11_IN_21
    import torch
    torch.set_num_threads(1)
    plan=prep.read(prep.PLAN);inventory=prep.read(run/'inventory.json');jobs={r['key']:r for r in plan['jobs']}
    selected=[r for r in inventory['rows'] if r['status']=='REJECT' and r['source_id']==2]
    rows=[]
    for r in selected:
        job=jobs[r['key']];raw=Path(job['raw_path']);op=raw.with_name(raw.name.replace('.mano.npy','.object.npy'))
        params=np.load(op,mmap_mode='r',allow_pickle=False);n=len(params)
        if n!=r['frames'] or params.ndim!=2 or params.shape[1]!=7:raise ValueError('original native frame clock/shape changed')
        packet=prep.ROOT/'outputs/ptv3-shared-wam-v5/hand-packets-20261019-r1/packets'/(r['key']+'.npz')
        with np.load(packet,allow_pickle=False) as z:
            if len(z['pose'])!=n:raise ValueError('packet/native clock mismatch')
        exact=sum(prep.array_size(shape,dtype) for shape,dtype in [((n,2,11,3),'f4'),((n,2),'?'),((n,2,4,4),'f4'),((n,2),'?')])+prep.META_BYTES
        rows.append(dict(key=r['key'],frames=n,base_exact_bytes=exact,conversion_exact_bytes=prep.fixed_conversion_sizes(n,2),raw=str(raw),object_raw=str(op),packet=str(packet)))
    exact=sum(r['base_exact_bytes']+r['conversion_exact_bytes'] for r in rows)
    budget=dict(git_commit=prep.subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),environment=dict(python=sys.version,executable=sys.executable,numpy=np.__version__),schema='ref2dex.native-rigid-recovery.v1',source_id=2,sequences=len(rows),exact_new_base_and_conversion_bytes=exact,
        base_exact_bytes=sum(r['base_exact_bytes'] for r in rows),conversion_exact_bytes=sum(r['conversion_exact_bytes'] for r in rows),
        free_before_bytes=prep.shutil.disk_usage(run).free,storage_floor_bytes=prep.FLOOR,
        future_remaining_conversion_bytes=sum(r['conversion_bytes'] for r in inventory['rows'] if r['status']=='CONVERT' and not (run/'receipts'/(r['key']+'.json')).exists()),
        training_allowed=False,device=a.device,rows=rows)
    print(json.dumps({k:v for k,v in budget.items() if k!='rows'}),flush=True)
    prep.write(run/'native-recovery-budget.json',budget)
    if not a.execute:return
    prep.guard(run,exact+budget['future_remaining_conversion_bytes']+16*prep.GIB)
    models={side:MANO('/home2/wyy/oyx_ws/Ref2Dex/dataset/arctic/data/body_models/mano/models',is_rhand=side=='right',use_pca=False,flat_hand_mean=False).to(a.device) for side in ['right','left']}
    # Independent original raw reconstruction, not using the hand packet to
    # create an alignment target from itself.
    def raw_arrays(job):
        raw=Path(job['raw_path']);hp=np.load(raw,allow_pickle=True).item()
        op=raw.with_name(raw.name.replace('.mano.npy','.object.npy'));params=np.load(op,allow_pickle=False)
        hand=[]
        for side in ['right','left']:
            f=hp[side]
            hand.append(native.reconstruct(dict(global_orient=f['rot'],hand_pose=f['pose'],transl=f['trans'],betas=np.broadcast_to(f['shape'],(len(params),10))),models[side],a.device))
        hand=np.stack(hand,axis=1);valid=np.isfinite(hand).all(axis=(-1,-2));hand[~valid]=0
        poses=native.arctic_poses(params)
        return dict(hand=hand.astype(np.float32),hand_valid=valid,poses=poses,pose_valid=native.rigid_valid(poses))
    old=prep.read(prep.OLD/'recipe/manifest.json')['sequences'];checks=[]
    for r in old:
        if r['source_id']!=2:continue
        raw=raw_arrays(jobs[r['key']]);folder=Path(r['path'])
        original=prep.load_arrays(folder,['joints','joint_valid','object_poses','object_pose_valid'])
        err=float(np.max(np.abs(raw['hand']-original['joints'][:,:,SPARSE11_IN_21])))
        perr=float(np.max(np.abs(raw['poses']-original['object_poses'])))
        if err>5e-5 or perr>1e-7 or not np.array_equal(raw['pose_valid'],original['object_pose_valid']):raise ValueError('cached native reconstruction parity failure')
        checks.append(dict(key=r['key'],hand_max_error_m=err,pose_max_error=perr))
    root=run/'recovered-native-inputs';root.mkdir(exist_ok=True);mappings=[];started=time.monotonic()
    for index,r in enumerate(rows):
        prep.guard(run,r['base_exact_bytes']+128*1024**2)
        if (run/'STOP').exists():raise InterruptedError('owner STOP')
        folder=root/r['key'];job=jobs[r['key']]
        if not (folder/'meta.json').exists():
            values=raw_arrays(job);pending=folder.with_name(folder.name+'.partial-'+str(__import__('os').getpid()));pending.mkdir()
            for k,v in values.items():np.save(pending/(k+'.npy'),v)
            prep.write(pending/'meta.json',dict(key=r['key'],status='COMPLETED',source_id=2,frames=r['frames'],
                original_raw_hand_sha256=prep.digest(r['raw']),original_raw_object_sha256=prep.digest(r['object_raw']),
                inherited_pose_slots=job['objects'],new_articulation_fields=False,parity_checks=checks),fixed=True)
            pending.rename(folder)
        actual=sum(p.stat().st_size for p in folder.iterdir())
        if actual!=r['base_exact_bytes']:raise ValueError('recovery exact size disagreement')
        newjob=dict(job,sequence_folder=str(folder));inspected=prep.inspect_job(newjob)
        mappings.append(dict(key=r['key'],job=newjob,inventory_updates=dict(inspected,status='CONVERT',conversion_bytes=r['conversion_exact_bytes'],sequence_id=next(x['sequence_id'] for x in selected if x['key']==r['key']))))
        if (index+1)%32==0:print(json.dumps(dict(stage='native_recovery',completed=index+1,total=len(rows),elapsed_s=time.monotonic()-started)),flush=True)
    prep.write(run/'recovered-inputs.json',dict(status='COMPLETED',parity_checks=checks,plan_sha256=inventory['plan_sha256'],rows=mappings))
    print(json.dumps(dict(status='COMPLETED',recovered_sequences=len(mappings),parity_checks=checks)),flush=True)


if __name__=='__main__':main()
