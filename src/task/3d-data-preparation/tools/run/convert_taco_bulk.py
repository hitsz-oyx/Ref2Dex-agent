"""Stream official ZIP members to compact sequences; no full pickle unpack."""
import argparse
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile
import numpy as np
import torch
from convert_taco_pilot import CPUReader
from convert_hot3d_pilot import geometry, SCHEMA, validate_sequence


def write(path,value):
    temp=path.with_suffix('.partial.json');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--mano-root',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda:0'],default='cpu')
    p.add_argument('--seconds',type=int,default=86400);a=p.parse_args()
    root=a.output.resolve();root.mkdir(parents=True,exist_ok=True);(root/'sequences').mkdir(exist_ok=True)
    (root/'geometry').mkdir(exist_ok=True);(root/'scratch').mkdir(exist_ok=True)
    started=time.monotonic();deadline=started+a.seconds
    archives=a.input/'archives'
    while not all((archives/n).exists() for n in ['Hand_Poses.zip','Object_Poses.zip','Object_Models.zip']):
        write(root/'progress.json',dict(status='WAITING_NATIVE_ARCHIVES',pid=os.getpid(),checked_at=time.time(),training_allowed=False))
        if time.monotonic()>deadline or (root/'STOP').exists():return
        time.sleep(10)
    for name,value in [('bool',bool),('int',int),('float',float),('complex',complex),('object',object),('str',str),('unicode',str)]:
        if name not in np.__dict__:setattr(np,name,value)
    import manopth
    outer=Path(manopth.__file__).resolve().parents[1]
    if (outer/'mano').is_dir() and str(outer) not in manopth.__path__:manopth.__path__.append(str(outer))
    from manopth.manopth.manolayer import ManoLayer
    torch.set_num_threads(4)
    if a.device!='cpu':
        gpu=os.environ.get('CUDA_VISIBLE_DEVICES','')
        if not gpu.isdigit():raise RuntimeError('explicit single physical GPU required')
        memory=int(subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
        if memory>64:raise RuntimeError('selected GPU became busy; existing tasks preserved')
        if not torch.cuda.is_available():raise RuntimeError('requested GPU unavailable')
    models={s:ManoLayer(mano_root=str(a.mano_root),use_pca=False,ncomps=45,side=s,center_idx=0,flat_hand_mean=True).to(a.device)
            for s in ['right','left']}
    manifest=json.loads((root/'manifest.json').read_text()) if (root/'manifest.json').exists() else dict(
        schema=SCHEMA,units='m',fps=30,source_fps_verified=False,training_allowed=False,objects=[],sequences=[],
        limitations=['30Hz nominal frame clock; native timestamp evidence pending.',
                     'Triplet-disjoint engineering split; participant identities unavailable.',
                     'Native annotation tool is target; not robot control supervision.'])
    ledger=json.loads((root/'conversion-ledger.json').read_text()) if (root/'conversion-ledger.json').exists() else {}
    known={r['key'] for r in manifest['objects']}
    conversion_started=time.monotonic();initial_ledger_count=len(ledger)
    with zipfile.ZipFile(archives/'Hand_Poses.zip') as hands,zipfile.ZipFile(archives/'Object_Poses.zip') as poses,zipfile.ZipFile(archives/'Object_Models.zip') as meshes:
        pose_names=set(poses.namelist());hand_names=set(hands.namelist());mesh_names=meshes.namelist()
        prefixes=sorted({str(Path(n).parent) for n in hand_names if Path(n).name in ['left_hand.pkl','right_hand.pkl']})
        triplets=sorted({Path(n).parts[1] for n in prefixes})
        dev=set(triplets[4::5])
        for index,prefix in enumerate(prefixes):
            identity='/'.join(Path(prefix).parts[1:]);triplet=Path(prefix).parts[1]
            if identity in ledger:continue
            if time.monotonic()>deadline or (root/'STOP').exists() or shutil.disk_usage(root).free<20*2**30:break
            try:
                object_prefix='Object_Poses/'+identity+'/'
                object_members=sorted(n for n in pose_names if n.startswith(object_prefix) and n.endswith('.npy'))
                object_members.sort(key=lambda n:(not Path(n).name.startswith('tool_'),n))
                if not 1<=len(object_members)<=6:raise ValueError('native one-to-six-object scene required')
                raw=[np.load(io.BytesIO(poses.read(n)),allow_pickle=False) for n in object_members]
                if any(r.ndim!=3 or r.shape[-2:]!=(4,4) for r in raw):raise ValueError('native rigid object poses required')
                object_poses=np.stack(raw,1);count=len(object_poses)
                if not 28<=count<=20000:raise ValueError('bounded native length28..20000')
                hand=np.full((count,2,11,3),np.nan,np.float32);valid=np.zeros(hand.shape[:-1],bool)
                frame_ids=None;missing_shapes=[]
                for slot,side in enumerate(['right','left']):
                    name=prefix+'/'+side+'_hand.pkl'
                    if name not in hand_names:continue
                    native=CPUReader(io.BytesIO(hands.read(name))).load();keys=sorted(native)
                    ids=np.array([int(k) for k in keys],np.int64)
                    if len(keys)!=count or not (np.diff(ids)==1).all():raise ValueError('native hand/object clock mismatch')
                    if frame_ids is not None and not np.array_equal(frame_ids,ids):raise ValueError('native side frame ID mismatch')
                    frame_ids=ids
                    shape_name=prefix+'/'+side+'_hand_shape.pkl'
                    if shape_name in hand_names:
                        beta=CPUReader(io.BytesIO(hands.read(shape_name))).load()['hand_shape']
                        beta=torch.as_tensor(beta).cpu().float().reshape(1,10)
                    else:
                        missing_shapes.append(side);beta=torch.zeros(1,10)
                    for offset in range(0,count,128):
                        chosen=keys[offset:offset+128]
                        theta=torch.stack([torch.as_tensor(native[k]['hand_pose']).cpu().float() for k in chosen])
                        trans=torch.stack([torch.as_tensor(native[k]['hand_trans']).cpu().float() for k in chosen])
                        with torch.no_grad():_,joints,_=models[side](theta.to(a.device),beta.repeat(len(chosen),1).to(a.device))
                        points=(joints.cpu()/1000+trans[:,None])[:,[0,1,5,9,13,17,4,8,12,16,20]].numpy()
                        hand[offset:offset+len(chosen),slot]=points;valid[offset:offset+len(chosen),slot]=np.isfinite(points).all(-1)
                if frame_ids is None:raise ValueError('no native hand')
                rotations=object_poses[:,:,:3,:3]
                object_known=(np.isfinite(object_poses).all(axis=(-1,-2))
                    & np.isclose(object_poses[:,:,3,:],[0,0,0,1],atol=1e-6).all(-1)
                    & np.isclose(rotations.swapaxes(-1,-2)@rotations,np.eye(3),atol=2e-4).all(axis=(-1,-2))
                    & np.isclose(np.linalg.det(rotations),1,atol=2e-4))
                sequence=dict(time=np.arange(count)/30,hand=hand,hand_valid=valid,object_poses=object_poses,
                              object_pose_valid=object_known,segment=np.zeros(count,np.int64),native_frame_id=frame_ids)
                validate_sequence(sequence);objects=[]
                for member in object_members:
                    uid=Path(member).stem.rsplit('_',1)[-1];key='TACO/'+uid;objects.append(key)
                    if key not in known:
                        mesh_name=next(n for n in mesh_names if Path(n).name==uid+'_cm.obj')
                        temporary=root/'scratch'/(uid+'_cm.obj');temporary.write_bytes(meshes.read(mesh_name))
                        np.savez_compressed(root/'geometry'/(uid+'.npz'),**geometry(temporary,.01));temporary.unlink()
                        manifest['objects'].append(dict(key=key,path='geometry/'+uid+'.npz'));known.add(key)
                filename=f'{index:05d}.npz';temp=root/'sequences'/('pending-'+filename)
                np.savez_compressed(temp,**sequence);temp.replace(root/'sequences'/filename)
                manifest['sequences'].append(dict(path='sequences/'+filename,format='canonical',objects=objects,target_slot=0,
                    target_selection='native annotated tool first',source_id=5,profile='human-mano-manopth-native11',
                    split='dev' if triplet in dev else 'train',group_id='TACO/'+triplet,source_sequence=identity))
                ledger[identity]=dict(status='CONVERTED',frames=count,objects=len(objects),missing_shape_zero_fallback=missing_shapes)
                print('CONVERTED',identity,count,flush=True)
            except Exception as error:
                ledger[identity]=dict(status='FAILED',error_type=type(error).__name__,reason=str(error))
                print('FAILED',identity,type(error).__name__,flush=True)
            write(root/'manifest.json',manifest);write(root/'conversion-ledger.json',ledger)
            write(root/'progress.json',dict(status='CONVERTING',pid=os.getpid(),converted_sequences=len(manifest['sequences']),
                converted_frames=sum(r.get('frames',0) for r in ledger.values()),failed_sequences=sum(r['status']=='FAILED' for r in ledger.values()),
                total_sequences=len(prefixes),device=a.device,conversion_elapsed_s=time.monotonic()-conversion_started,
                estimated_remaining_minutes=(len(prefixes)-len(ledger))*(time.monotonic()-conversion_started)/max(len(ledger)-initial_ledger_count,1)/60,
                gpu_allocated_mib=torch.cuda.memory_allocated()/2**20 if a.device!='cpu' else 0,
                elapsed_s=time.monotonic()-started,checked_at=time.time(),training_allowed=False))
    write(root/'progress.json',dict(status='FINISHED',pid=os.getpid(),converted_sequences=len(manifest['sequences']),
        converted_frames=sum(r.get('frames',0) for r in ledger.values()),failed_sequences=sum(r['status']=='FAILED' for r in ledger.values()),
        device=a.device,elapsed_s=time.monotonic()-started,checked_at=time.time(),training_allowed=False))


if __name__=='__main__':main()
