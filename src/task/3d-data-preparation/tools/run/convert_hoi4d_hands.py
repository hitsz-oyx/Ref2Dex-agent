"""Cache native HOI4D MANO joints in camera coordinates while pose admission is audited."""
import argparse
from collections import defaultdict
import io
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import time
import zipfile
import numpy as np
import torch


def write(path,value):
    temp=path.with_suffix('.partial.json');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def models(root,device):
    for name,value in [('bool',bool),('int',int),('float',float),('complex',complex),('object',object),('str',str),('unicode',str)]:
        if name not in np.__dict__:setattr(np,name,value)
    import manopth
    outer=Path(manopth.__file__).resolve().parents[1]
    if (outer/'mano').is_dir() and str(outer) not in manopth.__path__:manopth.__path__.append(str(outer))
    from manopth.manopth.manolayer import ManoLayer
    return {s:ManoLayer(mano_root=str(root),use_pca=False,ncomps=45,side=s,center_idx=None,flat_hand_mean=True).to(device)
            for s in ['right','left']}


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mano-root',type=Path,required=True);p.add_argument('--device',choices=['cpu','cuda:0'],default='cuda:0')
    p.add_argument('--limit',type=int);p.add_argument('--seconds',type=int,default=7200);a=p.parse_args()
    if not 60<=a.seconds<=86400:p.error('bounded deadline required')
    root=a.output.resolve();root.mkdir(parents=True,exist_ok=True);(root/'sequences').mkdir(exist_ok=True)
    started=time.monotonic();deadline=started+a.seconds;torch.set_num_threads(4)
    if a.device!='cpu':
        gpu=os.environ.get('CUDA_VISIBLE_DEVICES','')
        if not gpu.isdigit():raise RuntimeError('single physical GPU required')
        memory=int(subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
        if memory>64:raise RuntimeError('GPU already occupied; existing tasks preserved')
        if not torch.cuda.is_available():raise RuntimeError('requested GPU unavailable')
    layers=models(a.mano_root,a.device)
    manifest=json.loads((root/'manifest.json').read_text()) if (root/'manifest.json').exists() else dict(
        schema='ref2dex.hoi4d-native-hand.v1',units='m',reference_frame='per-frame moving camera',training_allowed=False,
        profile='hoi4d-author-mano-joint21',mano=dict(use_pca=False,flat_hand_mean=True,center_idx=None),sequences=[],
        limitations=['Hand-only intermediate; no common world/object pose admission yet.',
            'Native integer frame identities retained; no interpolated frames or inferred timestamps.'])
    ledger=json.loads((root/'conversion-ledger.json').read_text()) if (root/'conversion-ledger.json').exists() else {}
    phase='CONVERTING';all_errors=[]
    with zipfile.ZipFile(a.input/'archives/HOI4D_Hand_pose.zip') as hands,zipfile.ZipFile(a.input/'archives/camera_params.zip') as cameras:
        groups=defaultdict(lambda:defaultdict(list))
        for entry in hands.infolist():
            if not entry.filename.endswith('.pickle'):continue
            parts=entry.filename.split('/');side=parts[1][len('handpose_'):-len('_hand')]
            if side not in ('right','left') or len(parts)!=10:raise ValueError('unknown native hand archive path')
            identity='/'.join(parts[2:-1]);groups[identity][side].append((int(Path(parts[-1]).stem),entry.filename))
        identities=sorted(groups);identities=identities[:a.limit] if a.limit else identities
        for identity in identities:
            if identity in ledger:continue
            if time.monotonic()>deadline or (root/'STOP').exists() or shutil.disk_usage(root).free<20*2**30:
                phase='STOPPED';break
            try:
                native=groups[identity];ids=sorted({i for side in native.values() for i,_ in side});n=len(ids)
                if not 1<=n<=20000:raise ValueError('bounded native sequence required')
                offsets={i:k for k,i in enumerate(ids)}
                points=np.full((n,2,21,3),np.nan,np.float32);valid=np.zeros((n,2,21),bool)
                params=np.full((n,2,61),np.nan,np.float32);kps=np.full((n,2,21,2),np.nan,np.float32)
                projection=np.full((n,2,21),np.nan,np.float32)
                camera=identity.split('/')[0];intrinsic=np.load(io.BytesIO(cameras.read('camera_params/'+camera+'/intrin.npy')),allow_pickle=False)
                for slot,side in enumerate(['right','left']):
                    members=sorted(native.get(side,[]));seen=set();batch=[]
                    for frame,member in members:
                        if frame in seen:raise ValueError('duplicate native hand frame')
                        seen.add(frame);data=pickle.loads(hands.read(member),encoding='latin1')
                        theta=np.asarray(data['poseCoeff'],np.float32);beta=np.asarray(data['beta'],np.float32);trans=np.asarray(data['trans'],np.float32)
                        if theta.shape!=(48,) or beta.shape!=(10,) or trans.shape!=(3,):raise ValueError('MANO parameter shape mismatch')
                        k=offsets[frame];params[k,slot]=np.r_[theta,beta,trans]
                        if 'kps2D' in data:
                            if np.shape(data['kps2D'])!=(21,2):raise ValueError('native projection shape mismatch')
                            kps[k,slot]=data['kps2D']
                        if np.isfinite(params[k,slot]).all():batch.append(k)
                    for start in range(0,len(batch),128):
                        chosen=batch[start:start+128];native_params=params[chosen,slot]
                        theta=torch.from_numpy(native_params[:,:48].copy()).to(a.device)
                        beta=torch.from_numpy(native_params[:,48:58].copy()).to(a.device)
                        with torch.no_grad():_,joints,_=layers[side](theta,beta)
                        result=joints.cpu().numpy()/1000+native_params[:,None,58:]
                        points[chosen,slot]=result;valid[chosen,slot]=np.isfinite(result).all(-1)
                        screen=result@intrinsic.T
                        with np.errstate(divide='ignore',invalid='ignore'):screen=screen[...,:2]/screen[...,2:]
                        projection[chosen,slot]=np.linalg.norm(screen-kps[chosen,slot],axis=-1)
                filename=identity.replace('/','__')+'.npz';target=root/'sequences'/filename
                if target.exists():raise ValueError('unpublished sequence exists; never overwrite')
                pending=root/'sequences'/('pending-'+filename)
                np.savez_compressed(pending,native_frame_id=np.array(ids,np.int64),hand_camera=points,hand_valid=valid,
                    native_mano_camera=params,native_kps2d=kps,reprojection_error_px=projection,camera_intrinsic=intrinsic)
                pending.replace(target)
                finite=projection[np.isfinite(projection)]
                row=dict(source_sequence=identity,path='sequences/'+filename,frames=n,annotated_hand_frames=int(valid.any(-1).sum()),
                    p95_reprojection_error_px=float(np.quantile(finite,.95)) if len(finite) else None,
                    max_reprojection_error_px=float(finite.max()) if len(finite) else None)
                manifest['sequences'].append(row);ledger[identity]=dict(status='CONVERTED',**row)
                print('CONVERTED',identity,n,flush=True)
            except Exception as error:
                ledger[identity]=dict(status='FAILED',error_type=type(error).__name__,reason=str(error));all_errors.append(identity)
                print('FAILED',identity,type(error).__name__,flush=True)
            write(root/'manifest.json',manifest);write(root/'conversion-ledger.json',ledger)
            elapsed=time.monotonic()-started
            write(root/'progress.json',dict(status='CONVERTING',pid=os.getpid(),device=a.device,total_sequences=len(identities),
                converted_sequences=len(manifest['sequences']),converted_frames=sum(v['frames'] for v in manifest['sequences']),
                failed_sequences=sum(v['status']=='FAILED' for v in ledger.values()),elapsed_s=elapsed,
                estimated_remaining_minutes=(len(identities)-len(ledger))*elapsed/max(len(ledger),1)/60,
                gpu_allocated_mib=torch.cuda.memory_allocated()/2**20 if a.device!='cpu' else 0,
                checked_at=time.time(),training_allowed=False))
    write(root/'progress.json',dict(status='FINISHED' if phase!='STOPPED' and not all_errors else ('PARTIAL' if all_errors else phase),pid=os.getpid(),device=a.device,
        converted_sequences=len(manifest['sequences']),converted_frames=sum(v['frames'] for v in manifest['sequences']),
        failed_sequences=sum(v['status']=='FAILED' for v in ledger.values()),elapsed_s=time.monotonic()-started,checked_at=time.time(),training_allowed=False))


if __name__=='__main__':main()
