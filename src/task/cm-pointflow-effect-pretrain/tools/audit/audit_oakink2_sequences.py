#!/usr/bin/env python3
"""100-sequence GPU MANO/SE(3) data audit; no predictor training or interpolation."""
import argparse, ast, hashlib, json, pickle, time
from pathlib import Path
import numpy as np
# Old licensed MANO files contain chumpy; isolate NumPy compatibility in this process.
for name,value in [('bool',bool),('int',int),('float',float),('complex',complex),('object',object),('unicode',str),('str',str)]:
    if name not in np.__dict__: setattr(np,name,value)
import torch
import trimesh
from manotorch.manolayer import ManoLayer

KEYPOINTS=[0,1,5,9,13,17,4,8,12,16,20]  # wrist + five MCP + five fingertips


def summary(x):
    x=np.asarray(x);x=x[np.isfinite(x)]
    return dict(count=len(x),quantiles=np.quantile(x,[0,.5,.9,.99,1]).tolist()) if len(x) else dict(count=0,quantiles=[])


def dense_mano(anno,ids,layers,device):
    n=len(ids);xyz=np.zeros((n,2,11,3),np.float32);valid=np.zeros((n,2),bool);wrist_error=0.
    for side,(prefix,layer) in enumerate(zip(('rh__','lh__'),layers)):
        values=[];indices=[]
        for tick,fid in enumerate(ids):
            row=anno['raw_mano'].get(fid,{})
            if any(prefix+k not in row for k in ('pose_coeffs','betas','tsl')):continue
            pose,beta,tsl=[np.asarray(row[prefix+k]).reshape(-1) for k in ('pose_coeffs','betas','tsl')]
            if pose.size!=64 or beta.size!=10 or tsl.size!=3:continue
            if not all(np.isfinite(v).all() for v in (pose,beta,tsl)):continue
            if np.any(np.linalg.norm(pose.reshape(16,4),axis=-1)<1e-6):continue
            values.append((pose.reshape(16,4),beta,tsl));indices.append(tick)
        for begin in range(0,len(values),512):
            batch=values[begin:begin+512];idx=indices[begin:begin+512]
            pose=torch.as_tensor(np.stack([v[0] for v in batch]),device=device)
            beta=torch.as_tensor(np.stack([v[1] for v in batch]),device=device)
            tsl=torch.as_tensor(np.stack([v[2] for v in batch]),device=device)
            with torch.no_grad():j=layer(pose_coeffs=pose,betas=beta).joints+tsl[:,None]
            if not torch.isfinite(j).all():raise ValueError('nonfinite MANO reconstruction')
            wrist_error=max(wrist_error,float((j[:,0]-tsl).abs().max()))
            xyz[idx,side]=j[:,KEYPOINTS].cpu().numpy();valid[idx,side]=True
    assert wrist_error<1e-5
    return xyz,valid,wrist_error


def canonical_points(dataset,obj,cache):
    if obj in cache:return cache[obj]
    paths=list((dataset/'object_repair/align_ds'/obj).glob('*.ply'))+list((dataset/'object_repair/align_ds'/obj).glob('*.obj'))
    if not paths:paths=list((dataset/'object_raw/align_ds'/obj).glob('*.ply'))+list((dataset/'object_raw/align_ds'/obj).glob('*.obj'))
    if not paths:raise FileNotFoundError('no mesh '+obj)
    mesh=trimesh.load(paths[0],process=False,force='mesh',skip_materials=True)
    tri=np.asarray(mesh.triangles,dtype=np.float64)
    area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=-1)/2
    if not len(tri) or not np.isfinite(tri).all() or area.sum()<=0:raise ValueError('invalid mesh '+obj)
    rng=np.random.default_rng(int(hashlib.sha256(obj.encode()).hexdigest()[:8],16))
    chosen=tri[rng.choice(len(tri),512,p=area/area.sum())]
    u=np.sqrt(rng.random(512));v=rng.random(512)
    points=chosen[:,0]*(1-u[:,None])+chosen[:,1]*(u*(1-v))[:,None]+chosen[:,2]*(u*v)[:,None]
    cache[obj]=(points,paths[0],np.ptp(mesh.vertices,axis=0).tolist())
    return cache[obj]


def program_mask(dataset,seq,obj,ids):
    path=dataset/'program/program_info'/(seq+'.json');mask=np.zeros(len(ids),bool);modes={}
    if not path.exists():return mask,modes,False
    program=json.loads(path.read_text())
    for key,row in program.items():
        if obj not in row.get('obj_list',[]):continue
        mode=row.get('interaction_mode','unknown');modes[mode]=modes.get(mode,0)+1
        for interval in ast.literal_eval(key):
            if interval is not None:mask|=(ids>=interval[0])&(ids<interval[1])
    return mask,modes,True


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--mano-root',type=Path,required=True);p.add_argument('--device',default='cuda:0')
    p.add_argument('--seconds',type=int,default=1200);a=p.parse_args();start=time.monotonic();root=a.run_dir.resolve()
    torch.set_num_threads(2)
    if not torch.cuda.is_available():raise ValueError('GPU MANO audit required')
    layers=[ManoLayer(mano_assets_root=str(a.mano_root),rot_mode='quat',side=side,center_idx=0,use_pca=False,flat_hand_mean=True).to(a.device) for side in ('right','left')]
    manifest=json.loads((root/'download_manifest.json').read_text())
    if manifest['status']!='COMPLETED':raise ValueError('download incomplete')
    out=root/'audit';out.mkdir(exist_ok=False);(out/'sequences').mkdir();(out/'canonical').mkdir()
    reports=[];meshcache={};all_hand=[];all_obj=[];all_rot=[];all_distance=[];misses=[]
    counts=dict(windows=0,moving=0,program_windows=0,near_windows=0,interaction_windows=0,interaction_moving=0)
    max_identity_error=0.
    for name in manifest['selection']:
        if time.monotonic()-start>a.seconds:raise TimeoutError('audit deadline')
        path=root/'download'/name
        with path.open('rb') as f:anno=pickle.load(f)
        ids=np.asarray(anno['mocap_frame_id_list'],dtype=np.int64);n=len(ids)
        assert n>8 and np.all(np.diff(ids)>0)
        hand,hand_valid,wrist_error=dense_mano(anno,ids,layers,a.device)
        np.savez_compressed(out/'sequences'/(path.stem+'.npz'),frame_ids=ids,hand_xyz=hand,hand_valid=hand_valid,keypoint_ids=KEYPOINTS)
        contiguous=np.lib.stride_tricks.sliding_window_view(np.diff(ids)==1,8).all(-1)
        presence=np.lib.stride_tricks.sliding_window_view(hand_valid,9,axis=0)
        stable_mask=(presence==presence[:,:,:1]).all((1,2))&hand_valid[:-8].any(-1)
        base=contiguous&stable_mask
        delta=np.linalg.norm(hand[8:]-hand[:-8],axis=-1)
        moving_hand=np.where(hand_valid[:-8,:,None],delta,np.nan)
        all_hand.extend(moving_hand[base].reshape(-1).tolist())
        sequence=dict(sequence=path.stem,frames=n,valid_hand_frames=hand_valid.sum(0).tolist(),
                      both_hands_frames=int(hand_valid.all(-1).sum()),contiguous_stable_hand_windows=int(base.sum()),wrist_translation_error=wrist_error,objects=[])
        for obj in anno['obj_list']:
            try:points,mesh,extent=canonical_points(root/'dataset',obj,meshcache)
            except Exception as e:misses.append(dict(sequence=path.stem,object=obj,error=str(e)));continue
            if not (out/'canonical'/(obj+'.npz')).exists():np.savez_compressed(out/'canonical'/(obj+'.npz'),points=points,mesh_path=str(mesh))
            values=anno['obj_transf'].get(obj,{})
            T=np.full((n,4,4),np.nan)
            for t,fid in enumerate(ids):
                if int(fid) in values:T[t]=values[int(fid)]
            finite=np.isfinite(T).all((1,2));rot=np.where(finite[:,None,None],T[:,:3,:3],np.eye(3))
            orth=np.abs(rot.transpose(0,2,1)@rot-np.eye(3)).max((1,2));det=np.linalg.det(rot)
            rigid=finite&(orth<1e-3)&(np.abs(det-1)<1e-3)&(np.abs(T[:,3]-[0,0,0,1]).max(-1)<1e-6)
            good=base&np.lib.stride_tricks.sliding_window_view(rigid,9).all(-1)
            programme,modes,has_program=program_mask(root/'dataset',path.stem,obj,ids)
            pw=np.lib.stride_tricks.sliding_window_view(programme,9).all(-1)&good
            displacement=np.linalg.norm(T[8:,:3,3]-T[:-8,:3,3],axis=-1)
            relative=rot[8:]@rot[:-8].transpose(0,2,1)
            angle=np.arccos(np.clip((np.trace(relative,axis1=1,axis2=2)-1)/2,-1,1))
            moving=(displacement>.002)|(angle>.02)
            all_obj.extend(displacement[good].tolist());all_rot.extend(angle[good].tolist())
            # Dense distance reductions for audit only; no KNN architecture.
            distance=[]
            for b in range(0,n,128):
                r=torch.as_tensor(T[b:b+128,:3,:3],device=a.device,dtype=torch.float32)
                t=torch.as_tensor(T[b:b+128,:3,3],device=a.device,dtype=torch.float32)
                cloud=torch.as_tensor(points,device=a.device,dtype=torch.float32)@r.transpose(1,2)+t[:,None]
                h=torch.as_tensor(hand[b:b+128].reshape(-1,22,3),device=a.device)
                d=torch.cdist(h,cloud).amin(-1).reshape(-1,2,11)
                hv=torch.as_tensor(hand_valid[b:b+128],device=a.device)
                d=d.masked_fill(~hv[:,:,None],float('inf')).amin((1,2))
                distance.extend(d.cpu().numpy().tolist())
            distance=np.asarray(distance);all_distance.extend(distance[np.isfinite(distance)&rigid].tolist())
            near=np.lib.stride_tricks.sliding_window_view(distance<.05,9).any(-1)&good
            interacting=pw|near
            for k,v in [('windows',good),('moving',good&moving),('program_windows',pw),('near_windows',near),('interaction_windows',interacting),('interaction_moving',interacting&moving)]:counts[k]+=int(v.sum())
            probe=np.flatnonzero(good)[::max(1,int(good.sum())//8)][:8]
            error=0.
            for tick in probe:
                E=T[tick+8]@np.linalg.inv(T[tick]);now=points@T[tick,:3,:3].T+T[tick,:3,3]
                future=points@T[tick+8,:3,:3].T+T[tick+8,:3,3]
                error=max(error,float(np.abs(now@E[:3,:3].T+E[:3,3]-future).max()))
            assert error<1e-8;max_identity_error=max(max_identity_error,error)
            np.savez_compressed(out/'sequences'/(path.stem+'__'+obj+'.npz'),object_transform=T,object_valid=rigid,window_valid=good,program_frame=programme,near_window=near)
            sequence['objects'].append(dict(object=obj,mesh=str(mesh),extent_m=extent,rigid_frames=int(rigid.sum()),orthogonality_max=float(orth[finite].max()) if finite.any() else None,
                                           windows=int(good.sum()),moving_windows=int((good&moving).sum()),program_windows=int(pw.sum()),near_windows=int(near.sum()),program_modes=modes,has_program=has_program,
                                           translation_h8=summary(displacement[good]),rotation_h8=summary(angle[good]),hand_object_distance=summary(distance[rigid]),analytic_effect_error=error))
        reports.append(sequence)
        print(json.dumps(dict(sequences=len(reports),total=len(manifest['selection']),windows=counts['windows'],elapsed_seconds=time.monotonic()-start)),flush=True)
        (out/'progress.json').write_text(json.dumps(dict(sequences=len(reports),counts=counts,elapsed_seconds=time.monotonic()-start))+'\n')
    result=dict(status='DATA_AUDIT_COMPLETE',sequences=len(reports),source_revision=manifest['revision'],counts=counts,missing_meshes=misses,
                hand_displacement_h8=summary(all_hand),object_translation_h8=summary(all_obj),object_rotation_h8=summary(all_rot),hand_object_distance=summary(all_distance),
                analytic_effect_error_max=max_identity_error,fps=120,horizon_frames=8,horizon_seconds=8/120,hand_order=['right','left'],keypoint_ids=KEYPOINTS,
                static_thresholds=dict(translation_m=.002,rotation_rad=.02),interaction_threshold_m=.05,
                scope='Observational future hand trajectory -> object effect data readiness, not causal intervention or policy benefit.',
                elapsed_seconds=time.monotonic()-start,device=a.device,trials=reports)
    (out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='trials'}),flush=True)

if __name__=='__main__':main()
