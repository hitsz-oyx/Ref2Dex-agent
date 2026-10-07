"""Bounded FoundationPose pilot in the native EgoDex stationary origin.

Invalid estimates remain NaN/masked. Only windows with audited continuous
poses qualify; the official test split is never admitted to training.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree, ConvexHull, QhullError, distance

from native_data import SEMANTICS, eligible_rows, rigid_valid, sha
from prepare_egodex_hands import camera_object_to_origin, native_hands
from prepare_native import sample_surface
from object_geometry import exact_mesh_diameter


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--video',type=Path,required=True)
    p.add_argument('--hdf5',type=Path,required=True)
    p.add_argument('--mesh',type=Path,required=True)
    p.add_argument('--masks',type=Path,required=True)
    p.add_argument('--depth',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--sequence',default='egodex_test_basic_pick_place_101')
    p.add_argument('--object-key',default='egodex_tape_measure')
    p.add_argument('--seconds',type=int,default=1200)
    p.add_argument('--max-registrations',type=int,default=8)
    p.add_argument('--register-iterations',type=int,default=5)
    a=p.parse_args();a.root=a.root.resolve();a.output.mkdir(parents=True,exist_ok=False)
    if not 1<=a.max_registrations<=12 or not 1<=a.register_iterations<=5:
        p.error('registration budget=1..12 and refinement iterations=1..5')
    os.environ['PYOPENGL_PLATFORM']='egl'
    # Trusted official checkpoint provenance is recorded by acquisition manifest.
    os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1'
    os.environ['TMPDIR']=str(Path('tmp/ref5-data-expansion').resolve())
    os.environ['WARP_CACHE_PATH']=str(a.root/'warp-cache')
    start=time.monotonic();torch.set_num_threads(2);cv2.setNumThreads(2)
    vendor=a.root/'vendor/ObjectForesight-Data'
    sys.path.insert(0,str(vendor));sys.path.insert(0,str(vendor/'FoundationPose'))
    import estimater as foundationpose_module
    # The upstream random 10000-point broadcast stalls on this generated mesh.
    # This process-local replacement computes the same Euclidean geometry
    # exactly on the convex hull, without modifying the vendored estimator.
    foundationpose_module.compute_mesh_diameter = exact_mesh_diameter
    FoundationPose = foundationpose_module.FoundationPose
    from Utils import depth2xyzmap, nvdiffrast_render
    import nvdiffrast.torch as dr
    report=dict(status='RUNNING',stage='FOUNDATIONPOSE',training_allowed=False,
                label_type='estimated_object_pose',coordinate_frame='stationary_ARKit_origin',
                script_sha256=sha(Path(__file__)),input_sha256={str(x):sha(x) for x in
                (a.video,a.hdf5,a.mesh,a.masks,a.depth)})
    try:
        d=np.load(a.depth);ids=d['source_frame_ids'];depth=d['depth'];K=d['intrinsics']
        cameras=d['camera_to_origin'];H,W=depth.shape[1:]
        if len(ids)<28 or not np.all(np.diff(ids)==1):raise ValueError('noncontiguous native clock')
        raw=np.load(a.masks)['masks'][ids]
        masks=np.stack([cv2.resize(m.astype('uint8'),(W,H),interpolation=cv2.INTER_NEAREST)>0 for m in raw])
        with h5py.File(a.hdf5,'r') as hf:
            hands,hvalid,conf,known,names=native_hands(hf)
            hands,hvalid,conf,known=hands[ids],hvalid[ids],conf[ids],known[ids]
        mesh=trimesh.load(a.mesh,force='mesh',process=False)
        xyz=depth2xyzmap(depth[0],K)
        scene=xyz[masks[0]&(depth[0]>.001)]
        if len(scene)<50:raise ValueError('insufficient object depth for scale')
        lo,hi=np.quantile(scene[:,2],[.1,.9]);scene=scene[(scene[:,2]>=lo)&(scene[:,2]<=hi)]
        centered=scene-scene.mean(0)
        axis=np.linalg.svd(centered,full_matrices=False)[2][0]
        span=float(np.ptp(centered@axis))
        try:
            hull=scene[ConvexHull(scene).vertices]
            span=max(span,float(distance.pdist(hull).max()))
        except QhullError:
            # Coplanar visible faces still have a valid 2D diameter.
            projected=centered@np.linalg.svd(centered,full_matrices=False)[2][:2].T
            hull=scene[ConvexHull(projected).vertices]
            span=max(span,float(distance.pdist(hull).max()))
        # Match the visible cloud diameter to the mesh's Euclidean diameter.
        # A bounding-box diagonal can be larger than any actual vertex pair.
        diameter=exact_mesh_diameter(mesh.vertices)
        metric_scale=span/diameter
        if not .005< span <.5 or not np.isfinite(metric_scale):raise ValueError(f'object scale outside pilot: {span}')
        mesh.apply_scale(metric_scale);mesh.export(a.output/'mesh_metric.ply')
        est=FoundationPose(model_pts=mesh.vertices,model_normals=mesh.vertex_normals,
                           mesh=mesh,debug=0,debug_dir=str(a.output/'foundationpose_debug'),
                           glctx=dr.RasterizeCudaContext(),iou_min=.2)
        cap=cv2.VideoCapture(str(a.video));cap.set(cv2.CAP_PROP_POS_FRAMES,int(ids[0]))
        poses=np.full((len(ids),1,4,4),np.nan,dtype='float32')
        pose_camera=np.full((len(ids),4,4),np.nan,dtype='float32')
        ious=np.zeros(len(ids),'float32');quality=np.zeros(len(ids),bool)
        depth_rmse=np.full(len(ids),np.nan,dtype='float32')
        panels=[];register=True;attempts=0
        for t,fid in enumerate(ids):
            if time.monotonic()-start>a.seconds:raise TimeoutError('pose pilot deadline')
            ok,bgr=cap.read()
            if not ok:raise ValueError('video decode failed')
            bgr=cv2.resize(bgr,(W,H));rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
            if (masks[t]&(depth[t]>.001)).sum()<30:
                register=True;continue
            if register:
                if attempts>=a.max_registrations:break
                attempts+=1
                pose=est.register(K=K,rgb=rgb,depth=depth[t],ob_mask=masks[t],iteration=a.register_iterations)
            else:
                pose=est.track_one(rgb=rgb,depth=depth[t],K=K,iteration=2,ob_mask=masks[t])
            if pose is None or not rigid_valid(np.asarray(pose)):
                register=True;continue
            centered_pose=torch.as_tensor(pose,device='cuda',dtype=torch.float32) @ torch.linalg.inv(est.get_tf_to_centered_mesh())
            with torch.inference_mode():
                _,rendered_depth,_=nvdiffrast_render(K=K,H=H,W=W,ob_in_cams=centered_pose[None],
                    glctx=est.glctx,mesh_tensors=est.mesh_tensors,output_size=[H,W],use_light=False)
                rendered_depth=rendered_depth.detach().cpu().numpy().reshape(H,W)
                silhouette=rendered_depth>0
            iou=float((silhouette&masks[t]).sum()/max(1,(silhouette|masks[t]).sum()))
            intersection=silhouette&masks[t]&(depth[t]>.001)
            rmse=float(np.sqrt(np.mean((rendered_depth[intersection]-depth[t][intersection])**2))) if intersection.sum()>=20 else np.inf
            world=camera_object_to_origin(cameras[t],pose)
            poses[t,0]=world;pose_camera[t]=pose;ious[t]=iou;depth_rmse[t]=rmse
            quality[t]=iou>=.25 and rmse<=max(.025,.15*est.diameter)
            register=not quality[t]
            if t in [0,len(ids)//4,len(ids)//2,3*len(ids)//4,len(ids)-1]:
                viz=bgr.copy();viz[masks[t]]=(viz[masks[t]]*.5+[20,20,127]).clip(0,255)
                cv2.drawContours(viz,cv2.findContours(silhouette.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0],-1,(0,255,0),1)
                cv2.putText(viz,f'frame {fid} IoU {iou:.2f}',(5,22),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
                panels.append(viz)
            print(json.dumps(dict(frame=int(fid),iou=iou,depth_rmse_m=rmse,valid=bool(quality[t]))),flush=True)
        cap.release()
        pose_valid=rigid_valid(poses)&quality[:,None]
        hand_valid=hvalid.all(-1);hands[~hand_valid]=0
        cloud=sample_surface(mesh,a.object_key)
        near=np.zeros((len(ids),1),bool)
        hand_distance=np.full(len(ids),np.nan,dtype='float32')
        for t in np.flatnonzero(pose_valid[:,0]):
            world=cloud['points']@poses[t,0,:3,:3].T+poses[t,0,:3,3]
            distances=cKDTree(world).query(hands[t].reshape(22,3))[0].reshape(2,11)
            hand_distance[t]=np.min(np.where(hand_valid[t,:,None],distances,np.inf))
            near[t,0]=hand_distance[t]<.05
        rows=eligible_rows(hands,hand_valid,poses,pose_valid,near,ids/30,cloud['center'][None])
        np.savez_compressed(a.output/'object_poses.npz',poses=poses,object_to_camera=pose_camera,
                            pose_valid=pose_valid,iou=ious,depth_rmse_m=depth_rmse,
                            nearest_native_hand_m=hand_distance,source_frame_ids=ids,timestamps=ids/30)
        if panels:cv2.imwrite(str(a.output/'pose_review.jpg'),np.concatenate(panels,axis=1))
        root=a.output/'processed';seq=a.sequence
        dest=root/'sequences'/seq;dest.mkdir(parents=True)
        canon=root/'canonical';canon.mkdir()
        key=a.object_key;np.savez_compressed(canon/(key+'.npz'),**cloud,
                mesh_path=str((a.output/'mesh_metric.ply').resolve()),mesh_sha256=sha(a.output/'mesh_metric.ply'))
        arrays=dict(hand=hands,hand_valid=hand_valid,poses=poses,pose_valid=pose_valid,
                    program=np.zeros_like(near),near=near,frame_ids=ids,source_frame_ids=ids,
                    timestamps=ids/30,centers=cloud['center'][None],
                    hand_point_valid=hvalid,hand_confidence=conf,confidence_known=known,
                    object_mask_iou=ious,object_depth_rmse_m=depth_rmse)
        for k,v in arrays.items():np.save(dest/(k+'.npy'),v)
        record=dict(sequence=seq,source='egodex',split='test',objects=[key],program_available=False,
                    effect_label='estimated_ObjectForesight',training_allowed=False,eligible_windows=len(rows))
        (dest/'meta.json').write_text(json.dumps(record,indent=2)+'\n')
        for split in ['train','val','test']:
            index=np.column_stack((np.zeros(len(rows),dtype='int64'),rows)) if split=='test' else np.zeros((0,4),'int64')
            np.save(root/f'index_{split}.npy',index)
        manifest=dict(schema='ref2dex.native-wm30.v1',status='COMPLETED',fps=30,horizon=24,history=4,
                      hand_order=['right','left'],hand_semantics=SEMANTICS,units='m',training_allowed=False,
                      sequences=[seq],records=[record],splits=dict(train=[],val=[],test=[seq]))
        (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        report.update(status='COMPLETED',frames=len(ids),valid_pose_frames=int(pose_valid.sum()),
                      accepted_windows=len(rows),registrations=attempts,object_visible_span_m=span,
                      registration_budget=a.max_registrations,register_iterations=a.register_iterations,
                      normalized_mesh_to_metric_scale=metric_scale,
                      diameter_method='exact convex hull with bounded distance blocks',
                      median_iou=float(np.median(ious[quality])) if quality.any() else None,
                      median_depth_rmse_m=float(np.median(depth_rmse[quality])) if quality.any() else None,
                      median_nearest_hand_m=float(np.nanmedian(hand_distance)) if np.isfinite(hand_distance).any() else None,
                      verdict='UNCLEAR',limitations=['estimated depth/mesh scale','single rigid object engineering probe',
                      'continuous quality acceptance and native hand/object alignment require review'])
    except Exception as e:
        report.update(status='FAILED',error=repr(e));raise
    finally:
        report['elapsed_s']=time.monotonic()-start
        (a.output/'pose_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
