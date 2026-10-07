"""Convert original ObjectForesight camera poses to the native EgoDex schema.

Only quality auditing, coordinate composition and output formatting happen
here. All object scale, registration and tracking decisions belong upstream.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree

from native_data import SEMANTICS, eligible_rows, rigid_valid, sha
from prepare_egodex_hands import native_hands
from prepare_native import sample_surface
from object_geometry import exact_mesh_diameter


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['root', 'input', 'output', 'hdf5', 'depth', 'masks']:
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--sequence', required=True)
    p.add_argument('--object-key', required=True)
    a = p.parse_args()
    upstream = json.loads((a.input/'upstream_run_manifest.json').read_text())
    if upstream['status'] != 'COMPLETED':
        raise ValueError('original tracking did not complete; preserve diagnostic run')
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    vendor = a.root.resolve()/'vendor/ObjectForesight-Data/FoundationPose'
    sys.path.insert(0, str(vendor))
    from Utils import make_mesh_tensors, nvdiffrast_render
    import nvdiffrast.torch as dr
    d = np.load(a.depth)
    ids, depths, K, cameras = (d[k] for k in
                              ['source_frame_ids', 'depth', 'intrinsics', 'camera_to_origin'])
    with np.load(a.input/'clip/native_calibration.npz') as native:
        if not np.array_equal(ids, native['source_frame_ids']):
            raise ValueError('staged source clock changed')
        if not np.array_equal(cameras, native['camera_to_origin']):
            raise ValueError('staged native cameras changed')
    raw_masks = np.load(a.masks)['masks']
    H, W = depths.shape[1:]
    masks = np.stack([cv2.resize(raw_masks[i].astype('uint8'), (W,H),
                               interpolation=cv2.INTER_NEAREST)>0 for i in ids])
    mesh = trimesh.load(a.input/'mesh_metric.ply', force='mesh', process=False)
    mesh.export(a.output/'mesh_metric.ply')
    diameter = exact_mesh_diameter(mesh.vertices)
    tensors, context = make_mesh_tensors(mesh), dr.RasterizeCudaContext()
    poses_camera = np.full((len(ids),4,4), np.nan, dtype='float32')
    poses = np.full((len(ids),1,4,4), np.nan, dtype='float32')
    ious = np.zeros(len(ids),dtype='float32')
    rmses = np.full(len(ids),np.nan,dtype='float32')
    valid = np.zeros((len(ids),1),dtype=bool)
    folder = a.input/'clip/objects/object_0/foundationpose10/ob_in_cam'
    for local, fid in enumerate(ids):
        source = folder/f'{local}.txt'
        if not source.exists():
            continue
        pose = np.loadtxt(source).astype('float32')
        if pose.shape != (4,4) or not rigid_valid(pose):
            continue
        poses_camera[local] = pose
        poses[local,0] = cameras[local] @ pose
        with torch.inference_mode():
            _, rendered_depth, _ = nvdiffrast_render(
                K=K,H=H,W=W,ob_in_cams=torch.as_tensor(pose,device='cuda')[None],
                glctx=context,mesh_tensors=tensors,output_size=[H,W],use_light=False)
        depth_render = rendered_depth.detach().cpu().numpy().reshape(H,W)
        silhouette, mask = depth_render>0, masks[local]
        ious[local] = (silhouette & mask).sum()/max(1,(silhouette | mask).sum())
        overlap = silhouette & mask & (depths[local]>.001)
        if overlap.sum() >= 20:
            rmses[local] = np.sqrt(np.mean((depth_render[overlap]-depths[local][overlap])**2))
        valid[local,0] = ious[local]>=.25 and rmses[local]<=max(.025,.15*diameter)
    with h5py.File(a.hdf5) as hf:
        hand, point_valid, confidence, known, _ = native_hands(hf)
        hand, point_valid, confidence, known = (x[ids] for x in
                                               [hand,point_valid,confidence,known])
        if not np.allclose(cameras, np.asarray(hf['transforms/camera'])[ids], atol=1e-7):
            raise ValueError('estimated camera substituted for native camera')
    hand_valid = point_valid.all(-1)
    hand[~hand_valid] = 0
    cloud = sample_surface(mesh, a.object_key)
    near = np.zeros((len(ids),1),dtype=bool)
    hand_distance = np.full(len(ids),np.nan,dtype='float32')
    for t in np.flatnonzero(valid[:,0]):
        world = cloud['points']@poses[t,0,:3,:3].T+poses[t,0,:3,3]
        distances = cKDTree(world).query(hand[t].reshape(22,3))[0].reshape(2,11)
        hand_distance[t] = np.min(np.where(hand_valid[t,:,None],distances,np.inf))
        near[t,0] = hand_distance[t]<.05
    rows = eligible_rows(hand,hand_valid,poses,valid,near,ids/30,cloud['center'][None])
    np.savez_compressed(a.output/'object_poses.npz', poses=poses,
                        object_to_camera=poses_camera,pose_valid=valid,iou=ious,
                        depth_rmse_m=rmses,nearest_native_hand_m=hand_distance,
                        source_frame_ids=ids,timestamps=ids/30)
    root = a.output/'processed'
    seq = root/'sequences'/a.sequence
    seq.mkdir(parents=True)
    (root/'canonical').mkdir()
    np.savez_compressed(root/'canonical'/f'{a.object_key}.npz', **cloud,
                        mesh_path=str((a.output/'mesh_metric.ply').resolve()),
                        mesh_sha256=sha(a.output/'mesh_metric.ply'))
    arrays = dict(hand=hand,hand_valid=hand_valid,poses=poses,pose_valid=valid,
                  program=np.zeros_like(near),near=near,frame_ids=ids,source_frame_ids=ids,
                  timestamps=ids/30,centers=cloud['center'][None],
                  hand_point_valid=point_valid,hand_confidence=confidence,confidence_known=known,
                  object_mask_iou=ious,object_depth_rmse_m=rmses)
    for name, value in arrays.items():
        np.save(seq/f'{name}.npy', value)
    record = dict(sequence=a.sequence,source='egodex',split='test',objects=[a.object_key],
                  program_available=False,effect_label='estimated_ObjectForesight_original_step10',
                  training_allowed=False,eligible_windows=len(rows))
    (seq/'meta.json').write_text(json.dumps(record,indent=2)+'\n')
    for split in ['train','val','test']:
        index = np.column_stack((np.zeros(len(rows),dtype='int64'),rows)) if split=='test' else np.zeros((0,4),dtype='int64')
        np.save(root/f'index_{split}.npy',index)
    manifest = dict(schema='ref2dex.native-wm30.v1',status='COMPLETED',fps=30,horizon=24,history=4,
                    hand_order=['right','left'],hand_semantics=SEMANTICS,units='m',training_allowed=False,
                    sequences=[a.sequence],records=[record],splits=dict(train=[],val=[],test=[a.sequence]))
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    report = dict(status='COMPLETED',stage='ORIGINAL_STEP10_NATIVE_EXPORT',frames=len(ids),
                  valid_pose_frames=int(valid.sum()),accepted_windows=len(rows),training_allowed=False,
                  accuracy_verified=False,metric_diameter_m=diameter,
                  upstream_manifest_sha256=sha(a.input/'upstream_run_manifest.json'),
                  script_sha256=sha(Path(__file__)),
                  quality_gate=dict(iou_min=.25,depth_rmse_max_m=max(.025,.15*diameter)),
                  native_camera_preserved=True)
    (a.output/'pose_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__ == '__main__':
    main()
