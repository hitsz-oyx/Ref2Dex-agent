"""Audit estimated EgoDex labels against native clocks/cameras and WM geometry.

Engineering consistency does not establish reconstruction accuracy. The
official test split remains unavailable for training even when checks pass.
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import trimesh

from native_data import NativeWindows, rigid_valid, sha
from prepare_egodex_hands import native_hands
from oakink_wm.data import collate, transform_points


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--hdf5', type=Path, required=True)
    p.add_argument('--depth', type=Path, required=True)
    a = p.parse_args()
    target = a.input/'object_audit.json'
    if target.exists():
        raise FileExistsError(target)
    torch.set_num_threads(2)
    meta = json.loads((a.input/'processed/manifest.json').read_text())
    run = json.loads((a.input/'pose_manifest.json').read_text())
    if run['status'] != 'COMPLETED' or meta['training_allowed'] is not False:
        raise ValueError('incomplete or training-enabled engineering labels')
    if len(np.load(a.input/'processed/index_train.npy')) or len(np.load(a.input/'processed/index_val.npy')):
        raise ValueError('official test leaked into training/selection')
    result = dict(status='RUNNING', training_allowed=False,
                  label_type='estimated_object_pose', accuracy_verified=False,
                  script_sha256=sha(Path(__file__)), windows=0,
                  input_sha256={str(x): sha(x) for x in (a.hdf5, a.depth)},
                  maximum_world_composition_error=0., maximum_correspondence_error_m=0.)
    with np.load(a.input/'object_poses.npz') as poses, np.load(a.depth) as depth, h5py.File(a.hdf5) as hf:
        ids = poses['source_frame_ids']
        if not np.array_equal(ids, depth['source_frame_ids']) or not np.all(np.diff(ids) == 1):
            raise ValueError('source clock mismatch')
        if not np.allclose(poses['timestamps'], ids/30, atol=1e-8):
            raise ValueError('wrong physical timestamps')
        cameras = np.asarray(hf['transforms/camera'])[ids]
        if not np.allclose(depth['camera_to_origin'], cameras, atol=1e-7):
            raise ValueError('learned camera replaced native camera')
        valid = poses['pose_valid'][:, 0]
        if not rigid_valid(poses['poses'][valid]).all():
            raise ValueError('invalid accepted SE(3)')
        if valid.any():
            expected = cameras[valid] @ poses['object_to_camera'][valid]
            err = float(np.abs(expected - poses['poses'][valid, 0]).max())
            if err > 2e-6:
                raise ValueError('object/native-camera composition mismatch')
            result['maximum_world_composition_error'] = err
        result['valid_pose_frames'] = int(valid.sum())
        result['frames'] = len(ids)
        for name in ('iou', 'depth_rmse_m', 'nearest_native_hand_m'):
            values = poses[name][valid]
            values = values[np.isfinite(values)]
            result[name+'_quantiles'] = np.quantile(values, [.05, .5, .95]).tolist() if len(values) else None
        native, native_valid, confidence, known, _ = native_hands(hf)
        seq = a.input/'processed/sequences'/meta['sequences'][0]
        present = native_valid[ids].all(-1)
        hand = np.load(seq/'hand.npy')
        if not np.array_equal(np.load(seq/'hand_valid.npy'), present):
            raise ValueError('native validity semantics changed')
        if not np.allclose(hand[present], native[ids][present], atol=1e-7) or np.any(hand[~present] != 0):
            raise ValueError('native hand points changed or absent hand unmasked')
        if not np.array_equal(np.load(seq/'confidence_known.npy'), known[ids]):
            raise ValueError('unknown confidence fabricated')
        if not np.allclose(np.load(seq/'hand_confidence.npy'), confidence[ids], equal_nan=True):
            raise ValueError('native confidence changed')
    cloud_path = next((a.input/'processed/canonical').glob('*.npz'))
    with np.load(cloud_path) as cloud:
        mesh = trimesh.load(a.input/'mesh_metric.ply', force='mesh', process=False)
        expected = (mesh.vertices[mesh.faces[cloud['face_indices']]] * cloud['barycentric'][..., None]).sum(1)
        if np.linalg.norm(cloud['points']-expected, axis=-1).max() > 1e-6:
            raise ValueError('canonical surface provenance mismatch')
    count = len(np.load(a.input/'processed/index_test.npy'))
    data = NativeWindows(a.input, 'test') if count else None
    samples = []
    for i in range(count):
        sample = data[i]
        if not all(np.isfinite(v).all() for v in sample.values() if hasattr(v, 'dtype')):
            raise ValueError('nonfinite accepted WM tensor')
        seqidx, anchor, tick = sample['sample_id']
        sequence = data.sequence(data.sequences[seqidx])
        if not sequence['near'][tick, anchor]:
            raise ValueError('anchor selected without current proximity')
        future = sequence['poses'][tick+1:tick+25, anchor]
        current = sequence['poses'][tick, anchor]
        displacement = np.linalg.norm(future[:,:3,3]-current[:3,3],axis=-1)
        rotation = future[:,:3,:3] @ current[:3,:3].T
        angle = np.arccos(np.clip((np.trace(rotation,axis1=1,axis2=2)-1)/2,-1,1))
        expected_category = 1 if (displacement>.002).any() or (angle>.02).any() else 2
        if sample['category'] != expected_category:
            raise ValueError(f'full-horizon motion category mismatch: tick{tick}')
        canonical = data.canonical(sequence['objects'][anchor])[0]
        C = np.linalg.inv(sequence['poses'][tick, anchor])
        for h in range(24):
            actual = transform_points(sample['points'][0], sample['effect'][0, h])
            expected = transform_points(canonical, C @ sequence['poses'][tick+h+1, anchor])
            err = float(np.linalg.norm(actual-expected, axis=-1).max())
            result['maximum_correspondence_error_m'] = max(result['maximum_correspondence_error_m'], err)
        if len(samples) < 4:
            samples.append(sample)
    if samples:
        collate(samples)
    if result['maximum_correspondence_error_m'] > 2e-6:
        raise ValueError('fixed-current-frame effect mismatch')
    result.update(status='COMPLETED', windows=count,
                  verdict='ENGINEERING_PASS' if count else 'NO_USABLE_WINDOWS',
                  limitations=['estimated mesh/depth scale', 'visual reconstruction accuracy requires inspection',
                               'test engineering clips cannot be used for fitting or checkpoint selection'])
    target.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
