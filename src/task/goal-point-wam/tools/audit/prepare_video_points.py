#!/usr/bin/env python3
"""Build a causal-observation engineering clip from an existing offline teacher.

Only current/past RGB is passed to the observation depth model. Offline full
clip tracks supply labels. Geometry-quality admission is a separate decision.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np

TASK = Path(__file__).resolve().parents[2]
CM = TASK.parent/'cm-pointflow-effect-pretrain'
sys.path.insert(0, str(CM/'src'))
sys.path.insert(0, str(CM/'tools/run'))
sys.path.insert(0, str(TASK/'tools/audit'))
from oakink_wm.pointwam_data import source_identity, sha256, recover_metric_scale, resize_intrinsics, lift_tracks, resample_valid, select_scene
from prepare_pointwam_pilot import decode_clip, depth_from_omega
from prepare_egodex_actor import JOINTS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('hdf5', 'source-manifest', 'teacher', 'depth-vendor', 'depth-checkpoint', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--max-points', type=int, default=2048)
    p.add_argument('--goal-mask', type=Path, help='current-frame target mask at640x360; optional, never selected by future motion')
    a = p.parse_args()
    if a.output.exists() or not 32 <= a.max_points <= 4096:
        raise ValueError('new bounded output required')
    if sha256(a.depth_checkpoint) != 'c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934':
        raise ValueError('unexpected Omega weights')
    vendor_sha = subprocess.check_output(['git', '-C', str(a.depth_vendor), 'rev-parse', 'HEAD'], text=True).strip()
    if vendor_sha != '48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6':
        raise ValueError('unexpected Omega source')
    identity = source_identity(a.hdf5, a.hdf5.with_suffix('.mp4'), a.source_manifest)
    if identity['split'] != 'official_train':
        raise ValueError('official TRAIN only')
    teacher_sha = sha256(a.teacher)
    with np.load(a.teacher, allow_pickle=False) as f:
        t = {k: f[k].copy() for k in f.files}
    frame_ids = t['source_frame_ids']
    tick = int(frame_ids[0])
    if tick < 3 or len(frame_ids) != 11 or not np.array_equal(np.diff(frame_ids), np.full(10, 3)):
        raise ValueError('teacher requires1s/10Hz and at least3 observed history frames')
    with h5py.File(a.hdf5, 'r') as f:
        if 'confidences' not in f:
            raise ValueError('native hand confidence missing')
        names = [[side+j for j in JOINTS] for side in ('right', 'left')]
        hand = np.stack([np.stack([f['transforms'][j][:, :3, 3] for j in side], 1) for side in names], 1)
        hv = np.stack([np.stack([f['confidences'][j][:] >= .5 for j in side], 1) for side in names], 1)
        camera, intrinsic = f['transforms/camera'][:], f['camera/intrinsic'][:]
        session = str(f.attrs['session_name'])
    hv &= np.isfinite(hand).all(-1)
    prefix_ids = np.arange(tick-3, tick+1)
    rgb, original_wh = decode_clip(a.hdf5.with_suffix('.mp4'), prefix_ids, len(hand))
    intrinsic = resize_intrinsics(intrinsic, original_wh, (640, 360))
    if not np.allclose(t['camera_to_origin'], camera[frame_ids], atol=1e-5) or not np.allclose(t['intrinsic'], intrinsic, atol=1e-4):
        raise ValueError('teacher/source camera mismatch')
    a.output.mkdir(parents=True)
    started = time.time()
    print(json.dumps(dict(stage='PREFIX_DEPTH', observation_frame_ids=prefix_ids.tolist(), latest_allowed_frame=tick)), flush=True)
    depth = depth_from_omega(rgb, a.depth_vendor, a.depth_checkpoint, a.output)
    calibration = recover_metric_scale(depth, hand[prefix_ids], hv[prefix_ids], intrinsic, camera[prefix_ids])
    depth *= calibration['scale']
    # Use original query grid, NOT offline tracked current pixel estimates.
    yy, xx = np.mgrid[0:360:2, 0:640:2]
    queries = np.stack((xx.ravel(), yy.ravel()), -1).astype('float32')[None]
    if t['tracks_uv'].shape[1] != queries.shape[1]:
        raise ValueError('teacher query identity is not the documented every-second-pixel grid')
    observed, observed_valid = lift_tracks(queries, np.ones(queries.shape[:-1], bool), depth[-1:], intrinsic, camera[tick:tick+1])
    C = np.linalg.inv(camera[tick])
    def anchor(x):
        return (x @ C[:3, :3].T + C[:3, 3]).astype('float32')
    observed = anchor(observed)
    observed_valid &= (np.abs(observed) <= 1.99).all(-1)
    chosen = select_scene(observed, observed_valid, a.max_points)
    teacher_world, teacher_valid = lift_tracks(t['tracks_uv'][:, chosen], t['visibility'][:, chosen],
        t['depth_m'], intrinsic, t['camera_to_origin'])
    future_world, future_valid = resample_valid(teacher_world, teacher_valid, 31)
    actual = anchor(future_world)[1:25]
    target_valid = future_valid[1:25] & future_valid[0][None]
    query_xyz = observed[0, chosen]
    # Report teacher/causal depth disagreement; do not conceal it by rebasing labels.
    initial_error = np.linalg.norm(anchor(teacher_world)[0] - query_xyz, axis=-1)
    finite_initial = teacher_valid[0] & np.isfinite(initial_error)
    if not target_valid.any() or not finite_initial.any():
        raise ValueError('no teacher correspondence')
    sample = dict(query_xyz=query_xyz, query_valid=observed_valid[0, chosen], track_ids=chosen.astype('int64'),
                  point_target=actual, point_target_valid=target_valid,
                  hand=anchor(hand[tick-3:tick+25]), hand_valid=hv[tick-3:tick+25],
                  start_object_points=np.zeros((32, 3), 'float32'), goal_object_points=np.zeros((32, 3), 'float32'),
                  object_goal_valid=np.zeros(32, bool))
    if a.goal_mask:
        import cv2
        mask = cv2.imread(str(a.goal_mask), cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != (360, 640):
            raise ValueError('current target mask must be640x360')
        pixel = queries[0, chosen].astype(int)
        ids = np.flatnonzero(mask[pixel[:, 1], pixel[:, 0]] > 0)
        # Goal teacher availability masks labels; target selection is current-mask only.
        ids = ids[np.linspace(0, len(ids)-1, min(32, len(ids))).astype(int)] if len(ids) else ids
        n = len(ids)
        sample['start_object_points'][:n] = query_xyz[ids]
        sample['goal_object_points'][:n] = actual[-1, ids]
        sample['object_goal_valid'][:n] = target_valid[-1, ids]
    path = a.output/'clip.npz'
    np.savez_compressed(path, **sample)
    record = dict(path=path.name, sha256=sha256(path), split='train', session=session,
                  source_split='official_train', source_identity=identity, teacher_sha256=teacher_sha,
                  raw_hdf5_sha256=sha256(a.hdf5), teacher_path=str(a.teacher.resolve()), observation_frame_ids=prefix_ids.tolist(),
                  teacher_frame_ids=frame_ids.tolist(), source_current_frame=tick)
    report = dict(schema='goal-point-wam.video-points.v1', status='COMPLETED', training_allowed=False,
        engineering_only=True, causal_observation=True, fps=30, history=4, horizon=24, records=[record],
        current_frame='native current camera, metric', native_hand_keypoints=JOINTS,
        hand_order=['right', 'left'], world_labels='offline teacher 10Hz, interpolated30Hz; NOT native30Hz measurements',
        current_scene_features='current-only depth; scene normal/history fields unavailable and zeroed; native hand history retained',
        goal_available=bool(sample['object_goal_valid'].any()), goal_mask_sha256=sha256(a.goal_mask) if a.goal_mask else None,
        calibration=calibration, depth_vendor_sha256=vendor_sha, depth_checkpoint_sha256=sha256(a.depth_checkpoint),
        code_sha256=sha256(Path(__file__)), elapsed_seconds=time.time()-started,
        teacher_current_disagreement_mm=dict(median=float(np.median(initial_error[finite_initial])*1000),
                                             p90=float(np.percentile(initial_error[finite_initial], 90)*1000)),
        admission_limitation='Engineering only: inspect depth/track drift and overlay; require separate sessions for scientific probe.')
    (a.output/'manifest.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
