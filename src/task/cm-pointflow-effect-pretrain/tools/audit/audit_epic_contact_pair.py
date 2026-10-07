#!/usr/bin/env python3
"""Audit and convert one EPIC-Contact clip into a non-training candidate pack.

EPIC-Contact already contains the hand and object in the same per-frame camera
coordinates.  This tool keeps that pair together and records the source clock
and any interpolation explicitly.  It deliberately does not claim that the
camera coordinates are a stationary world frame, so the resulting pack is
marked ``training_allowed=false`` until that contract is separately settled.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


SEMANTIC_11 = [0, 1, 5, 9, 13, 17, 4, 8, 12, 16, 20]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def rigid_valid(poses):
    poses = np.asarray(poses)
    R = poses[..., :3, :3]
    return (np.isfinite(poses).all((-1, -2))
            & (np.abs(R.swapaxes(-1, -2) @ R - np.eye(3)).max((-1, -2)) < 1e-3)
            & (np.abs(np.linalg.det(R) - 1) < 1e-3)
            & (np.abs(poses[..., 3, :] - [0, 0, 0, 1]).max(-1) < 1e-6))


def sample_surface(vertices, faces, count=512, seed=0):
    tri = vertices[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = np.linalg.norm(cross, axis=1) * .5
    if not np.isfinite(area).all() or area.sum() <= 0:
        raise ValueError('EPIC-Contact object mesh has no finite surface area')
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(area), count, p=area / area.sum())
    u = np.sqrt(rng.random(count))
    v = rng.random(count)
    bary = np.stack((1 - u, u * (1 - v), u * v), axis=-1)
    points = (tri[selected] * bary[:, :, None]).sum(1)
    normals = cross[selected] / (2 * area[selected, None])
    center = vertices.mean(0)
    radius = float(np.sqrt(np.mean(np.sum((points - center) ** 2, axis=-1))))
    return points.astype('float32'), normals.astype('float32'), center.astype('float32'), radius


def quality_rows(path, clip):
    rows = []
    with Path(path).open(newline='') as f:
        for row in csv.DictReader(f):
            if row['clip'] == clip:
                rows.append(row)
    if not rows:
        raise ValueError('quality CSV contains no rows for ' + clip)
    return rows


def interpolate_grid(source_frames, hand, poses, grid):
    """Interpolate hand linearly and object pose with SO(3) slerp."""
    source_frames = np.asarray(source_frames, dtype=float)
    grid = np.asarray(grid, dtype=float)
    if grid[0] < source_frames[0] or grid[-1] > source_frames[-1]:
        raise ValueError('nominal grid falls outside source frame range')
    hand_grid = np.empty((len(grid),) + hand.shape[1:], dtype='float32')
    for side in range(hand.shape[1]):
        for joint in range(hand.shape[2]):
            for axis in range(3):
                hand_grid[:, side, joint, axis] = np.interp(
                    grid, source_frames, hand[:, side, joint, axis])
    sl = Slerp(source_frames, Rotation.from_matrix(poses[:, :3, :3]))
    out = np.broadcast_to(np.eye(4), (len(grid), 4, 4)).copy()
    out[:, :3, :3] = sl(grid).as_matrix()
    for axis in range(3):
        out[:, axis, 3] = np.interp(grid, source_frames, poses[:, axis, 3])
    return hand_grid, out.astype('float32')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--npz', type=Path, required=True)
    p.add_argument('--quality-csv', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--objectforesight-poses', type=Path)
    p.add_argument('--objectforesight-spatracker', type=Path)
    p.add_argument('--objectforesight-start-frame', type=int)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    with np.load(args.npz, allow_pickle=True) as z:
        clip = str(z['_clip'].item())
        frames = np.asarray(z['_frame_num'], dtype=np.int64)
        hand_raw = np.asarray(z['mano.j3d.cam.r'], dtype='float32')
        hand_valid = np.asarray(z['right_valid'], dtype='float32') > .5
        object_vertices = np.asarray(z['object.v.cam'], dtype='float32')
        object_faces = np.asarray(z['object.f'][0], dtype=np.int64)
        rotations = np.asarray(z['object.rot'], dtype='float32')
        translations = np.asarray(z['object.cam_t'], dtype='float32')
        diameter = float(np.asarray(z['object.diameter']).reshape(-1)[0])
        verified = bool(np.asarray(z['clip_verified']).item())

    if len(frames) != len(hand_raw) or len(frames) < 2:
        raise ValueError('inconsistent or too-short EPIC-Contact clip')
    if not np.all(np.diff(frames) > 0):
        raise ValueError('EPIC-Contact frame numbers are not strictly increasing')
    poses = np.broadcast_to(np.eye(4), (len(frames), 4, 4)).copy().astype('float32')
    poses[:, :3, :3] = rotations
    poses[:, :3, 3] = translations
    hand = np.zeros((len(frames), 2, 11, 3), dtype='float32')
    hand[:, 0] = hand_raw[:, SEMANTIC_11]
    hand_valid = hand_valid & np.isfinite(hand).all((1, 2, 3))

    canonical = (object_vertices - translations[:, None, :]) @ rotations
    canonical_reference = canonical[0]
    canonical_rms = np.sqrt(np.mean((canonical - canonical_reference[None]) ** 2, axis=(1, 2)))
    points, normals, center, radius = sample_surface(canonical_reference, object_faces,
                                                      seed=int(hashlib.sha256(clip.encode()).hexdigest()[:8], 16))
    rows = quality_rows(args.quality_csv, clip)
    quality_frames = np.asarray([int(r['frame_num']) for r in rows])
    if not np.array_equal(quality_frames, frames):
        raise ValueError('quality rows and NPZ frame clock disagree')
    high = np.asarray([int(r['high_confidence']) for r in rows], dtype=bool)
    quality_hand_valid = np.asarray([int(r['hand_valid_ok']) for r in rows], dtype=bool)
    hand_valid &= quality_hand_valid

    coordinate_frame = 'per-frame_camera_coordinates'
    camera_extrinsics = None
    if args.objectforesight_spatracker:
        if args.objectforesight_start_frame is None:
            raise ValueError('--objectforesight-start-frame is required with SpaTracker extrinsics')
        with np.load(args.objectforesight_spatracker) as z:
            extrinsics = np.asarray(z['extrinsics'], dtype='float32')
        local = frames - args.objectforesight_start_frame
        if np.any(local < 0) or np.any(local >= len(extrinsics)):
            raise ValueError('Contact rows fall outside SpaTracker extrinsics')
        camera_extrinsics = extrinsics[local]
        # The trajectory metadata declares c2w.  Apply the same camera-to-world
        # transform to hands and object poses before any temporal resampling.
        poses = np.einsum('tij,tjk->tik', camera_extrinsics, poses)
        hand = np.einsum('tij,tnkj->tnki', camera_extrinsics[:, :3, :3], hand)
        hand += camera_extrinsics[:, None, None, :3, 3]
        coordinate_frame = 'objectforesight_spatracker_world_c2w'
    hand[~hand_valid, 0] = 0
    pose_valid = rigid_valid(poses)

    # The source is ~59.94 Hz and Contact rows target ~30 Hz.  Resample in
    # time, rather than assuming every pair of source frame IDs is present.
    source_fps = 59.94005994005994
    source_times = (frames - frames[0]).astype(float) / source_fps
    target_times = np.arange(int(np.floor(source_times[-1] * 30 + 1e-8)) + 1, dtype=float) / 30
    target_nearest = frames[np.abs(source_times[:, None] - target_times[None, :]).argmin(axis=0)]
    right = np.searchsorted(source_times, target_times, side='right').clip(1, len(source_times) - 1)
    left = right - 1
    gap_target = (frames[right] - frames[left]) > 2
    interpolated_hand, interpolated_poses = interpolate_grid(
        source_times, hand[:, :1], poses, target_times)
    interpolated_hand = interpolated_hand[:, 0]
    candidate_hand_valid = np.zeros((len(target_times), 2), dtype=bool)
    candidate_hand_valid[:, 0] = True
    observed_nominal = np.ones(len(target_times), dtype=bool)
    runs = []
    for value in np.split(frames, np.flatnonzero(np.diff(frames) > 2) + 1):
        if len(value):
            runs.append(len(value))
    longest_exact_run = max(runs, default=0)
    out_npz = args.output / 'contact_candidate.npz'
    np.savez_compressed(
        out_npz, hand=interpolated_hand[:, None], hand_valid=candidate_hand_valid,
        poses=interpolated_poses[:, None], pose_valid=np.ones((len(target_times), 1), dtype=bool),
        canonical_points=points, canonical_normals=normals, canonical_center=center,
        canonical_radius=np.asarray(radius, dtype='float32'), source_frame_ids=frames,
        nominal_frame_ids=target_nearest, nominal_timestamps=target_times,
        interpolated_frame_mask=gap_target,
        source_hand_valid=hand_valid, source_high_confidence=high,
        source_canonical_rms=canonical_rms.astype('float32'))

    audit = dict(
        status='CANDIDATE_ONLY', training_allowed=False,
        source='EPIC-Contact', clip=clip, source_npz_sha256=sha(args.npz),
        quality_csv_sha256=sha(args.quality_csv), clip_verified=verified,
        coordinate_frame=coordinate_frame,
        coordinate_frame_training_blocker='candidate pack remains non-training until camera/clock contract is validated',
        hand_semantics=SEMANTIC_11, annotated_side='right', left_hand_present=False,
        source_frames=len(frames), source_frame_range=[int(frames[0]), int(frames[-1])],
        source_frame_diffs=sorted(set(np.diff(frames).tolist())),
        nominal_30hz_frames=len(target_times), nominal_frame_range=[int(target_nearest[0]), int(target_nearest[-1])],
        interpolation_count=int(gap_target.sum()),
        longest_exact_nominal_run=int(longest_exact_run),
        exact_4_plus_24_windows=max(0, longest_exact_run - 27),
        interpolated_4_plus_24_windows=max(0, len(target_times) - 27),
        all_source_hand_valid=bool(hand_valid.all()), all_source_pose_valid=bool(pose_valid.all()),
        high_confidence_fraction=float(high.mean()), object_diameter_m=diameter,
        canonical_mesh_extent=np.ptp(canonical_reference, axis=0).astype(float).tolist(),
        canonical_rms_max_m=float(canonical_rms.max()), canonical_rms_median_m=float(np.median(canonical_rms)),
        candidate_npz_sha256=sha(out_npz),
        objectforesight=None,
    )
    if camera_extrinsics is not None:
        audit['camera_extrinsics'] = dict(
            source=str(args.objectforesight_spatracker), sha256=sha(args.objectforesight_spatracker),
            convention='c2w', applied=True, translation_extent_m=np.ptp(camera_extrinsics[:, :3, 3], axis=0).tolist(),
        )
    if args.objectforesight_poses:
        if args.objectforesight_start_frame is None:
            raise ValueError('--objectforesight-start-frame is required with OF poses')
        with np.load(args.objectforesight_poses) as z:
            of_ids = np.asarray(z['frame_ids'], dtype=np.int64)
            of_init = int(np.asarray(z['init_from_frame']).item())
        local = frames - args.objectforesight_start_frame
        in_range = (local >= 0) & (local < len(of_ids))
        audit['objectforesight'] = dict(
            poses_sha256=sha(args.objectforesight_poses), frame_count=len(of_ids),
            init_from_frame=of_init, contact_local_frames=local.tolist(),
            all_contact_frames_in_pose_range=bool(in_range.all()),
            exact_pose_frame_match=bool(np.array_equal(of_ids[local[in_range]], local[in_range])) if in_range.all() else False,
            used_for_training=False,
            reason='independent overlap audit only; raw OF GLB/pose scale is not injected',
        )
    (args.output / 'audit.json').write_text(json.dumps(audit, indent=2, allow_nan=False) + '\n')
    print(json.dumps(audit, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
