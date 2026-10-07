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
SOURCE_FPS = 59.94005994005994


def camera_to_world(extrinsics, convention):
    """Respect the trajectory's recorded convention; never infer it from poses."""
    extrinsics = np.asarray(extrinsics, dtype='float64')
    if convention not in ('c2w', 'w2c'):
        raise ValueError('unknown ObjectForesight extrinsics_conv: ' + str(convention))
    if not rigid_valid(extrinsics).all():
        raise ValueError('non-rigid ObjectForesight camera extrinsics')
    return (np.linalg.inv(extrinsics) if convention == 'w2c' else extrinsics).astype('float32')


def load_convention(metadata, clip):
    """Read official JSON or trajectory metadata; missing keys are blockers."""
    if metadata is None or not clip:
        raise ValueError('camera conversion requires convention metadata and ObjectForesight clip ID')
    metadata = Path(metadata)
    if metadata.suffix == '.parquet':
        import pyarrow.parquet as pq
        rows = pq.read_table(metadata, columns=['clip', 'extrinsics_conv']).to_pylist()
        values = {row['extrinsics_conv'] for row in rows if row['clip'] == clip}
        if len(values) != 1:
            raise ValueError('missing/conflicting extrinsics_conv for ' + clip)
        value = next(iter(values))
    else:
        conventions = json.loads(metadata.read_text())
        value = conventions.get(clip)
    if value not in ('c2w', 'w2c'):
        raise ValueError('missing/unknown extrinsics_conv for ' + clip)
    return value


def grid_support(source_times, frames, target_times, max_frame_gap=2):
    """Exact observations use their own mask; interpolations require both ends."""
    hi = np.searchsorted(source_times, target_times, side='left').clip(0, len(frames) - 1)
    exact = np.isclose(source_times[hi], target_times, atol=1e-9, rtol=0)
    lo = np.maximum(hi - 1, 0)
    lo[exact] = hi[exact]
    gap = (~exact) & ((frames[hi] - frames[lo]) > max_frame_gap)
    return lo, hi, exact, gap


def supported_mask(valid, lo, hi, gap):
    shape = (len(gap),) + (1,) * (valid.ndim - 1)
    return valid[lo] & valid[hi] & ~gap.reshape(shape)


def count_windows(valid, length=28):
    return sum(bool(valid[t:t + length].all()) for t in range(max(0, len(valid) - length + 1)))


def metric_summary(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if not len(values):
        return dict(count=0, median=None, p95=None, max=None, rms=None)
    return dict(count=int(len(values)), median=float(np.median(values)),
                p95=float(np.quantile(values, .95)), max=float(values.max()),
                rms=float(np.sqrt(np.mean(values ** 2))))


def relative_geometry(hand, poses, points, hand_valid, pose_valid, frames):
    """Audit object-relative landmarks and sampled-surface gaps, not GT contact."""
    relative = np.einsum('tij,tnkj->tnki', poses[:, :3, :3].transpose(0, 2, 1),
                         hand - poses[:, None, None, :3, 3])
    # Bounded: only the 22 semantic landmarks and 512 sampled surface points.
    distances = np.linalg.norm(relative[..., None, :] - points[None, None, None], axis=-1).min(-1)
    result = dict(distance_semantics='unsigned semantic-point to sampled object surface; not contact GT')
    for side, name in enumerate(('right', 'left')):
        valid = hand_valid[:, side] & pose_valid
        near = np.min(distances[:, side], axis=-1)
        adjacent = valid[1:] & valid[:-1] & (np.diff(frames) <= 2)
        rms_step = np.sqrt(np.mean(np.sum(np.diff(relative[:, side], axis=0) ** 2, axis=-1), axis=-1))
        speed = rms_step / (np.diff(frames) / SOURCE_FPS)
        result[name] = dict(valid_frames=int(valid.sum()), nearest_surface_m=metric_summary(near[valid]),
                            relative_landmark_rms_speed_m_per_s=metric_summary(speed[adjacent]),
                            surface_gap_step_m=metric_summary(np.abs(np.diff(near))[adjacent]))
    return result


def compare_object_poses(contact_poses, contact_valid, frames, of_ids, of_poses, start_frame,
                         camera_extrinsics=None):
    """Independent matched camera poses and anchor-relative motion diagnostics.

    Different meshes can have different canonical origins/orientations. Raw
    discrepancy is descriptive; anchor-relative camera rotation cancels the
    constant object-frame orientation. The optional single-scale fit is an
    in-sample diagnostic, never a calibration applied to candidate data.
    """
    of_ids = np.asarray(of_ids, dtype=np.int64)
    of_poses = np.asarray(of_poses)
    if of_ids.ndim != 1 or of_poses.shape != (len(of_ids), 4, 4) or np.any(np.diff(of_ids) <= 0):
        raise ValueError('invalid ObjectForesight pose/frame_ids contract')
    local = frames - start_frame
    index = np.searchsorted(of_ids, local)
    match = index < len(of_ids)
    match[match] &= of_ids[index[match]] == local[match]
    valid = match & contact_valid
    valid[match] &= rigid_valid(of_poses[index[match]])
    c = contact_poses[valid]
    o = of_poses[index[valid]]
    result = dict(contact_local_frames=local.tolist(), matched_frame_count=int(match.sum()),
                  valid_matched_frame_count=int(valid.sum()), exact_pose_frame_match=bool(match.all()),
                  all_contact_frames_in_pose_range=bool(len(of_ids) and ((local >= of_ids[0]) & (local <= of_ids[-1])).all()),
                  used_for_training=False, coordinate_frame='per-frame camera',
                  reason='independent pose audit only; no pose, mesh or fitted scale injected')
    if camera_extrinsics is not None and match.any():
        world_contact = camera_extrinsics @ contact_poses
        world_of = camera_extrinsics[match] @ of_poses[index[match]]
        world_result = compare_object_poses(world_contact, contact_valid, frames, local[match], world_of, start_frame)
        world_result['coordinate_frame'] = 'shared ObjectForesight world candidate'
        result['world_comparison'] = world_result
    if not len(c):
        return result
    result['raw_center_discrepancy_m'] = metric_summary(np.linalg.norm(c[:, :3, 3] - o[:, :3, 3], axis=-1))
    dc = c[:, :3, 3] - c[0, :3, 3]
    do = o[:, :3, 3] - o[0, :3, 3]
    result['anchor_relative_translation_discrepancy_m'] = metric_summary(np.linalg.norm(dc - do, axis=-1))
    rc = c[:, :3, :3] @ c[0, :3, :3].T
    ro = o[:, :3, :3] @ o[0, :3, :3].T
    rotation_error = Rotation.from_matrix(rc @ ro.transpose(0, 2, 1)).magnitude()
    result['anchor_relative_rotation_discrepancy_deg'] = metric_summary(np.rad2deg(rotation_error))
    cc = c[:, :3, 3] - c[:, :3, 3].mean(0)
    oo = o[:, :3, 3] - o[:, :3, 3].mean(0)
    denom = float(np.sum(cc * cc))
    scale = float(np.sum(cc * oo) / denom) if denom > 1e-12 else None
    result['single_scale_translation_fit'] = dict(scale=scale, positive_scale=scale is not None and scale > 0,
                                                 diagnostic_only=True, applied=False)
    if scale is not None:
        result['single_scale_translation_fit']['residual_m'] = metric_summary(np.linalg.norm(scale * cc - oo, axis=-1))
    return result


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
    p.add_argument('--objectforesight-clip', help='exact clip key in extrinsics_conv.json')
    p.add_argument('--objectforesight-conventions', type=Path,
                   help='official extrinsics_conv.json or trajectories.parquet; no implicit c2w fallback')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    with np.load(args.npz, allow_pickle=True) as z:
        clip = str(z['_clip'].item())
        frames = np.asarray(z['_frame_num'], dtype=np.int64)
        hand_raw = np.stack([np.asarray(z['mano.j3d.cam.' + s], dtype='float32')
                             for s in ('r', 'l')], axis=1)
        hand_valid = np.stack([np.asarray(z[s + '_valid'], dtype='float32') > .5
                               for s in ('right', 'left')], axis=1)
        joint_valid = np.stack([np.asarray(z['joints_valid_' + s])[:, SEMANTIC_11] > .5
                                if 'joints_valid_' + s in z else np.ones((len(frames), 11), dtype=bool)
                                for s in ('r', 'l')], axis=1)
        object_vertices = np.asarray(z['object.v.cam'], dtype='float32')
        object_faces = np.asarray(z['object.f'][0], dtype=np.int64)
        rotations = np.asarray(z['object.rot'], dtype='float32')
        translations = np.asarray(z['object.cam_t'], dtype='float32')
        diameter = float(np.asarray(z['object.diameter']).reshape(-1)[0])
        verified = bool(np.asarray(z['clip_verified']).item())
        source_videos = set(np.asarray(z['video_id']).reshape(-1).astype(str)) if 'video_id' in z else set()

    if len(source_videos) > 1:
        raise ValueError('EPIC-Contact clip spans multiple source videos')
    if args.objectforesight_spatracker or args.objectforesight_poses:
        if not args.objectforesight_clip:
            raise ValueError('--objectforesight-clip is required for independent overlap audit')
        if source_videos and not args.objectforesight_clip.startswith(next(iter(source_videos)) + '_'):
            raise ValueError('EPIC-Contact and ObjectForesight do not share the source video')

    if len(frames) != len(hand_raw) or len(frames) < 2:
        raise ValueError('inconsistent or too-short EPIC-Contact clip')
    if not np.all(np.diff(frames) > 0):
        raise ValueError('EPIC-Contact frame numbers are not strictly increasing')
    poses = np.broadcast_to(np.eye(4), (len(frames), 4, 4)).copy().astype('float32')
    poses[:, :3, :3] = rotations
    poses[:, :3, 3] = translations
    if hand_raw.shape != (len(frames), 2, 21, 3) or hand_valid.shape != (len(frames), 2):
        raise ValueError('EPIC-Contact hand arrays do not satisfy the two-side contract')
    hand = hand_raw[:, :, SEMANTIC_11].copy()
    hand_valid &= np.isfinite(hand).all((2, 3)) & joint_valid.all(-1)
    source_pose_valid = rigid_valid(poses)
    if not source_pose_valid.any():
        raise ValueError('no valid EPIC-Contact object pose')

    canonical = (object_vertices - translations[:, None, :]) @ rotations
    canonical_reference = canonical[np.flatnonzero(source_pose_valid)[0]]
    canonical_rms = np.sqrt(np.mean((canonical - canonical_reference[None]) ** 2, axis=(1, 2)))
    points, normals, center, radius = sample_surface(canonical_reference, object_faces,
                                                      seed=int(hashlib.sha256(clip.encode()).hexdigest()[:8], 16))
    rows = quality_rows(args.quality_csv, clip)
    quality_frames = np.asarray([int(r['frame_num']) for r in rows])
    if not np.array_equal(quality_frames, frames):
        raise ValueError('quality rows and NPZ frame clock disagree')
    high = np.asarray([int(r['high_confidence']) for r in rows], dtype=bool)
    central = np.asarray([int(r.get('is_keyframe', '0')) for r in rows], dtype=bool)
    quality_hand_valid = np.asarray([int(r['hand_valid_ok']) for r in rows], dtype=bool)
    sides = {r['annotated_side'] for r in rows}
    if len(sides) != 1 or not sides <= {'r', 'l'}:
        raise ValueError('quality rows disagree on the annotated hand side')
    annotated_side = next(iter(sides))
    annotated_index = 0 if annotated_side == 'r' else 1
    hand_valid[:, annotated_index] &= quality_hand_valid
    hand_valid &= high[:, None]
    source_pose_valid &= high & quality_hand_valid
    camera_poses = poses.copy()

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
        convention = load_convention(args.objectforesight_conventions, args.objectforesight_clip)
        camera_extrinsics = camera_to_world(extrinsics[local], convention)
        # Convert the recorded per-clip convention before moving BOTH hands
        # and the object into the same candidate world frame.
        poses = np.einsum('tij,tjk->tik', camera_extrinsics, poses)
        hand = np.einsum('tij,tnkj->tnki', camera_extrinsics[:, :3, :3], hand)
        hand += camera_extrinsics[:, None, None, :3, 3]
        coordinate_frame = 'objectforesight_spatracker_world_c2w'
    hand[~hand_valid] = 0
    pose_valid = source_pose_valid & rigid_valid(poses)
    geometry = relative_geometry(hand, poses, points, hand_valid, pose_valid, frames)
    # Safe interpolation values cannot make an invalid row valid. These zero /
    # identity placeholders are always masked by the endpoint validity below.
    poses[~pose_valid] = np.eye(4)

    # The source is ~59.94 Hz and Contact rows target ~30 Hz.  Resample in
    # time, rather than assuming every pair of source frame IDs is present.
    source_fps = SOURCE_FPS
    source_times = (frames - frames[0]).astype(float) / source_fps
    target_times = np.arange(int(np.floor(source_times[-1] * 30 + 1e-8)) + 1, dtype=float) / 30
    target_nearest = frames[np.abs(source_times[:, None] - target_times[None, :]).argmin(axis=0)]
    left, right, exact_target, gap_target = grid_support(source_times, frames, target_times)
    interpolated_hand, interpolated_poses = interpolate_grid(
        source_times, hand, poses, target_times)
    candidate_hand_valid = supported_mask(hand_valid, left, right, gap_target)
    candidate_pose_valid = supported_mask(pose_valid, left, right, gap_target)
    interpolated_hand[~candidate_hand_valid] = 0
    interpolated_poses[~candidate_pose_valid] = np.eye(4)
    runs = []
    for value in np.split(frames, np.flatnonzero(np.diff(frames) > 2) + 1):
        if len(value):
            runs.append(len(value))
    longest_exact_run = max(runs, default=0)
    out_npz = args.output / 'contact_candidate.npz'
    np.savez_compressed(
        out_npz, hand=interpolated_hand, hand_valid=candidate_hand_valid,
        poses=interpolated_poses[:, None], pose_valid=candidate_pose_valid[:, None],
        canonical_points=points, canonical_normals=normals, canonical_center=center,
        canonical_radius=np.asarray(radius, dtype='float32'), source_frame_ids=frames,
        nominal_frame_ids=target_nearest, nominal_timestamps=target_times,
        interpolated_frame_mask=~exact_target, gap_crossing_mask=gap_target,
        source_bracket_indices=np.stack((left, right), axis=-1),
        source_hand_valid=hand_valid, source_high_confidence=high,
        source_central_frame=central,
        source_pose_valid=pose_valid,
        source_canonical_rms=canonical_rms.astype('float32'))

    audit = dict(
        status='CANDIDATE_ONLY', training_allowed=False,
        schema='ref2dex.epic-contact-candidate.v2', audit_tool_sha256=sha(__file__),
        source='EPIC-Contact', clip=clip, source_npz_sha256=sha(args.npz),
        source_video=next(iter(source_videos)) if source_videos else None,
        quality_csv_sha256=sha(args.quality_csv), clip_verified=verified,
        coordinate_frame=coordinate_frame,
        coordinate_frame_training_blocker='candidate pack remains non-training until camera/clock contract is validated',
        hand_semantics=SEMANTIC_11, hand_side_order=['right', 'left'],
        annotated_side='right' if annotated_side == 'r' else 'left',
        left_hand_present=bool(hand_valid[:, 1].any()), right_hand_present=bool(hand_valid[:, 0].any()),
        supervision_semantics='propagated fixed hand-object relative geometry with approximate absolute camera placement',
        central_frame_rows=int(central.sum()),
        source_contracts=[
            'https://huggingface.co/datasets/raivn/ObjectForesight-EPIC#camera--pose-conventions',
            'https://huggingface.co/datasets/Sid2697/epic-contact/blob/0df7796dba1acdc4d0260b69662524916e9f7079/DATASET.md'],
        source_quality_policy='both interpolation endpoints valid and high-confidence; gaps >2 source frames masked',
        relative_geometry=geometry,
        source_frames=len(frames), source_frame_range=[int(frames[0]), int(frames[-1])],
        source_fps=SOURCE_FPS, target_fps=30,
        source_frame_diffs=sorted(set(np.diff(frames).tolist())),
        nominal_30hz_frames=len(target_times), nominal_frame_range=[int(target_nearest[0]), int(target_nearest[-1])],
        interpolation_count=int((~exact_target).sum()), gap_crossing_count=int(gap_target.sum()),
        candidate_valid_hand_frames=candidate_hand_valid.sum(0).tolist(),
        candidate_valid_pose_frames=int(candidate_pose_valid.sum()),
        longest_exact_nominal_run=int(longest_exact_run),
        exact_4_plus_24_windows=count_windows(candidate_pose_valid & candidate_hand_valid[:, annotated_index] & exact_target),
        interpolated_4_plus_24_windows=count_windows(candidate_pose_valid & candidate_hand_valid[:, annotated_index]),
        unmasked_candidate_4_plus_24_windows=max(0, len(target_times) - 27),
        all_source_hand_valid=bool(hand_valid.all()), all_source_pose_valid=bool(pose_valid.all()),
        high_confidence_fraction=float(high.mean()), object_diameter_m=diameter,
        canonical_mesh_extent=np.ptp(canonical_reference, axis=0).astype(float).tolist(),
        canonical_rms_max_m=float(canonical_rms[np.isfinite(canonical_rms)].max()),
        canonical_rms_median_m=float(np.median(canonical_rms[np.isfinite(canonical_rms)])),
        candidate_npz_sha256=sha(out_npz),
        objectforesight=None,
    )
    if camera_extrinsics is not None:
        audit['camera_extrinsics'] = dict(
            source=str(args.objectforesight_spatracker), sha256=sha(args.objectforesight_spatracker),
            source_convention=convention, convention='c2w', applied=True,
            convention_metadata=str(args.objectforesight_conventions),
            convention_metadata_sha256=sha(args.objectforesight_conventions),
            convention_clip=args.objectforesight_clip,
            translation_extent_m=np.ptp(camera_extrinsics[:, :3, 3], axis=0).tolist(),
        )
    if args.objectforesight_poses:
        if args.objectforesight_start_frame is None:
            raise ValueError('--objectforesight-start-frame is required with OF poses')
        with np.load(args.objectforesight_poses) as z:
            of_ids = np.asarray(z['frame_ids'], dtype=np.int64)
            of_init = int(np.asarray(z['init_from_frame']).item())
            of_poses = np.asarray(z['T_c_o'], dtype='float32')
        audit['objectforesight'] = compare_object_poses(camera_poses, source_pose_valid,
            frames, of_ids, of_poses, args.objectforesight_start_frame, camera_extrinsics)
        audit['objectforesight'].update(poses_sha256=sha(args.objectforesight_poses),
                                      frame_count=len(of_ids), init_from_frame=of_init)
    (args.output / 'audit.json').write_text(json.dumps(audit, indent=2, allow_nan=False) + '\n')
    print(json.dumps(audit, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
