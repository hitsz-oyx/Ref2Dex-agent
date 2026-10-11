#!/usr/bin/env python3
"""Build a bounded EPIC scene-flow candidate from one ObjectForesight clip.

This converter intentionally does not read FoundationPose poses.  Static scene
points come from metric depth plus the SpaTracker camera trajectory; the moved
object points come from RGB Lucas-Kanade tracks and depth.  EPIC-Contact only
supplies the hand points.  The output is an auditable candidate pack, not a
training manifest.
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from audit_epic_contact_pair import SEMANTIC_11, SOURCE_FPS, grid_support, sha  # noqa: E402


SCHEMA = 'ref2dex.epic-scene-flow.v2'
TARGET_FPS = 30.0
WINDOW = 28


def source_clock(meta, n_frames, requested_start=None):
    """Metadata start already includes padding; stop is exclusive upstream."""
    start, stop = int(meta['start_frame']), int(meta['stop_frame'])
    fps = float(meta['fps'])
    if not np.isfinite(fps) or not 0 < fps < 240 or stop - start != n_frames:
        raise ValueError('metadata clock does not match the decoded frame count')
    if requested_start is not None and requested_start != start:
        raise ValueError('requested start frame disagrees with action.meta.json')
    return start, fps


def contact_quality(path, clip, frames):
    """Missing, duplicate or unverified quality rows never qualify a hand."""
    result = np.zeros((len(frames), 2), dtype=bool)
    rows = {}
    with Path(path).open() as f:
        for row in csv.DictReader(f):
            if row['clip'] != clip:
                continue
            key = (int(row['frame_num']), row['annotated_side'])
            if key in rows:
                raise ValueError('duplicate Contact quality row')
            rows[key] = row
    truth = lambda value: str(value).lower() in ('1', 'true')
    for i, frame in enumerate(frames):
        for s, side in enumerate(('r', 'l')):
            row = rows.get((int(frame), side), {})
            result[i, s] = all(truth(row.get(k, '')) for k in
                               ('hand_valid_ok', 'high_confidence', 'clip_verified'))
    return result


def source_splits(scene, quality_csv, train_file, val_file):
    """Retain both authorities; evaluation rows cannot become training rows."""
    membership = []
    key = scene + '/objects/0+object_0'
    for name, path in (('train', train_file), ('val', val_file)):
        if key in Path(path).read_text().splitlines():
            membership.append(name)
    if len(membership) != 1:
        raise ValueError('ObjectForesight object has missing or conflicting split')
    contact = {'epic_contact_train_frame_quality.csv': 'train',
               'epic_contact_test_frame_quality.csv': 'test'}.get(Path(quality_csv).name)
    if contact is None:
        raise ValueError('unrecognized Contact split authority')
    return dict(objectforesight=membership[0], epic_contact=contact,
                train_eligible=membership[0] == contact == 'train',
                authorities_sha256={str(Path(p).resolve()): sha(p) for p in
                                    (quality_csv, train_file, val_file)})


def rigid_valid(poses):
    poses = np.asarray(poses)
    R = poses[..., :3, :3]
    return (np.isfinite(poses).all((-1, -2))
            & (np.abs(R.swapaxes(-1, -2) @ R - np.eye(3)).max((-1, -2)) < 1e-3)
            & (np.abs(np.linalg.det(R) - 1) < 1e-3)
            & (np.abs(poses[..., 3, :] - [0, 0, 0, 1]).max(-1) < 1e-6))


def read_video(path, size):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), size,
                                 interpolation=cv2.INTER_AREA))
    cap.release()
    if not frames:
        raise ValueError('action.mp4 has no decodable frames')
    return np.stack(frames)


def resize_mask(mask, shape):
    h, w = shape
    return cv2.resize(np.asarray(mask, dtype='uint8'), (w, h),
                      interpolation=cv2.INTER_NEAREST).astype(bool)


def load_object_masks(path, shape, n_frames):
    z = np.load(path)
    if not all(str(i) in z for i in range(n_frames)):
        raise ValueError('object mask archive is missing video frames')
    return np.stack([resize_mask(z[str(i)], shape) for i in range(n_frames)])


def load_hand_masks(path, shape, n_frames):
    with np.load(path) as z:
        if 'arr' not in z or z['arr'].shape[0] != n_frames:
            raise ValueError('twohands mask archive does not match video length')
        return np.stack([resize_mask(x, shape) for x in z['arr']])


def backproject(pixels, depth, K):
    pixels = np.asarray(pixels, dtype='float64')
    depth = np.asarray(depth, dtype='float64')
    return np.stack(((pixels[..., 0] - K[0, 2]) * depth / K[0, 0],
                     (pixels[..., 1] - K[1, 2]) * depth / K[1, 1], depth), axis=-1)


def camera_to_world(points, w2c):
    c2w = np.linalg.inv(w2c)
    return np.einsum('ij,...j->...i', c2w[:3, :3], points) + c2w[:3, 3]


def world_to_camera(points, w2c):
    return np.einsum('ij,...j->...i', w2c[:3, :3], points) + w2c[:3, 3]


def static_tracks(depths, Ks, w2cs, object_masks, hand_masks, target_frames,
                  spacing=3, max_points=8192):
    """Backproject frame-zero background and validate it against every depth."""
    h, w = depths.shape[1:]
    ys, xs = np.mgrid[2:h - 2:spacing, 2:w - 2:spacing]
    keep = ((depths[0, ys, xs] > .05) & ~object_masks[0][ys, xs]
            & ~hand_masks[0][ys, xs])
    pixels0 = np.stack((xs[keep], ys[keep]), axis=-1).astype('float64')
    depth0 = depths[0, ys, xs][keep]
    world0 = camera_to_world(backproject(pixels0, depth0, Ks[0]), w2cs[0])
    points = np.zeros((len(target_frames), len(world0), 3), dtype='float32')
    valid = np.zeros((len(target_frames), len(world0)), dtype=bool)
    residuals = []
    for j, frame in enumerate(target_frames):
        camera = world_to_camera(world0, w2cs[frame])
        q = camera @ Ks[frame].T
        uv = q[:, :2] / np.maximum(q[:, 2:,], 1e-8)
        ui = np.rint(uv[:, 0]).astype(np.int64)
        vi = np.rint(uv[:, 1]).astype(np.int64)
        good = ((camera[:, 2] > .05) & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h))
        sampled = np.zeros(len(world0), dtype='float32')
        sampled[good] = depths[frame, vi[good], ui[good]]
        residual = np.abs(sampled - camera[:, 2])
        good &= sampled > .05
        good &= residual <= np.maximum(.03, .05 * camera[:, 2])
        visible = np.flatnonzero(good)
        good[visible] &= (~object_masks[frame, vi[visible], ui[visible]]
                          & ~hand_masks[frame, vi[visible], ui[visible]])
        valid[j] = good
        residuals.extend(residual[good].tolist())
        if good.any():
            observed = backproject(np.stack((ui[good], vi[good]), axis=-1),
                                   sampled[good], Ks[frame])
            points[j, good] = camera_to_world(observed, w2cs[frame]).astype('float32')
    fraction = valid.mean(0) if len(valid) else np.zeros(len(world0))
    eligible = np.flatnonzero(fraction >= .8)
    if len(eligible) > max_points:
        # Deterministic spread across the original image grid.
        eligible = eligible[np.linspace(0, len(eligible) - 1, max_points, dtype=int)]
    points, valid = points[:, eligible], valid[:, eligible]
    return points, valid, dict(
        candidate_points=int(len(world0)), selected_points=int(len(eligible)),
        valid_fraction_p50=float(np.median(fraction)) if len(fraction) else 0.,
        valid_fraction_ge_08=int((fraction >= .8).sum()),
        residual_median=float(np.median(residuals)) if residuals else None,
        residual_p95=float(np.quantile(residuals, .95)) if residuals else None,
        residual_count=len(residuals))


def track_object_rgb(gray, object_masks, target_frames, max_corners=512,
                     forward_backward_threshold_px=None):
    """Track object-mask corners through RGB; invalid tracks remain masked."""
    p = cv2.goodFeaturesToTrack(gray[0], mask=object_masks[0].astype('uint8'),
                                maxCorners=max_corners, qualityLevel=.001,
                                minDistance=1, blockSize=3)
    if p is None or len(p) < 16:
        raise ValueError('object RGB track produced fewer than 16 corners')
    positions = np.full((len(gray), len(p), 2), np.nan, dtype='float32')
    positions[0] = p.reshape(-1, 2)
    previous = gray[0]
    current = positions[0].copy()
    active = np.ones(len(p), dtype=bool)
    for frame in range(1, len(gray)):
        indices = np.flatnonzero(active)
        if not len(indices):
            break
        q, status, _ = cv2.calcOpticalFlowPyrLK(
            previous, gray[frame], current[indices].reshape(-1, 1, 2), None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, .03))
        if q is None or status is None:
            break
        q = q.reshape(-1, 2)
        good = ((status.reshape(-1) > 0) & np.isfinite(q).all(1)
                & (q[:, 0] >= 0) & (q[:, 0] < gray.shape[2])
                & (q[:, 1] >= 0) & (q[:, 1] < gray.shape[1]))
        ui = np.rint(np.nan_to_num(q[:, 0], nan=-1)).astype(int).clip(0, gray.shape[2] - 1)
        vi = np.rint(np.nan_to_num(q[:, 1], nan=-1)).astype(int).clip(0, gray.shape[1] - 1)
        good &= object_masks[frame, vi, ui]
        if forward_backward_threshold_px is not None:
            back, back_status, _ = cv2.calcOpticalFlowPyrLK(
                gray[frame], previous, np.nan_to_num(q).reshape(-1, 1, 2), None,
                winSize=(21, 21), maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, .03))
            if back is None or back_status is None:
                break
            back = back.reshape(-1, 2)
            good &= ((back_status.reshape(-1) > 0) & np.isfinite(back).all(1)
                     & (np.linalg.norm(back - current[indices], axis=1)
                        <= forward_backward_threshold_px))
        active[indices[~good]] = False
        positions[frame, indices[good]] = q[good]
        current[indices[good]] = q[good]
        previous = gray[frame]
    target = positions[target_frames]
    return target


def object_tracks(depths, Ks, w2cs, object_masks, gray, target_frames, max_points=512):
    pixels = track_object_rgb(gray, object_masks, target_frames, max_corners=max_points)
    t, n = pixels.shape[:2]
    points = np.zeros((t, n, 3), dtype='float32')
    valid = np.isfinite(pixels).all(2)
    for j, frame in enumerate(target_frames):
        ui = np.rint(np.nan_to_num(pixels[j, :, 0], nan=-1)).astype(int).clip(0, depths.shape[2] - 1)
        vi = np.rint(np.nan_to_num(pixels[j, :, 1], nan=-1)).astype(int).clip(0, depths.shape[1] - 1)
        depth = depths[frame, vi, ui]
        valid[j] &= depth > .05
        if valid[j].any():
            points[j, valid[j]] = camera_to_world(
                backproject(pixels[j, valid[j]], depth[valid[j]], Ks[frame]), w2cs[frame]).astype('float32')
    fraction = valid.mean(0)
    eligible = np.flatnonzero(fraction >= .8)
    if len(eligible) > max_points:
        eligible = eligible[np.linspace(0, len(eligible) - 1, max_points, dtype=int)]
    return points[:, eligible], valid[:, eligible], dict(
        initial_corners=int(n), selected_points=int(len(eligible)),
        valid_fraction_ge_08=int((fraction >= .8).sum()))


def _kabsch_similarity(source, target):
    """Return row-vector similarity source -> target for paired 3D points."""
    ca, cb = source.mean(0), target.mean(0)
    ac, bc = source - ca, target - cb
    u, _, vh = np.linalg.svd(ac.T @ bc)
    r = u @ vh
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vh
    scale = float(np.sum((ac @ r) * bc) / max(np.sum(ac * ac), 1e-12))
    t = cb - scale * (ca @ r)
    return scale, r, t


def fit_similarity(source, target, max_source=700, max_target=2400, iterations=8,
                   initial=None, fixed_scale=None):
    """Fit a bounded unordered-point-cloud similarity transform with ICP.

    The Contact mesh and the depth mask have no vertex correspondence.  PCA
    gives deterministic initial orientations; nearest-neighbour ICP then
    refines each candidate.  The returned transform maps source rows to target
    rows as ``scale * source @ rotation + translation``.
    """
    source = np.asarray(source, dtype='float64')
    target = np.asarray(target, dtype='float64')
    source = source[np.isfinite(source).all(1)]
    target = target[np.isfinite(target).all(1)]
    if len(source) < 32 or len(target) < 32:
        return None
    if len(source) > max_source:
        source = source[np.linspace(0, len(source) - 1, max_source, dtype=int)]
    if len(target) > max_target:
        target = target[np.linspace(0, len(target) - 1, max_target, dtype=int)]
    ca, cb = source.mean(0), target.mean(0)
    sa = np.linalg.eigvalsh(np.cov(source, rowvar=False))
    sb = np.linalg.eigvalsh(np.cov(target, rowvar=False))
    scale0 = float(np.sqrt(max(sb.sum(), 1e-12) / max(sa.sum(), 1e-12)))
    # eigh returns columns from the smallest to largest principal direction.
    _, ea = np.linalg.eigh(np.cov(source, rowvar=False))
    _, eb = np.linalg.eigh(np.cov(target, rowvar=False))
    tree = cKDTree(target)
    best = None
    # A previous frame's world transform is a strong initialization when the
    # target cloud is in a stationary-world coordinate system.  It is added to
    # the PCA candidates below rather than trusted without a residual check.
    if initial is not None:
        x = initial['scale'] * source @ initial['rotation'] + initial['translation']
        d, _ = tree.query(x, k=1)
        best = (float(np.sqrt(np.mean(np.minimum(d, .08) ** 2))),
                float(initial['scale']), np.asarray(initial['rotation']),
                np.asarray(initial['translation']))
    if initial is None:
        # Enumerate axis permutations and sign choices.  Only proper rotations
        # are retained; this also handles the plate's unresolved front/back axes.
        import itertools
        for perm in itertools.permutations(range(3)):
            p = np.eye(3)[:, perm]
            for signs in itertools.product((-1.0, 1.0), repeat=3):
                q = p @ np.diag(signs)
                if np.linalg.det(q) < 0:
                    continue
                r0 = ea @ q @ eb.T
                x = scale0 * (source - ca) @ r0 + cb
                d, _ = tree.query(x, k=1)
                score = float(np.sqrt(np.mean(np.minimum(d, .08) ** 2)))
                if best is None or score < best[0]:
                    best = (score, scale0, r0, cb - scale0 * ca @ r0)
    if best is None:
        return None
    _, scale, rotation, translation = best
    for _ in range(iterations):
        transformed = scale * source @ rotation + translation
        distances, indices = tree.query(transformed, k=1)
        # Reject gross outliers before the paired Kabsch update.  A generous
        # threshold keeps partial hand occlusion from collapsing the fit.
        keep = distances <= max(.025, float(np.quantile(distances, .85)) * 2.0)
        if keep.sum() < 32:
            break
        new_scale, new_rotation, new_translation = _kabsch_similarity(
            source[keep], target[indices[keep]])
        # Partial depth masks can produce a degenerate correspondence in which
        # Kabsch shrinks the thin plate toward one visible edge.  Preserve the
        # previous metric scale in that case while still accepting its rotation
        # and centroid update.
        if fixed_scale is not None:
            new_scale = float(fixed_scale)
            new_translation = target[indices[keep]].mean(0) - new_scale * (
                source[keep].mean(0) @ new_rotation)
        elif not .5 <= new_scale <= 1.5:
            new_scale = scale
            new_translation = target[indices[keep]].mean(0) - new_scale * (
                source[keep].mean(0) @ new_rotation)
        scale, rotation, translation = new_scale, new_rotation, new_translation
    transformed = scale * source @ rotation + translation
    distances, _ = tree.query(transformed, k=1)
    return dict(scale=float(scale), rotation=rotation.astype('float32'),
                translation=translation.astype('float32'),
                rms=float(np.sqrt(np.mean(distances ** 2))),
                p95=float(np.quantile(distances, .95)),
                matched_points=int(len(distances)))


def load_contact_hand(contact_npz, target_frames, start_frame, max_frame_gap=3,
                      quality_csv=None, source_fps=SOURCE_FPS):
    with np.load(contact_npz, allow_pickle=True) as z:
        frames = np.asarray(z['_frame_num'], dtype=np.int64)
        hand = np.stack([np.asarray(z['mano.j3d.cam.' + s], dtype='float32')
                         for s in ('r', 'l')], axis=1)[:, :, SEMANTIC_11]
        valid = np.stack([np.asarray(z[s + '_valid'], dtype='float32') > .5
                          for s in ('right', 'left')], axis=1)
        joint_valid = np.stack([np.asarray(z['joints_valid_' + s]) > .5
                                for s in ('r', 'l')], axis=1)[:, :, SEMANTIC_11]
        row_valid = np.asarray(z['is_valid']) > .5
        object_vertices = np.asarray(z['object.v.cam'], dtype='float32')
        object_t = np.asarray(z['object.cam_t'], dtype='float32')
        object_rot = np.asarray(z['object.rot'], dtype='float32')
    local = frames - start_frame
    if len(frames) < 2 or np.any(np.diff(frames) <= 0):
        raise ValueError('EPIC-Contact frame numbering is invalid')
    if quality_csv is None:
        raise ValueError('Contact quality CSV is required')
    quality = contact_quality(quality_csv, Path(contact_npz).stem, frames)
    semantic_valid = joint_valid & np.isfinite(hand).all(-1)
    valid &= semantic_valid.all(-1) & row_valid[:, None] & quality
    valid &= (np.isfinite(object_t).all(-1) & np.isfinite(object_rot).all((-1, -2)))[:, None]
    # The Contact camera has a different metric origin from SpaTracker.  Keep
    # only object-relative hand geometry here; it will be placed by fitting the
    # Contact object mesh to the scene depth mask for each target frame.
    hand_can = np.empty_like(hand)
    object_can = np.empty_like(object_vertices)
    for i in range(len(frames)):
        hand_can[i] = np.einsum('svi,ij->svj', hand[i] - object_t[i], object_rot[i])
        object_can[i] = np.einsum('vi,ij->vj', object_vertices[i] - object_t[i], object_rot[i])
    source_times = local.astype(float) / source_fps
    target_times = target_frames.astype(float) / source_fps
    lo, hi, exact, gap = grid_support(source_times, local, target_times,
                                       max_frame_gap=max_frame_gap)
    out = np.zeros((len(target_frames), 2, 11, 3), dtype='float32')
    for side in range(2):
        alpha = ((target_times - source_times[lo]) /
                 np.maximum(source_times[hi] - source_times[lo], 1e-9))
        out[:, side] = (hand_can[lo, side] * (1 - alpha[:, None, None])
                        + hand_can[hi, side] * alpha[:, None, None])
    outside = (target_times < source_times[0]) | (target_times > source_times[-1])
    out_valid = valid[lo] & valid[hi] & ~(gap | outside)[:, None]
    out[~out_valid] = 0
    # The mesh is effectively static in Contact canonical coordinates.  The
    # per-frame spread is recorded so this assumption is auditable.
    canonical = object_can[0]
    canonical_spread = float(np.median(np.linalg.norm(object_can - canonical[None], axis=2)))
    return out, out_valid, ~exact, canonical.astype('float32'), dict(
        source_rows=int(len(frames)), source_frame_range=[int(frames[0]), int(frames[-1])],
        target_valid_frames=out_valid.sum(0).tolist(), interpolated_frames=int((~exact).sum()),
        gap_crossing_frames=int(gap.sum()), max_source_gap_allowed=int(max_frame_gap),
        outside_source_support_frames=int(outside.sum()),
        invalid_semantic_joint_entries=int((~semantic_valid).sum()),
        source_qualified_side_rows=valid.sum(0).tolist(),
        quality_qualified_side_rows=quality.sum(0).tolist(),
        quality_csv_sha256=sha(quality_csv),
        strict_gap_crossing_frames=int(grid_support(source_times, local, target_times,
                                                    max_frame_gap=2)[3].sum()),
        source_sha256=sha(contact_npz), canonical_mesh_vertices=int(len(canonical)),
        canonical_mesh_frame_spread_m=canonical_spread)


def align_contact_hand_to_scene(hand_can, hand_valid, canonical_object, depths, Ks,
                                object_masks, w2cs, target_frames):
    """Place Contact hand geometry using only per-frame scene object depth."""
    out = np.zeros_like(hand_can, dtype='float32')
    aligned_valid = np.zeros_like(hand_valid, dtype=bool)
    fits = []
    previous_fit = None
    for j, frame in enumerate(target_frames):
        mask = object_masks[frame] & (depths[frame] > .05)
        yy, xx = np.nonzero(mask)
        if len(xx) < 32:
            fits.append(dict(frame=int(frame), valid=False, reason='too_few_depth_pixels'))
            continue
        pix = np.stack((xx, yy), axis=1).astype('float64')
        scene_object = backproject(pix, depths[frame, yy, xx], Ks[frame])
        # Fit in the stationary SpaTracker world frame.  This makes the prior
        # frame a meaningful initialization and prevents camera-motion-induced
        # orientation flips in the partially occluded plate cloud.
        scene_object = camera_to_world(scene_object, w2cs[frame])
        fit = fit_similarity(canonical_object, scene_object, initial=previous_fit,
                             fixed_scale=(previous_fit['scale'] if previous_fit else None))
        if fit is None:
            fits.append(dict(frame=int(frame), valid=False, reason='fit_failed'))
            continue
        fit_ok = fit['rms'] <= .03 and .35 <= fit['scale'] <= 3.0
        fit_record = dict(frame=int(frame), valid=bool(fit_ok), rms_m=fit['rms'],
                          p95_m=fit['p95'], scale=fit['scale'],
                          matched_points=fit['matched_points'])
        fits.append(fit_record)
        if not fit_ok:
            continue
        previous_fit = fit
        transformed = fit['scale'] * hand_can[j] @ fit['rotation'] + fit['translation']
        valid = hand_valid[j]
        for side in range(2):
            if valid[side]:
                out[j, side] = transformed[side].astype('float32')
                aligned_valid[j, side] = True
    rms = [x['rms_m'] for x in fits if x.get('valid')]
    return out, aligned_valid, dict(
        method='contact_object_similarity_to_scene_depth',
        direct_contact_absolute_camera_coordinates_used=False,
        valid_frames=aligned_valid.sum(0).tolist(),
        fit_valid_frames=int(sum(x.get('valid', False) for x in fits)),
        fit_rms_median=float(np.median(rms)) if rms else None,
        fit_rms_p95=float(np.quantile(rms, .95)) if rms else None,
        fit_records=fits)


def static_convention_diagnostic(depths, Ks, extrinsics, target_frames):
    """Compare depth reprojection residual under both matrix conventions."""
    h, w = depths.shape[1:]
    ys, xs = np.mgrid[10:h - 10:8, 10:w - 10:8]
    keep = depths[0, ys, xs] > .05
    pix = np.stack((xs[keep], ys[keep]), axis=-1)
    dep = depths[0, ys, xs][keep]
    result = {}
    for name, w2c in [('as_c2w', extrinsics), ('as_w2c', extrinsics)]:
        matrices = np.linalg.inv(extrinsics) if name == 'as_c2w' else extrinsics
        world = camera_to_world(backproject(pix, dep, Ks[0]), matrices[0])
        residuals = []
        for frame in target_frames[1:]:
            camera = world_to_camera(world, matrices[frame])
            q = camera @ Ks[frame].T
            uv = q[:, :2] / np.maximum(q[:, 2:,], 1e-8)
            ui = np.rint(uv[:, 0]).astype(int); vi = np.rint(uv[:, 1]).astype(int)
            good = ((camera[:, 2] > .05) & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h))
            ds = np.zeros(len(world)); ds[good] = depths[frame, vi[good], ui[good]]
            good &= ds > .05
            residuals.extend(np.abs(ds[good] - camera[good, 2]).tolist())
        result[name] = dict(count=len(residuals), median_m=float(np.median(residuals)),
                            p95_m=float(np.quantile(residuals, .95)))
    return result


def infer_extrinsics_convention(diagnostic, min_ratio=5.0):
    """Choose the matrix direction only when static depth makes it clear."""
    medians = {
        'c2w': float(diagnostic['as_c2w']['median_m']),
        'w2c': float(diagnostic['as_w2c']['median_m']),
    }
    if not all(np.isfinite(v) and v > 0 for v in medians.values()):
        raise ValueError('cannot infer extrinsics convention from invalid residuals')
    winner = min(medians, key=medians.get)
    loser = 'w2c' if winner == 'c2w' else 'c2w'
    ratio = medians[loser] / medians[winner]
    if ratio < min_ratio:
        raise ValueError(
            'ambiguous extrinsics convention: reprojection separation '
            f'{ratio:.3g}x is below the {min_ratio:.3g}x threshold')
    return dict(convention=winner, separation_ratio=float(ratio),
                median_residual_m=medians)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scene-dir', type=Path, required=True)
    p.add_argument('--contact-npz', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--start-frame', type=int)
    p.add_argument('--window-start', type=int, default=0)
    p.add_argument('--quality-csv', type=Path, required=True)
    p.add_argument('--of-train-file', type=Path, required=True)
    p.add_argument('--of-val-file', type=Path, required=True)
    p.add_argument('--extrinsics-convention', choices=('auto', 'w2c', 'c2w'),
                   default='auto')
    p.add_argument('--metadata-convention', choices=('w2c', 'c2w'))
    p.add_argument('--convention-min-separation', type=float, default=5.0)
    p.add_argument('--max-static-points', type=int, default=8192)
    p.add_argument('--max-object-points', type=int, default=512)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    scene = args.scene_dir
    with np.load(scene / 'spatracker.npz') as z:
        depths = np.asarray(z['depths'], dtype='float32')
        Ks = np.asarray(z['intrinsics'], dtype='float32')
        raw_extrinsics = np.asarray(z['extrinsics'], dtype='float32')
    if not rigid_valid(raw_extrinsics).all():
        raise ValueError('invalid SpaTracker extrinsics')
    n_frames, h, w = depths.shape
    meta_path = scene / 'action.meta.json'
    start_frame, source_fps = source_clock(json.loads(meta_path.read_text()), n_frames,
                                         args.start_frame)
    splits = source_splits(scene.name, args.quality_csv, args.of_train_file, args.of_val_file)
    gray = read_video(scene / 'action.mp4', (w, h))
    if len(gray) != n_frames:
        raise ValueError('video and SpaTracker frame counts differ')
    object_masks = load_object_masks(scene / 'objects/0+object_0/masks.npz', (h, w), n_frames)
    hand_masks = load_hand_masks(scene / 'egohos/twohands_masks.npz', (h, w), n_frames)
    if args.window_start < 0:
        raise ValueError('window start must be nonnegative')
    target_frames = args.window_start + np.arange(WINDOW, dtype=np.int64) * 2
    if target_frames[-1] >= n_frames:
        raise ValueError('clip is shorter than one 4+24 30-Hz window')
    # Convention is a clip-level property.  Use the full clip (bounded to 64
    # evenly spaced frames) rather than the short training window, because a
    # low-motion 0.9 s window can make both directions look deceptively close.
    diagnostic_frames = np.unique(np.linspace(
        0, n_frames - 1, min(n_frames, 64), dtype=np.int64))
    diagnostic = static_convention_diagnostic(
        depths, Ks, raw_extrinsics, diagnostic_frames)
    inference = infer_extrinsics_convention(
        diagnostic, min_ratio=args.convention_min_separation)
    selected_convention = (inference['convention'] if args.extrinsics_convention == 'auto'
                           else args.extrinsics_convention)
    # Output files are produced by the upstream SpaTracker script as inverse
    # c2w matrices; source provenance and the selected convention are recorded.
    w2cs = (raw_extrinsics if selected_convention == 'w2c'
            else np.linalg.inv(raw_extrinsics))
    if not rigid_valid(w2cs).all():
        raise ValueError('selected camera convention is not rigid')
    static, static_valid, static_audit = static_tracks(
        depths, Ks, w2cs, object_masks, hand_masks, target_frames,
        max_points=args.max_static_points)
    moved, moved_valid, moved_audit = object_tracks(
        depths, Ks, w2cs, object_masks, gray, target_frames,
        max_points=args.max_object_points)
    (hand_can, hand_valid, hand_interpolated, canonical_object,
     hand_audit) = load_contact_hand(args.contact_npz, target_frames, start_frame,
                                    quality_csv=args.quality_csv, source_fps=source_fps)
    hand, hand_valid, alignment_audit = align_contact_hand_to_scene(
        hand_can, hand_valid, canonical_object, depths, Ks, object_masks, w2cs,
        target_frames)
    hand_audit.update(alignment_audit)
    scene_points = np.concatenate((static, moved), axis=1)
    scene_valid = np.concatenate((static_valid, moved_valid), axis=1)
    point_kind = np.concatenate((np.zeros(static.shape[1], np.int8),
                                 np.ones(moved.shape[1], np.int8)))
    c2ws = np.linalg.inv(w2cs)
    out_npz = args.output / 'scene_flow_candidate.npz'
    np.savez_compressed(
        out_npz, scene_points_world=scene_points.astype('float32'),
        point_track_valid=scene_valid, point_kind=point_kind,
        hand_points_world=hand, hand_valid=hand_valid,
        hand_interpolated_mask=hand_interpolated,
        source_frame_ids=target_frames + start_frame, local_frame_ids=target_frames,
        timestamps=target_frames / source_fps,
        camera_c2w=c2ws.astype('float32'),
        static_point_count=np.asarray(static.shape[1]), moved_point_count=np.asarray(moved.shape[1]))
    audit = dict(
        status='CANDIDATE_ONLY', training_allowed=False, schema=SCHEMA,
        source_clip=scene.name,
        source_scene_dir=str(scene.resolve()),
        source_start_frame=start_frame, source_splits=splits,
        source_scene_sha256={
            'action.mp4': sha(scene / 'action.mp4'),
            'action.meta.json': sha(meta_path),
            'spatracker.npz': sha(scene / 'spatracker.npz'),
            'object_masks.npz': sha(scene / 'objects/0+object_0/masks.npz'),
            'twohands_masks.npz': sha(scene / 'egohos/twohands_masks.npz')},
        contact_npz_sha256=sha(args.contact_npz),
        coordinate_frame='SpaTracker world from explicit camera convention',
        extrinsics_convention=selected_convention,
        convention_inference=dict(
            requested=args.extrinsics_convention,
            metadata=args.metadata_convention,
            inferred=inference['convention'],
            separation_ratio=inference['separation_ratio'],
            metadata_mismatch=(args.metadata_convention is not None
                               and args.metadata_convention != inference['convention']),
            min_separation=float(args.convention_min_separation),
            diagnostic_frame_count=int(len(diagnostic_frames)),
            median_residual_m=inference['median_residual_m']),
        extrinsics_provenance='ObjectForesight-Data step9_spatracker.py saves inverse(c2w_traj)',
        upstream_source='https://raw.githubusercontent.com/RustinS/ObjectForesight-Data/main/step9_spatracker.py',
        forbidden_supervision=['FoundationPose T_c_o', 'TRELLIS mesh', 'propagated object pose'],
        target_fps=TARGET_FPS, sampled_fps=source_fps / 2,
        strict_native_30hz_clock=bool(np.isclose(2 / source_fps, 1 / TARGET_FPS, atol=1e-5, rtol=0)),
        window_start_local_frame=args.window_start,
        window_frames=WINDOW, source_fps=source_fps,
        static_tracks=static_audit, moved_object_tracks=moved_audit, hand=hand_audit,
        scene_points=int(scene_points.shape[1]),
        scene_valid_fraction=float(scene_valid.mean()),
        scene_points_valid_all_frames=int(scene_valid.all(0).sum()),
        scene_points_valid_ge_08=int((scene_valid.mean(0) >= .8).sum()),
        hand_valid_frames=hand_valid.sum(0).tolist(),
        h4_k24_windows=int(hand_valid[:, 0].all() and moved_valid.all(0).sum() >= 16),
        moved_points_valid_all_frames=int(moved_valid.all(0).sum()),
        camera_self_consistency=diagnostic,
        candidate_npz_sha256=sha(out_npz),
        limitations=['single clip; window count is candidate geometry only, not training eligibility',
                     '29.97 Hz subsampling is not native 30 Hz resampling',
                     'rigid-object native decoder cannot consume scene point trajectories',
                     'LK status does not prove drift-free surface correspondence',
                     'camera convention must be audited per release',
                     'scene tracks use depth/RGB pseudo-labels',
                     'hand is EPIC-Contact object-relative geometry placed by depth-mask fit',
                     'Contact and SpaTracker absolute camera origins are not shared',
                     'no stationary-world training manifest emitted'])
    (args.output / 'audit.json').write_text(json.dumps(audit, indent=2, allow_nan=False) + '\n')
    print(json.dumps(audit, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
