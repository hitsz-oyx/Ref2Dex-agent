"""Metric scene/hand contracts for the bounded PointWAM data-pipeline smoke.

No model, training, or robot-action semantics live here. Offline reconstruction
is teacher data and is not certified causal perception. Palm offsets are an
explicit input: the paper does not publish its unquantized offset table.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import h5py
import numpy as np

HAND_ORDER = ("right", "left")
JOINT_NAMES = ("Hand", "MiddleFingerKnuckle", "IndexFingerKnuckle",
               "LittleFingerKnuckle", "RingFingerKnuckle", "ThumbTip",
               "IndexFingerTip", "MiddleFingerTip", "RingFingerTip")
# ARKit gravity is +y up. Use a proper rotation, never a reflection.
ARKIT_TO_ZUP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_identity(hdf5: Path, video: Path, manifest: Path | None = None) -> dict:
    """Keep release split distinct from pilot admission; verify TRAIN provenance."""
    hdf5, video = hdf5.resolve(), video.resolve()
    if video != hdf5.with_suffix(".mp4"):
        raise ValueError("source files must be one original paired episode")
    if manifest is None:
        if hdf5.parent.parent.name != "test":
            raise ValueError("non-test input requires a verified acquisition manifest")
        return dict(split="official_test", training_allowed=False,
                    episode_id="test/" + hdf5.parent.name + "/" + hdf5.stem)
    manifest = manifest.resolve()
    data = json.loads(manifest.read_text())
    urls = {f"https://ml-site.cdn-apple.com/datasets/egodex/part{i}.zip":
            ("official_train", f"part{i}") for i in range(1, 6)}
    urls["https://ml-site.cdn-apple.com/datasets/egodex/test.zip"] = ("official_test", "test")
    expected = urls.get(data.get("url"))
    if (data.get("status") != "COMPLETED" or expected is None
            or data.get("split") != expected[0]):
        raise ValueError("unqualified release split or incomplete acquisition")
    members = []
    for path in (hdf5, video):
        member = path.relative_to(manifest.parent / "raw").as_posix()
        if Path(member).parts[0] != expected[1]:
            raise ValueError("archive member disagrees with release split")
        records = [r for r in data.get("files", []) if r["member"] == member]
        if (len(records) != 1 or records[0]["bytes"] != path.stat().st_size
                or records[0]["sha256"] != sha256(path)):
            raise ValueError("source input differs from verified acquisition")
        members.append(member)
    return dict(split=expected[0], training_allowed=False,
                episode_id=members[0][:-5], acquisition_manifest=str(manifest),
                acquisition_manifest_sha256=sha256(manifest),
                source_archive_etag=data.get("archive_etag"))


def load_episode(path: str | Path) -> dict:
    """Retain original named joints, confidence masks, camera and language."""
    with h5py.File(path, "r") as root:
        camera = np.asarray(root["transforms/camera"], dtype=np.float32)
        n = len(camera)
        xyz = np.full((n, 2, len(JOINT_NAMES), 3), np.nan, np.float32)
        valid = np.zeros(xyz.shape[:-1], bool)
        for side_id, side in enumerate(HAND_ORDER):
            for joint_id, suffix in enumerate(JOINT_NAMES):
                name = side + suffix
                if name not in root["transforms"]:
                    continue
                transform = np.asarray(root["transforms"][name], dtype=np.float32)
                if transform.shape != (n, 4, 4):
                    raise ValueError(f"invalid transform shape: {name}")
                keep = np.isfinite(transform).all(axis=(1, 2))
                confidence = root.get("confidences/" + name)
                if confidence is not None:
                    confidence = np.asarray(confidence)
                    keep &= np.isfinite(confidence) & (confidence > 0)
                xyz[:, side_id, joint_id] = transform[:, :3, 3]
                valid[:, side_id, joint_id] = keep
        xyz[~valid] = np.nan
        intrinsic = np.asarray(root["camera/intrinsic"], dtype=np.float32)
        attrs = {k: str(v.decode() if isinstance(v, bytes) else v)
                 for k, v in root.attrs.items()}
    if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
        raise ValueError("invalid camera intrinsic")
    rotation = camera[:, :3, :3]
    if (not np.isfinite(camera).all()
            or not np.allclose(rotation.transpose(0, 2, 1) @ rotation, np.eye(3), atol=1e-3)
            or not np.allclose(np.linalg.det(rotation), 1, atol=1e-3)
            or not np.allclose(camera[:, 3], [0, 0, 0, 1], atol=1e-5)):
        raise ValueError("invalid camera-to-origin SE(3)")
    language_key = "llm_description"
    if attrs.get("llm_type") == "reversible" and attrs.get("which_llm_description") == "2":
        language_key = "llm_description2"
    return dict(joints=xyz, joint_valid=valid, camera_to_origin=camera,
                intrinsics=intrinsic, language=attrs.get(language_key, ""), attrs=attrs)


def palm_keypoints(joints: np.ndarray, valid: np.ndarray,
                   offsets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Six rigid offsets in a middle-knuckle frame, then four observed tips.

    x points wrist->middle knuckle; y points little->index knuckle with its
    x component removed; z=x cross y. Each hand is constructed independently.
    Offsets may be (6,3) or side-specific (2,6,3), in this documented basis.
    """
    joints = np.asarray(joints, dtype=np.float32)
    valid = np.asarray(valid, bool) & np.isfinite(joints).all(-1)
    if joints.shape[-2:] != (len(JOINT_NAMES), 3) or valid.shape != joints.shape[:-1]:
        raise ValueError("invalid named-joint contract")
    offsets = np.asarray(offsets, np.float32)
    if offsets.shape not in ((6, 3), (2, 6, 3)) or not np.isfinite(offsets).all():
        raise ValueError("six finite explicit palm offsets are required")
    origin = joints[..., 1, :]
    x = origin - joints[..., 0, :]
    xn = np.linalg.norm(x, axis=-1, keepdims=True)
    x = x / np.maximum(xn, 1e-8)
    y = joints[..., 2, :] - joints[..., 3, :]
    y = y - np.sum(y * x, axis=-1, keepdims=True) * x
    yn = np.linalg.norm(y, axis=-1, keepdims=True)
    y = y / np.maximum(yn, 1e-8)
    basis = np.stack((x, y, np.cross(x, y)), axis=-1)
    frame_valid = valid[..., :4].all(-1) & (xn[..., 0] > 1e-6) & (yn[..., 0] > 1e-6)
    palm = origin[..., None, :] + np.einsum("...ij,...kj->...ki", basis, offsets)
    out = np.concatenate((palm, joints[..., 5:9, :]), axis=-2)
    mask = np.concatenate((np.repeat(frame_valid[..., None], 6, axis=-1), valid[..., 5:9]), axis=-1)
    out[~mask] = np.nan
    return out, mask


def candidate_clips(joints: np.ndarray, valid: np.ndarray,
                    max_clips: int = 2) -> tuple[list[np.ndarray], list[dict]]:
    """30Hz source -> 11 frames at 10Hz, 1s duration, .5s stride."""
    selected, audit = [], []
    for start in range(0, len(joints) - 30, 15):
        ids = start + np.arange(11) * 3
        mask = valid[ids, :, 0] & np.isfinite(joints[ids, :, 0]).all(-1)
        delta = np.linalg.norm(np.diff(joints[ids, :, 0], axis=0), axis=-1)
        # A fully observed wrist is needed to make a motion-filter decision.
        complete = mask.all(0)
        motion = np.sum(np.where(mask[1:] & mask[:-1], delta, 0), axis=0)
        moving = complete & (motion >= .02)
        audit.append(dict(start_frame=start, wrist_path_m=motion.tolist(),
                          wrist_complete=complete.tolist(), passes_motion=bool(moving.any())))
        if moving.any() and len(selected) < max_clips:
            selected.append(ids)
    return selected, audit


def resize_intrinsics(intrinsic: np.ndarray, original_wh: tuple[int, int],
                      target_wh: tuple[int, int]) -> np.ndarray:
    out = np.array(intrinsic, dtype=np.float32, copy=True)
    out[0] *= target_wh[0] / original_wh[0]
    out[1] *= target_wh[1] / original_wh[1]
    return out


def resize_valid_depth(depth: np.ndarray, target_wh: tuple[int, int]) -> np.ndarray:
    """Resize without converting invalid-depth boundaries into valid surfaces."""
    output = []
    for frame in np.asarray(depth, np.float32):
        valid = np.isfinite(frame) & (frame > 0)
        resized = cv2.resize(np.where(valid, frame, 0), target_wh, interpolation=cv2.INTER_LINEAR)
        support = cv2.resize(valid.astype(np.float32), target_wh, interpolation=cv2.INTER_LINEAR)
        resized[support < 1-1e-6] = 0
        output.append(resized)
    return np.stack(output)


def sample_depth(depth: np.ndarray, uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear depth; require all four sampled pixels finite and positive."""
    depth, uv = np.asarray(depth, np.float32), np.asarray(uv, np.float32)
    if depth.ndim != 3 or uv.shape[0] != len(depth) or uv.shape[-1] != 2:
        raise ValueError("depth/uv contract mismatch")
    h, w = depth.shape[1:]
    flat = uv.reshape(len(depth), -1, 2)
    result = np.zeros(flat.shape[:-1], np.float32)
    mask = np.isfinite(flat).all(-1) & (flat[..., 0] >= 0) & (flat[..., 0] < w - 1)
    mask &= (flat[..., 1] >= 0) & (flat[..., 1] < h - 1)
    for t in range(len(depth)):
        safe = np.nan_to_num(flat[t], nan=-1, posinf=-1, neginf=-1)
        x = np.clip(np.floor(safe[:, 0]).astype(np.int64), 0, w - 2)
        y = np.clip(np.floor(safe[:, 1]).astype(np.int64), 0, h - 2)
        pixels = np.stack((depth[t, y, x], depth[t, y, x+1],
                           depth[t, y+1, x], depth[t, y+1, x+1]), -1)
        mask[t] &= np.isfinite(pixels).all(-1) & (pixels > 0).all(-1)
        dx, dy = safe[:, 0]-x, safe[:, 1]-y
        weight = np.stack(((1-dx)*(1-dy), dx*(1-dy), (1-dx)*dy, dx*dy), -1)
        result[t] = np.sum(np.where(np.isfinite(pixels), pixels, 0) * weight, -1)
    result[~mask] = 0
    return result.reshape(uv.shape[:-1]), mask.reshape(uv.shape[:-1])


def project_world(xyz: np.ndarray, intrinsic: np.ndarray,
                  camera_to_origin: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    camera = np.einsum("tji,t...j->t...i", camera_to_origin[:, :3, :3],
                       xyz - camera_to_origin[:, None, None, :3, 3])
    homogeneous = camera @ intrinsic.T
    uv = homogeneous[..., :2] / np.where(camera[..., 2:] > 0, camera[..., 2:], np.nan)
    return uv, camera[..., 2]


def recover_metric_scale(depth: np.ndarray, joints: np.ndarray, valid: np.ndarray,
                         intrinsic: np.ndarray, camera_to_origin: np.ndarray) -> dict:
    """Paper B.1 hand z/depth median, excluding ratios outside [.5,2]*median."""
    uv, z = project_world(joints, intrinsic, camera_to_origin)
    sampled, depth_valid = sample_depth(depth, uv)
    keep = np.asarray(valid, bool) & depth_valid & np.isfinite(z) & (z > 0)
    ratios = z[keep] / sampled[keep]
    ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
    if len(ratios) < 20:
        raise ValueError(f"insufficient hand/depth scale samples: {len(ratios)}")
    median = float(np.median(ratios))
    inlier = ratios[(ratios >= median / 2) & (ratios <= median * 2)]
    if len(inlier) < 20:
        raise ValueError("insufficient hand/depth scale inliers")
    scale = float(np.median(inlier))
    return dict(scale=scale, samples=int(len(ratios)), inliers=int(len(inlier)),
                relative_iqr=float(np.subtract(*np.quantile(inlier, [.75, .25]))/scale),
                per_frame_valid_samples=keep.sum((1, 2)).tolist(),
                note="joint-to-visible-surface depth calibration is approximate")


def lift_tracks(uv: np.ndarray, visibility: np.ndarray, depth: np.ndarray,
                intrinsic: np.ndarray, camera_to_origin: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z, good_depth = sample_depth(depth, uv)
    valid = np.asarray(visibility, bool) & good_depth
    rays = np.concatenate((uv, np.ones(uv.shape[:-1] + (1,), np.float32)), -1) @ np.linalg.inv(intrinsic).T
    camera = rays * z[..., None]
    world = np.einsum("tij,tnj->tni", camera_to_origin[:, :3, :3], camera)
    world += camera_to_origin[:, None, :3, 3]
    valid &= np.isfinite(world).all(-1)
    world[~valid] = np.nan
    return world.astype(np.float32), valid


def resample_valid(xyz: np.ndarray, valid: np.ndarray, target_frames: int = 31) -> tuple[np.ndarray, np.ndarray]:
    """Keep original observations and mask interpolation across missing gaps."""
    xyz = np.asarray(xyz, np.float32)
    valid = np.asarray(valid, bool) & np.isfinite(xyz).all(-1)
    if len(xyz) < 2 or valid.shape != xyz.shape[:-1] or target_frames < 2:
        raise ValueError("invalid interpolation contract")
    positions = np.linspace(0, len(xyz)-1, target_frames)
    out = np.full((target_frames,)+xyz.shape[1:], np.nan, np.float32)
    mask = np.zeros((target_frames,)+valid.shape[1:], bool)
    for t, position in enumerate(positions):
        nearest = int(round(position))
        if abs(position-nearest) < 1e-7:
            mask[t] = valid[nearest]
            out[t] = np.where(mask[t][..., None], xyz[nearest], np.nan)
        else:
            lo, hi = int(np.floor(position)), int(np.ceil(position))
            mask[t] = valid[lo] & valid[hi]
            out[t] = np.where(mask[t][..., None],
                              xyz[lo]*(hi-position)+xyz[hi]*(position-lo), np.nan)
    return out, mask


def select_scene(world: np.ndarray, valid: np.ndarray, max_points: int = 12000) -> np.ndarray:
    """Only initial observation selects identities; never filter on future motion."""
    candidates = np.flatnonzero(valid[0] & np.isfinite(world[0]).all(-1))
    if not len(candidates):
        raise ValueError("no initial visible/depth-valid scene points")
    voxels = np.floor(world[0, candidates] / .015).astype(np.int64)
    _, first = np.unique(voxels, axis=0, return_index=True)
    chosen = candidates[np.sort(first)]
    if len(chosen) > max_points:
        chosen = chosen[np.linspace(0, len(chosen)-1, max_points).astype(np.int64)]
    return chosen


def pack_clip(scene: np.ndarray, scene_valid: np.ndarray, hand: np.ndarray,
              hand_valid: np.ndarray, initial_rgb: np.ndarray, source_frame_ids: np.ndarray,
              language: str) -> dict:
    """31 positions include t0; each displacement target has exactly 30 steps."""
    if len(scene) != 11 or len(hand) != 11 or len(source_frame_ids) != 11:
        raise ValueError("raw clip must contain 11 aligned observations")
    if not np.array_equal(np.diff(source_frame_ids), np.full(10, 3)):
        raise ValueError("expected 10Hz samples from 30Hz source")
    scene = scene @ ARKIT_TO_ZUP.T
    hand = hand @ ARKIT_TO_ZUP.T
    chosen = select_scene(scene, scene_valid)
    sx, sm = resample_valid(scene[:, chosen], scene_valid[:, chosen])
    hx, hm = resample_valid(hand, hand_valid)
    s_target = sx[1:]-sx[0]
    h_target = hx[1:]-hx[0]
    s_mask = sm[1:] & sm[0]
    h_mask = hm[1:] & hm[0]
    s_target[~s_mask] = np.nan
    h_target[~h_mask] = np.nan
    return dict(scene_xyz_m=sx, scene_valid=sm, scene_rgb=initial_rgb[chosen],
                hand_xyz_m=hx, hand_valid=hm, hand_order=np.asarray(HAND_ORDER),
                scene_displacement_m=s_target, scene_target_clipped_m=np.clip(s_target, -.5, .5),
                scene_target_valid=s_mask, hand_displacement_m=h_target, hand_target_valid=h_mask,
                track_ids=chosen, source_frame_ids=source_frame_ids,
                timestamps_s=np.linspace(source_frame_ids[0]/30, source_frame_ids[-1]/30, 31),
                language=np.asarray(language), arkit_to_zup=ARKIT_TO_ZUP,
                training_allowed=np.asarray(False),
                schema=np.asarray("ref2dex.pointwam-data-smoke.v1"))
