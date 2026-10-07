"""Native ContactPose transport trajectories; never recover world motion from Stage3.

Official annotations store world-in-object ``oTw`` and (for moving hands)
object-in-hand ``hTo``. Fixed hand articulation is an annotation assumption,
not a measurement of finger motion or time-varying contact.
"""
import hashlib
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

SEMANTICS = ['wrist', 'thumb_mcp', 'index_mcp', 'middle_mcp', 'ring_mcp',
             'little_mcp', 'thumb_tip', 'index_tip', 'middle_tip', 'ring_tip', 'little_tip']
# OpenPose wrist + five first finger joints + five fingertips. The legacy
# thumb_mcp name denotes the first thumb joint, anatomically the thumb CMC.
OPENPOSE_11 = [0, 1, 5, 9, 13, 17, 4, 8, 12, 16, 20]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def subject_split(subject):
    value = int(hashlib.sha256(('contactpose:' + subject).encode()).hexdigest()[:8], 16) % 100
    return 'train' if value < 80 else ('val' if value < 90 else 'test')


def rigid_valid(poses):
    finite = np.isfinite(poses).all((-1, -2))
    R = np.where(finite[..., None, None], poses[..., :3, :3], np.eye(3))
    return (finite & (np.abs(R.swapaxes(-1, -2) @ R - np.eye(3)).max((-1, -2)) < 1e-3)
            & (np.abs(np.linalg.det(R) - 1) < 1e-3)
            & (np.abs(poses[..., 3, :] - [0, 0, 0, 1]).max(-1) < 1e-6))


def pose_matrix(payload):
    q = np.asarray(payload['rotation'], dtype=float)
    t = np.asarray(payload['translation'], dtype=float)
    if q.shape != (4,) or t.shape != (3,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-9:
        raise ValueError('invalid ContactPose quaternion/translation')
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
    result[:3, 3] = t
    return result


def native_trajectory(annotation):
    """Return real per-frame clock, world object pose, and ordered world joints."""
    frames = annotation['frames']
    cameras = [key for key, value in annotation['cameras'].items() if value['valid']]
    if not cameras or len(frames) < 2:
        raise ValueError('missing valid camera or trajectory')
    camera = 'kinect2_middle' if 'kinect2_middle' in cameras else sorted(cameras)[0]
    # Subtract integer origin before conversion, preserving nanosecond resolution.
    origin = frames[0]['time'][camera]
    ns0 = int(origin['sec']) * 10**9 + int(origin['nsec'])
    times = np.asarray([(int(f['time'][camera]['sec']) * 10**9 + int(f['time'][camera]['nsec']) - ns0) / 1e9
                        for f in frames], dtype=float)
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('ContactPose camera clock must be strictly increasing')
    poses = np.full((len(frames), 4, 4), np.nan)
    hand = np.zeros((len(frames), 2, 11, 3))
    valid = np.zeros((len(frames), 2), bool)
    relative = np.full((len(frames), 2, 4, 4), np.nan)
    joints = np.zeros((2, 11, 3))
    for slot, hand_idx in enumerate((1, 0)):  # official [left,right] -> WM [right,left]
        h = annotation['hands'][hand_idx]
        if not h['valid']:
            continue
        raw = np.asarray(h['joints'], dtype=float)
        if raw.shape != (21, 3) or not np.isfinite(raw).all():
            raise ValueError('invalid native 21-joint hand')
        joints[slot] = raw[OPENPOSE_11]
        for k, frame in enumerate(frames):
            try:
                relative[k, slot] = np.linalg.inv(pose_matrix(frame['hTo'][hand_idx])) if h.get('moving') else np.eye(4)
                valid[k, slot] = rigid_valid(relative[k, slot])
            except (KeyError, IndexError, ValueError, np.linalg.LinAlgError):
                pass
    for k, frame in enumerate(frames):
        try:
            poses[k] = np.linalg.inv(pose_matrix(frame['oTw']))
        except (KeyError, ValueError, np.linalg.LinAlgError):
            pass
        for slot in range(2):
            if valid[k, slot] and rigid_valid(poses[k]):
                T = poses[k] @ relative[k, slot]
                hand[k, slot] = joints[slot] @ T[:3, :3].T + T[:3, 3]
    present = np.array([bool(annotation['hands'][i]['valid']) for i in (1, 0)])
    if not present.any():
        raise ValueError('no valid annotated hand')
    return dict(times=times, poses=poses, hand=hand, hand_valid=valid, relative=relative,
                joints=joints, present=present, camera=camera, timestamp_origin_ns=ns0)


def interpolate_pose(times, poses, query):
    result = np.broadcast_to(np.eye(4), (len(query), 4, 4)).copy()
    result[:, :3, :3] = Slerp(times, Rotation.from_matrix(poses[:, :3, :3]))(query).as_matrix()
    for axis in range(3):
        result[:, axis, 3] = np.interp(query, times, poses[:, axis, 3])
    return result


def resample_segments(raw, max_gap_s=.075, max_translation_speed=3., max_angular_speed=15.):
    """Split before interpolation at missing/invalid/anomalous raw edges.

    Each output segment has its own contiguous 30 Hz clock. No window can
    bridge a cut, and interpolated hands follow the same interpolated object.
    """
    times, poses = raw['times'], raw['poses']
    dt = np.diff(times)
    good = rigid_valid(poses) & (raw['hand_valid'][:, raw['present']]).all(-1)
    trans = np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=-1)
    R = poses[1:, :3, :3] @ poses[:-1, :3, :3].swapaxes(-1, -2)
    angle = np.arccos(np.clip((np.trace(R, axis1=1, axis2=2) - 1) / 2, -1, 1))
    hspeed = np.linalg.norm(np.diff(raw['hand'][:, raw['present']], axis=0), axis=-1).max((1, 2)) / dt
    edge_good = (good[:-1] & good[1:] & (dt <= max_gap_s)
                 & (trans / dt <= max_translation_speed) & (angle / dt <= max_angular_speed)
                 & (hspeed <= max_translation_speed))
    cuts = np.r_[0, np.flatnonzero(~edge_good) + 1, len(times)]
    packs = []
    for start, stop in zip(cuts[:-1], cuts[1:]):
        if stop - start < 2 or not good[start:stop].all():
            continue
        query = times[start] + np.arange(int(np.floor((times[stop-1] - times[start]) * 30 + 1e-8)) + 1) / 30
        if len(query) < 28:
            continue
        T = interpolate_pose(times[start:stop], poses[start:stop], query)
        hands = np.zeros((len(query), 2, 11, 3), np.float32)
        for slot in np.flatnonzero(raw['present']):
            relative = interpolate_pose(times[start:stop], raw['relative'][start:stop, slot], query)
            world = T @ relative
            hands[:, slot] = np.einsum('tij,nj->tni', world[:, :3, :3], raw['joints'][slot]) + world[:, None, :3, 3]
        right = np.searchsorted(times, query, side='right').clip(start+1, stop-1)
        left = right - 1
        alpha = (query - times[left]) / (times[right] - times[left])
        nearest = np.where(alpha <= .5, left, right)
        packs.append(dict(hand=hands, hand_valid=np.broadcast_to(raw['present'], (len(query), 2)).copy(),
                          poses=T[:, None].astype('float32'), pose_valid=np.ones((len(query), 1), bool),
                          timestamps=query, frame_ids=nearest.astype('int64'), source_frame_ids=np.stack((left, right), -1),
                          interpolation_alpha=alpha, raw_range=[int(start), int(stop)]))
    return packs, dict(raw_frames=len(times), gaps=int((dt > max_gap_s).sum()),
                       invalid_raw_frames=int((~good).sum()), rejected_raw_edges=int((~edge_good).sum()))


def window_rows(pack, near):
    rows = []
    T = pack['poses'][:, 0]
    for tick in range(3, len(T) - 24):
        if not near[tick, 0] or not pack['pose_valid'][tick-3:tick+25].all():
            continue
        future = T[tick+1:tick+25]
        delta = np.linalg.norm(future[:, :3, 3] - T[tick, :3, 3], axis=-1)
        R = future[:, :3, :3] @ T[tick, :3, :3].T
        angle = np.arccos(np.clip((np.trace(R, axis1=1, axis2=2) - 1) / 2, -1, 1))
        moving = (delta > .002).any() or (angle > .02).any()
        rows.append([0, tick, 1 if moving else 2])
    return np.asarray(rows, dtype=np.int64).reshape(-1, 3)


def contactpose_windows(root, split):
    """Lazy Torch import for CPU-only conversion, retaining the established tensors."""
    from .data import Windows

    class ContactPoseWindows(Windows):
        @lru_cache(maxsize=32)
        def sequence(self, sequence):
            data = dict(super().sequence(sequence))
            clock = np.load(self.root / 'processed/sequences' / sequence / 'timestamps.npy', mmap_mode='r')
            if not np.allclose(np.diff(clock), 1/30, atol=1e-7, rtol=0):
                raise ValueError('ContactPose segment clock is not 30 Hz')
            # Parent has a legacy OakInk 120Hz-index assertion. The verified
            # resampled clock supplies this internal grid; raw IDs stay on disk.
            data['frame_ids'] = 4 * np.arange(len(clock), dtype=np.int64)
            return data

    return ContactPoseWindows(root, split)
