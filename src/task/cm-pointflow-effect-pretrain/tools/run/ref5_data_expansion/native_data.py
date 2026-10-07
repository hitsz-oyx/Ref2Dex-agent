"""30 Hz native-source contract, sharing the existing WM tensor interface."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

TASK = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(TASK / 'src'))
from oakink_wm.data import Windows

SEMANTICS = ['wrist', 'thumb_mcp', 'index_mcp', 'middle_mcp', 'ring_mcp',
             'little_mcp', 'thumb_tip', 'index_tip', 'middle_tip', 'ring_tip', 'little_tip']
# smplx MANO outputs wrist, index/middle/little/ring/thumb (three joints each).
MANO_MCP = [13, 1, 4, 10, 7]
MANO_TIPS = [744, 320, 443, 554, 671]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def rigid_valid(pose):
    finite = np.isfinite(pose).all((-1, -2))
    R = np.where(finite[..., None, None], pose[..., :3, :3], np.eye(3))
    return (finite & (np.abs(R.swapaxes(-1, -2) @ R - np.eye(3)).max((-1, -2)) < 1e-3)
            & (np.abs(np.linalg.det(R) - 1) < 1e-3)
            & (np.abs(pose[..., 3, :] - [0, 0, 0, 1]).max(-1) < 1e-6))


def arctic_poses(params):
    """Official ARCTIC: root rotvec; translation mm; top rotates by -angle about z."""
    from scipy.spatial.transform import Rotation
    root = np.broadcast_to(np.eye(4), (len(params), 4, 4)).copy()
    root[:, :3, :3] = Rotation.from_rotvec(params[:, 1:4]).as_matrix()
    root[:, :3, 3] = params[:, 4:7] / 1000
    articulation = np.broadcast_to(np.eye(4), root.shape).copy()
    articulation[:, :3, :3] = Rotation.from_rotvec(
        np.column_stack((np.zeros((len(params), 2)), -params[:, 0]))).as_matrix()
    return np.stack((root, root @ articulation), axis=1).astype('float32')


def eligible_rows(hand, hand_valid, poses, pose_valid, near, timestamps, centers):
    """Current-only anchors; all local objects and hand clocks valid through 0.8s."""
    from oakink_wm.data import local_objects
    dt_bad = ~np.isclose(np.diff(timestamps), 1 / 30, atol=1e-5, rtol=0)
    h_bad = ((np.linalg.norm(np.diff(hand, axis=0), axis=-1) > .15)
             & hand_valid[1:, :, None] & hand_valid[:-1, :, None]).any((1, 2))
    displacement = np.linalg.norm(np.diff(poses[..., :3, 3], axis=0), axis=-1)
    R = poses[1:, :, :3, :3] @ poses[:-1, :, :3, :3].swapaxes(-1, -2)
    angle = np.arccos(np.clip((np.trace(R, axis1=-2, axis2=-1) - 1) / 2, -1, 1))
    p_bad = (displacement > .15) | (angle > .6)
    rows = []
    for t in range(3, len(hand) - 24):
        start, stop = t - 3, t + 25
        if dt_bad[start:stop-1].any() or h_bad[start:stop-1].any():
            continue
        # Presence changes are excluded; absent hands remain fully masked.
        if not hand_valid[t].any() or not (hand_valid[start:stop] == hand_valid[t]).all():
            continue
        for anchor in np.flatnonzero(near[t] & pose_valid[t]):
            selected = local_objects(poses[t], pose_valid[t], anchor, centers)
            if not pose_valid[start:stop, selected].all() or p_bad[start:stop-1, selected].any():
                continue
            # Match OakInk's horizon predicate, including motion that returns
            # to the current pose before the last future frame.
            future = poses[t+1:t+25, anchor]
            delta = np.linalg.norm(future[:, :3, 3] - poses[t, anchor, :3, 3], axis=-1)
            relative_rotation = future[:, :3, :3] @ poses[t, anchor, :3, :3].T
            future_angle = np.arccos(np.clip(
                (np.trace(relative_rotation, axis1=1, axis2=2) - 1) / 2, -1, 1))
            moving = bool((delta > .002).any() or (future_angle > .02).any())
            # No fabricated program labels. Near-hand moving/static categories.
            rows.append((anchor, t, 1 if moving else 2))
    return np.asarray(rows, dtype=np.int64).reshape(-1, 3)


class NativeWindows(Windows):
    """Read native source frame IDs without changing OakInk2's loader or live training.

    The parent loader has a source-specific 120Hz frame-ID assertion. We validate
    the true timestamps here, then supply its internal four-tick grid in memory.
    Disk frame_ids/source_frame_ids remain the real source identities.
    """
    def __init__(self, root, split):
        super().__init__(root, split)
        if self.meta['schema'] != 'ref2dex.native-wm30.v1':
            raise ValueError('wrong native-source schema')

    def sequence(self, seq):
        data = dict(super().sequence(seq))
        path = self.root / 'processed/sequences' / seq
        timestamps = np.load(path / 'timestamps.npy', mmap_mode='r')
        if not np.allclose(np.diff(timestamps), 1 / 30, atol=1e-5, rtol=0):
            raise ValueError('native clock is not contiguous 30 Hz')
        data['frame_ids'] = 4 * np.arange(len(timestamps), dtype=np.int64)
        return data
