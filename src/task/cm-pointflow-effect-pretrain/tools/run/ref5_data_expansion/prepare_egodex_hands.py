"""Preserve native EgoDex hands/camera/confidences for object-only reconstruction.

This is explicitly HAND_ONLY, never a trainable effect dataset. Missing object
poses/geometry are not invented. Run in an existing read-only h5py environment.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

JOINT_SUFFIXES = ['Hand', 'ThumbKnuckle', 'IndexFingerKnuckle', 'MiddleFingerKnuckle',
                  'RingFingerKnuckle', 'LittleFingerKnuckle', 'ThumbTip', 'IndexFingerTip',
                  'MiddleFingerTip', 'RingFingerTip', 'LittleFingerTip']


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def jsonable(value):
    if isinstance(value, np.ndarray):
        return [jsonable(x) for x in value.tolist()]
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, bytes):
        return value.decode('utf8')
    return value


def native_hands(f):
    names = [[side + suffix for suffix in JOINT_SUFFIXES] for side in ('right', 'left')]
    transforms = np.stack([np.stack([f['transforms'][n][:] for n in side], axis=1)
                           for side in names], axis=1)
    n = len(transforms)
    confidence = np.full((n, 2, 11), np.nan, dtype=np.float32)
    known = np.zeros((n, 2, 11), dtype=bool)
    for side, joint_names in enumerate(names):
        for point, name in enumerate(joint_names):
            if 'confidences' in f and name in f['confidences']:
                confidence[:, side, point] = f['confidences'][name][:]
                known[:, side, point] = True
    valid = np.isfinite(transforms).all((-1, -2))
    valid &= ~known | (np.isfinite(confidence) & (confidence > 0))
    xyz = transforms[..., :3, 3].copy()
    xyz[~valid] = 0
    return xyz, valid, confidence, known, names


def camera_object_to_origin(camera_to_origin, object_to_camera):
    """Known native camera calibration only; never use predicted VGGT world poses.

    EgoDex's official projection uses inv(camera_to_origin) directly with K.
    No extra OpenGL y/z flip is introduced. Estimated depth/mesh scale must
    separately be in meters before calling this composition.
    """
    return camera_to_origin @ object_to_camera


def main():
    import cv2
    import h5py
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(schema='ref2dex.egodex-native-hands.v1', status='RUNNING',
                    dataset_stage='HAND_ONLY_OBJECT_PENDING', fps=30, horizon=24, history=4,
                    split='official_test', training_allowed=False, coordinate_frame='stationary_ARKit_origin',
                    hand_order=['right', 'left'], hand_mapping_note='EgoDex Knuckle is the native joint counterpart; cross-source anatomical equivalence remains approximate',
                    clips=[], object_effect_windows=0, code_sha256=sha(Path(__file__)))
    start = time.monotonic()
    cv2.setNumThreads(2)
    try:
        for path in sorted(a.input.rglob('*.hdf5')):
            with h5py.File(path, 'r') as f:
                hand, valid, confidence, known, names = native_hands(f)
                camera = f['transforms/camera'][:]
                K = f['camera/intrinsic'][:]
                attrs = {k: jsonable(v) for k, v in f.attrs.items()}
            video = path.with_suffix('.mp4')
            cap = cv2.VideoCapture(str(video))
            nvideo = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if not cap.isOpened() or nvideo != len(hand) or abs(fps-30) > 1e-3:
                raise ValueError('annotation/video clock mismatch: ' + str(path))
            if len(camera) != len(hand) or not np.isfinite(camera).all():
                raise ValueError('invalid camera stream')
            if not np.allclose(camera[:, :3, :3].swapaxes(-1, -2) @ camera[:, :3, :3], np.eye(3), atol=1e-3):
                raise ValueError('nonrigid camera stream')
            if np.any(known & ((confidence < 0) | (confidence > 1))):
                raise ValueError('confidence outside [0,1]')
            name = path.parent.name + '_' + path.stem
            dest = a.output/name; dest.mkdir()
            np.savez_compressed(dest/'native_hands.npz', hand=hand, point_valid=valid,
                                confidence=confidence, confidence_known=known,
                                camera_to_origin=camera, intrinsics=K,
                                source_frame_ids=np.arange(len(hand)), timestamps=np.arange(len(hand))/30,
                                joint_names=np.asarray(names))
            # At least one fully observed hand through 4 history +24 future frames.
            hand_valid = valid.all(-1)
            full = np.lib.stride_tricks.sliding_window_view(hand_valid, 28, axis=0).all(-1)
            windows = np.flatnonzero(full.any(-1)) + 3
            np.save(dest/'hand_window_ticks.npy', windows)
            # Small review contact sheet at begin/middle/end, with native reprojections.
            tiles = []
            in_image, observed = 0, 0
            for tick in [0, len(hand)//2, len(hand)-1]:
                cap.set(cv2.CAP_PROP_POS_FRAMES, tick)
                ok, frame = cap.read()
                if not ok:
                    raise ValueError('video decode failed')
                points = (hand[tick].reshape(22, 3) - camera[tick, :3, 3]) @ camera[tick, :3, :3]
                projected = points @ K.T
                positive = points[:, 2] > .01
                xy = projected[:, :2] / np.maximum(projected[:, 2:], 1e-8)
                mask = valid[tick].reshape(22) & positive
                inside = mask & (xy[:, 0] >= 0) & (xy[:, 0] < width) & (xy[:, 1] >= 0) & (xy[:, 1] < height)
                in_image += int(inside.sum()); observed += int(mask.sum())
                for j in np.flatnonzero(inside):
                    cv2.circle(frame, tuple(np.rint(xy[j]).astype(int)), 7,
                               (0, 255, 0) if j < 11 else (255, 120, 0), -1)
                cv2.putText(frame, f'{name} frame {tick}', (25, 45), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 255, 255), 2)
                tiles.append(cv2.resize(frame, (640, 360)))
            cap.release()
            cv2.imwrite(str(dest/'native_projection.jpg'), np.concatenate(tiles, axis=1))
            record = dict(clip=name, frames=len(hand), hand_windows=len(windows),
                          video_shape=[height, width], source_hdf5=str(path.resolve()),
                          source_video=str(video.resolve()), hdf5_sha256=sha(path), video_sha256=sha(video),
                          confidence_known_fraction=float(known.mean()),
                          zero_confidence_count=int((known & (confidence == 0)).sum()),
                          observed_hand_frames=hand_valid.sum(0).tolist(),
                          sampled_projected_inside_fraction=in_image/max(observed, 1), metadata=attrs,
                          object_status='NOT_ESTIMATED', object_6dof_available=False)
            (dest/'meta.json').write_text(json.dumps(record, indent=2)+'\n')
            manifest['clips'].append(record)
            print(json.dumps({k: record[k] for k in ('clip', 'frames', 'hand_windows')}), flush=True)
        manifest['status'] = 'COMPLETED'
        manifest['hand_windows'] = sum(c['hand_windows'] for c in manifest['clips'])
    except Exception as e:
        manifest.update(status='FAILED', error=repr(e))
        raise
    finally:
        manifest['elapsed_s'] = time.monotonic()-start
        (a.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
