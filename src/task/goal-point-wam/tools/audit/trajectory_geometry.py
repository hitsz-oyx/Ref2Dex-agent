"""Read a fixed, once-calibrated offline geometry span without recomputation."""
import hashlib
import json
from pathlib import Path

import numpy as np


def shared_trajectory_depth(path, frames, intrinsic, camera, source_video, cache=None):
    path = Path(path)
    metadata_path = path.with_name('result.json')
    key = str(path.resolve())
    cached = cache.get(key) if cache is not None else None
    metadata = cached[0] if cached else json.loads(metadata_path.read_text())
    if (metadata.get('status') != 'COMPLETED'
            or metadata.get('engineering_only') is not True
            or metadata.get('training_allowed') is not False):
        raise ValueError('explicit engineering geometry required')
    if Path(metadata['video']).resolve() != Path(source_video).resolve():
        raise ValueError('trajectory geometry source mismatch')
    digest = cached[1] if cached else hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != metadata['output_sha256']:
        raise ValueError('trajectory geometry identity drift')
    if cached:
        data = cached[2]
    else:
        with np.load(path, allow_pickle=False) as loaded:
            data = {k:loaded[k].copy() for k in loaded.files}
        if cache is not None:
            cache.clear()  # Only one owned immutable recording is resident.
            cache[key] = (metadata, digest, data)
    ids = data['source_frame_ids']
    if (ids.ndim != 1 or ids.dtype.kind not in 'iu' or not len(ids)
            or ids[0] < 0 or (np.diff(ids) <= 0).any()
            or ids.tolist() != metadata['frames']):
        raise ValueError('invalid trajectory geometry clock')
    frames = np.asarray(frames)
    offsets = np.searchsorted(ids, frames)
    if (offsets >= len(ids)).any() or not np.array_equal(ids[offsets], frames):
        raise ValueError('trajectory geometry does not cover requested frames')
    if (not np.allclose(data['intrinsic'], intrinsic, atol=1e-4)
            or not np.allclose(data['camera_to_origin'][offsets], camera[frames], atol=1e-5)):
        raise ValueError('trajectory geometry camera mismatch')
    depth = data['depth_m'][offsets].copy()
    if depth.shape != (len(frames), 360, 640) or not np.isfinite(depth).all() or (depth <= 0).any():
        raise ValueError('invalid shared metric depth')
    calibration = metadata['calibration']
    if not np.isfinite(calibration['scale']) or calibration['scale'] <= 0:
        raise ValueError('invalid shared scale')
    provenance = dict(trajectory_depth_path=str(path.resolve()), trajectory_depth_sha256=digest,
                      trajectory_geometry_manifest_path=str(metadata_path.resolve()),
                      trajectory_geometry_manifest_sha256=hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
                      depth_context_frame_ids=ids.tolist())
    return depth, calibration, provenance
