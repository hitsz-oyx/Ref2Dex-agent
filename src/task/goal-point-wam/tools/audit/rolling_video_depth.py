"""Generate labels with the same observed-four-frame frontend as current input.

Each label frame may use its own past, including frames after the actor's
current time. Those predictions are labels only. No teacher/current offset is
subtracted and no predicted camera replaces the native metric camera.
"""
import sys
import time
from pathlib import Path

import numpy as np


def rolling_prefix_ids(frame_ids):
    frames = np.asarray(frame_ids)
    if frames.ndim != 1 or not len(frames) or frames.dtype.kind not in 'iu':
        raise ValueError('integer frame ids required')
    if frames[0] < 3 or (np.diff(frames) <= 0).any():
        raise ValueError('four observed frames and increasing labels required')
    return frames[:, None] + np.arange(-3, 1)[None]


def rolling_metric_depth(rgb, decoded_frame_ids, label_frame_ids, hand, valid,
                         intrinsic, camera, vendor, checkpoint, destination):
    import cv2
    import torch
    from oakink_wm.pointwam_data import recover_metric_scale, resize_valid_depth

    sys.path.insert(0, str(vendor))
    from vggt_omega.models import VGGTOmega
    from vggt_omega.utils.load_fn import load_and_preprocess_images

    if not torch.cuda.is_available():
        raise RuntimeError('rolling depth requires GPU')
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    ids = rolling_prefix_ids(label_frame_ids)
    if len(rgb) != len(decoded_frame_ids) or not np.isin(ids, decoded_frame_ids).all():
        raise ValueError('missing decoded observation frame')
    images_root = Path(destination)/'omega_input'
    images_root.mkdir()
    paths = {}
    for frame_id, image in zip(decoded_frame_ids, rgb):
        path = images_root/f'f{int(frame_id):06d}.png'
        cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        paths[int(frame_id)] = str(path)
    model = VGGTOmega().eval()
    weights = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(weights, strict=True)
    del weights
    model.cuda()
    depth, calibrations = [], []
    started = time.monotonic()
    for row in ids:
        images = load_and_preprocess_images([paths[int(t)] for t in row], image_resolution=512).cuda()
        with torch.inference_mode():
            prediction = model(images)
        raw = prediction['depth'].float().cpu().numpy()
        if raw.shape[:2] != (1, 4) or raw.shape[-1] != 1:
            raise ValueError('unexpected four-frame Omega depth')
        prefix_depth = resize_valid_depth(raw[0, ..., 0], (640, 360))
        calibration = recover_metric_scale(prefix_depth, hand[row], valid[row], intrinsic, camera[row])
        depth.append(prefix_depth[-1]*calibration['scale'])
        calibrations.append(dict(frame_ids=row.tolist(), **calibration))
        print(__import__('json').dumps(dict(stage='ROLLING_DEPTH', label_frame=int(row[-1]),
              done=len(depth), total=len(ids), elapsed_seconds=time.monotonic()-started,
              eta_seconds=(time.monotonic()-started)/len(depth)*(len(ids)-len(depth)),
              allocated_mib=torch.cuda.memory_allocated()/2**20,
              peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)), flush=True)
        del images, prediction
    del model
    torch.cuda.empty_cache()
    return np.stack(depth), calibrations
