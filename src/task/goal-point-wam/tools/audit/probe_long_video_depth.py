#!/usr/bin/env python3
"""Measure one whole-record offline reconstruction; never admit it to training."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch

TASK = Path(__file__).resolve().parents[2]
CM = TASK.parent / 'cm-pointflow-effect-pretrain'
sys.path.insert(0, str(CM / 'src'))
sys.path.insert(0, str(CM / 'tools/run'))
from oakink_wm.pointwam_data import (project_world, recover_metric_scale,
    resize_intrinsics, resize_valid_depth, sample_depth, sha256, source_identity)
from prepare_pointwam_pilot import decode_clip
from prepare_egodex_actor import JOINTS

_MODEL_CACHE = {}
_CHECKPOINT_CACHE = {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--member', default='part2/fold_stack_unstack_unfold_cloths/1.hdf5')
    parser.add_argument('--max-frames', type=int, default=450)
    parser.add_argument('--seconds', type=int, default=1200)
    parser.add_argument('--reuse-model', action='store_true', help='Reuse pinned weights within a persistent full-corpus worker')
    parser.add_argument('--lossless-compressed', action='store_true')
    parser.add_argument('--discard-decoded-images', action='store_true', help='Remove only PNG scratch created by this invocation after a completed cache')
    args = parser.parse_args()
    if args.output.exists() or not 2 <= args.max_frames <= 500 or not 30 <= args.seconds <= 1200:
        raise ValueError('new bounded output/frame/deadline required')
    plan = json.loads(args.plan.read_text())
    records = [r for r in plan['records'] if r['member'] == args.member]
    if len(records) != 1:
        raise ValueError('one existing original episode required')
    record = records[0]
    source = Path(record['raw_hdf5'])
    video = source.with_suffix('.mp4')
    vendor = Path(plan['depth_vendor'])
    checkpoint = Path(plan['depth_checkpoint'])
    vendor_sha = subprocess.check_output(['git', '-C', str(vendor), 'rev-parse', 'HEAD'], text=True).strip()
    if vendor_sha != '48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6':
        raise ValueError('unexpected vendor source')
    checkpoint_key = str(checkpoint.resolve())
    checkpoint_sha = _CHECKPOINT_CACHE.get(checkpoint_key) if args.reuse_model else None
    if checkpoint_sha is None:
        checkpoint_sha = sha256(checkpoint)
    if checkpoint_sha != 'c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934':
        raise ValueError('unexpected depth checkpoint')
    if args.reuse_model: _CHECKPOINT_CACHE[checkpoint_key] = checkpoint_sha
    identity = source_identity(source, video, Path(record['source_manifest']))
    if identity['split'] != 'official_train' or not torch.cuda.is_available():
        raise ValueError('verified TRAIN acquisition and GPU required')
    args.output.mkdir(parents=True)
    started = time.monotonic()
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    progress_path = args.output / 'progress.json'

    def progress(stage, **fields):
        elapsed = time.monotonic() - started
        if elapsed > args.seconds:
            raise TimeoutError('whole-record reconstruction deadline')
        event = dict(stage=stage, seconds=elapsed, **fields)
        partial = progress_path.with_suffix('.part')
        partial.write_text(json.dumps(event, allow_nan=False) + '\n')
        partial.replace(progress_path)
        print(json.dumps(event), flush=True)

    with h5py.File(source, 'r') as raw:
        camera = raw['transforms/camera'][:]
        intrinsic = raw['camera/intrinsic'][:]
        names = [[side + joint for joint in JOINTS] for side in ('right', 'left')]
        hand = np.stack([np.stack([raw['transforms'][j][:, :3, 3] for j in side], 1) for side in names], 1)
        valid = np.stack([np.stack([raw['confidences'][j][:] >= .5 for j in side], 1) for side in names], 1)
    valid &= np.isfinite(hand).all(-1)
    frames = np.arange(0, len(camera), 3)
    if not 2 <= len(frames) <= args.max_frames:
        raise ValueError('whole-record context exceeds declared frame budget')
    progress('DECODE', sampled_frames=len(frames), raw_frames=len(camera))
    rgb, original_wh = decode_clip(video, frames, len(camera))
    intrinsic = resize_intrinsics(intrinsic, original_wh, (640, 360))
    image_root = args.output / 'images'
    image_root.mkdir()
    paths = []
    for frame, image in zip(frames, rgb):
        path = image_root / ('f%06d.png' % frame)
        if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
            raise IOError('PNG write failed')
        paths.append(str(path))
    del rgb
    sys.path.insert(0, str(vendor))
    from vggt_omega.models import VGGTOmega
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    progress('MODEL_LOAD')
    model = _MODEL_CACHE.get(checkpoint_key) if args.reuse_model else None
    if model is None:
        model = VGGTOmega().eval()
        weights = torch.load(checkpoint, map_location='cpu', weights_only=True)
        model.load_state_dict(weights, strict=True)
        del weights
        model.cuda()
        if args.reuse_model: _MODEL_CACHE[checkpoint_key] = model
    images = load_and_preprocess_images(paths, image_resolution=512).cuda()
    input_shape = list(images.shape)
    torch.cuda.synchronize()
    inference_started = time.monotonic()
    hooks = []
    for index, block in enumerate(model.aggregator.inter_frame_blocks):
        def after_block(module, inputs, output, index=index):
            torch.cuda.synchronize()
            forward_seconds = time.monotonic() - inference_started
            progress('AGGREGATOR', block=index + 1, blocks=24,
                     forward_seconds=forward_seconds,
                     estimated_remaining_aggregator_seconds=forward_seconds / (index + 1) * (23 - index),
                     allocated_bytes=torch.cuda.memory_allocated(),
                     reserved_bytes=torch.cuda.memory_reserved(),
                     peak_allocated_bytes=torch.cuda.max_memory_allocated())
        hooks.append(block.register_forward_hook(after_block))
    progress('FORWARD', input_shape=list(images.shape), allocated_bytes=torch.cuda.memory_allocated())
    with torch.inference_mode():
        prediction = model(images)
    torch.cuda.synchronize()
    forward_seconds = time.monotonic() - inference_started
    peak_allocated = torch.cuda.max_memory_allocated()
    raw_depth = prediction['depth'].float().cpu().numpy()
    if raw_depth.shape[:2] != (1, len(frames)) or raw_depth.shape[-1] != 1:
        raise ValueError('whole-record depth clock mismatch')
    confidence = prediction['depth_conf'].float().cpu().numpy()
    confidence_percentiles = np.percentile(confidence.reshape(len(frames), -1)[:, ::64], [10, 50, 90]).tolist()
    peak_allocated = torch.cuda.max_memory_allocated()
    for hook in hooks:
        hook.remove()
    del images, model, prediction, confidence
    torch.cuda.empty_cache()
    depth = resize_valid_depth(raw_depth[0, ..., 0], (640, 360))
    del raw_depth
    if not np.isfinite(depth).all() or (depth <= 0).any():
        raise ValueError('nonfinite/nonpositive whole-record depth')
    progress('GLOBAL_CALIBRATION', forward_seconds=forward_seconds, peak_allocated_bytes=peak_allocated)
    h, hv, c = hand[frames], valid[frames], camera[frames]
    calibration = recover_metric_scale(depth, h, hv, intrinsic, c)
    uv, z = project_world(h, intrinsic, c)
    sampled, depth_valid = sample_depth(depth, uv)
    keep = hv & depth_valid & np.isfinite(z) & (z > 0)
    segments = []
    for indices in np.array_split(np.arange(len(frames)), 10):
        mask = keep[indices]
        ratios = z[indices][mask] / sampled[indices][mask]
        ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
        segments.append(dict(first_frame=int(frames[indices[0]]), last_frame=int(frames[indices[-1]]),
            samples=len(ratios), median_scale=float(np.median(ratios)) if len(ratios) else None,
            scale_relative_to_global=float(np.median(ratios) / calibration['scale']) if len(ratios) else None))
    depth *= calibration['scale']
    progress('SAVE', calibration=calibration)
    saver = np.savez_compressed if args.lossless_compressed else np.savez
    saver(args.output / 'depth.npz', source_frame_ids=frames, depth_m=depth,
             intrinsic=intrinsic, camera_to_origin=c)
    elapsed = time.monotonic() - started
    report = dict(status='COMPLETED', engineering_only=True, training_allowed=False,
        schema='goal-point-wam.trajectory-depth-feasibility.v1', frames=frames.tolist(),
        raw_frames=len(camera), sampled_fps=10, source_fps=30, raw_duration_seconds=len(camera) / 30,
        calibration=calibration, calibration_anchor='native2x11 named joints, confidence>=.5; no synthetic palm offsets',
        segment_scale_audit=segments, calibration_screen_pass=calibration['relative_iqr'] <= .35,
        video=str(video), source_identity=identity, raw_hdf5_sha256=sha256(source),
        depth_checkpoint_sha256=checkpoint_sha, depth_vendor_sha256=vendor_sha,
        output_sha256=sha256(args.output / 'depth.npz'), code_sha256=sha256(Path(__file__)),
        seconds=elapsed, forward_seconds=forward_seconds, peak_allocated_bytes=peak_allocated,
        input_shape=input_shape, confidence_sample_percentiles=confidence_percentiles,
        source_split=record['split'], source_session=record['session'],
        limitation='Complete recorded duration at10Hz, not30Hz measured depth. Offline future context; approximate hand-to-surface scale. No long point tracking or scientific admission/training.')
    (args.output / 'result.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    if args.discard_decoded_images:
        for path in paths: Path(path).unlink()
        image_root.rmdir()
    progress('COMPLETED', forward_seconds=forward_seconds, total_seconds=elapsed)


if __name__ == '__main__':
    main()
