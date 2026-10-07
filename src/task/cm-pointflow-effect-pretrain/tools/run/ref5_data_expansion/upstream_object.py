"""Adapt EgoDex inputs to the original ObjectForesight step10 entrypoint.

No initialization, scale selection or tracking loop is reimplemented here.
The original process_single_object owns those decisions. Native calibration
and frame identities are retained in a separate sidecar for output conversion.
"""
import argparse
import importlib.util
import json
import os
import signal
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import trimesh

from native_data import sha
from object_geometry import exact_mesh_diameter


class RunDeadlineExceeded(BaseException):
    """Stop the whole run, bypassing upstream recoverable Exception handlers."""


def stage(a):
    a.output.mkdir(parents=True, exist_ok=False)
    d = np.load(a.depth)
    ids = d['source_frame_ids']
    if len(ids) < 28 or not np.all(np.diff(ids) == 1):
        raise ValueError('expected consecutive native 30 Hz frames')
    clean = [int(x) for x in a.clean_frames.split(',')]
    if not clean or not set(clean) <= set(ids.tolist()):
        raise ValueError('clean frame IDs must be within the selected interval')
    seq = a.output / 'clip'
    obj = seq / 'objects' / 'object_0'
    (obj / 'trellis').mkdir(parents=True)
    mesh = trimesh.load(a.mesh, force='mesh', process=False)
    mesh.export(obj / 'trellis' / 'model.glb')
    # Resize spatially only; do not resample timestamps or substitute cameras.
    cap = cv2.VideoCapture(str(a.video))
    if abs(cap.get(cv2.CAP_PROP_FPS) - 30) > .01:
        raise ValueError('source video is not native 30 Hz')
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(ids[0]))
    writer = cv2.VideoWriter(str(seq / 'action.mp4'),
                            cv2.VideoWriter_fourcc(*'mp4v'), 30, (a.width, a.height))
    if not writer.isOpened():
        raise RuntimeError('staged video encoder unavailable')
    masks = np.load(a.masks)['masks']
    all_masks, clean_masks = {}, {}
    try:
        for local, native in enumerate(ids):
            ok, image = cap.read()
            if not ok:
                raise ValueError(f'source decode failed at {native}')
            writer.write(cv2.resize(image, (a.width, a.height)))
            mask = cv2.resize(masks[native].astype('uint8'), (a.width, a.height),
                              interpolation=cv2.INTER_NEAREST)
            all_masks[str(local)] = mask
            if int(native) in clean:
                clean_masks[str(local)] = mask
    finally:
        cap.release()
        writer.release()
    # The upstream reader scales K from the depth resolution to the RGB size.
    np.savez_compressed(seq / 'spatracker.npz', depths=d['depth'],
                        intrinsics=d['intrinsics'])
    np.savez_compressed(seq / 'native_calibration.npz', source_frame_ids=ids,
                        timestamps=d['timestamps'], camera_to_origin=d['camera_to_origin'],
                        intrinsics=d['intrinsics'])
    np.savez_compressed(obj / 'vas_masks.npz', **all_masks)
    np.savez_compressed(obj / 'vas_clean_masks.npz', **clean_masks)
    report = dict(status='STAGED', training_allowed=False, frames=len(ids), fps=30,
                  clean_source_frames=clean, native_camera_preserved=True,
                  image_size=[a.height, a.width],
                  inputs={str(p.resolve()): sha(p) for p in
                          (a.video, a.hdf5, a.mesh, a.masks, a.depth)},
                  script_sha256=sha(Path(__file__)),
                  note='Existing masks/mesh/native-calibrated depth; original step10 core.')
    (a.output / 'adapter_manifest.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


def run(a):
    if not (a.output/'adapter_manifest.json').exists():
        raise FileNotFoundError('stage the clip before running original step10')
    if (a.output/'upstream_run_manifest.json').exists():
        raise FileExistsError('use a fresh run directory; retain existing results')
    import torch
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    vendor = a.root.resolve() / 'vendor' / 'ObjectForesight-Data'
    sys.path.insert(0, str(vendor))
    sys.path.insert(0, str(vendor / 'FoundationPose'))
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    os.environ['EGL_DEVICE_ID'] = a.egl_device
    import estimater
    # Exact alternative to the estimator's sampled diameter calculation;
    # avoids its 10000x10000 random point distance allocation. This may differ
    # slightly from the sampled diameter; record the substitution explicitly.
    estimater.compute_mesh_diameter = exact_mesh_diameter
    source = vendor / 'step10_fpose.py'
    spec = importlib.util.spec_from_file_location('objectforesight_step10', source)
    upstream = importlib.util.module_from_spec(spec)
    source_text = source.read_text()
    # Pinned upstream has one extra NaN argument in the budget-exhausted log.
    # Fix only that logging call, retaining the original file and failed run.
    old = '*[np.nan] * 10,'
    if source_text.count(old) != 1:
        raise ValueError('pinned upstream budget-log patch no longer matches')
    patched_text = source_text.replace(old, '*[np.nan] * 9,')
    old_video = 'frame = rendered_frames.get(frame_idx, reader.get_color(frame_idx))'
    new_video = ('frame = rendered_frames.get(frame_idx, compose_with_mask_panel('
                 'reader.get_color(frame_idx), np.zeros((reader.H, reader.W), dtype=bool)))')
    if patched_text.count(old_video) != 1:
        raise ValueError('pinned upstream video-size patch no longer matches')
    patched_text = patched_text.replace(old_video,new_video)
    exec(compile(patched_text, str(source), 'exec'), upstream.__dict__)
    from upstream_runtime import install_native_resolution_batching
    install_native_resolution_batching()
    # Stock step10 assumes Hopper. This machine's RTX3090 is sm86.
    os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6'
    old_argv = sys.argv
    try:
        sys.argv = [str(source), '--debug', '0']
        cfg = upstream.load_config()
    finally:
        sys.argv = old_argv
    # Capture final scaled geometry for native output, preserving the original
    # function's arguments, return values and all scale/pose choices.
    original_refine = upstream.refine_scale_with_silhouette
    def capture_mesh(*args, **kwargs):
        result = original_refine(*args, **kwargs)
        result[0].export(a.output / 'mesh_metric.ply')
        return result
    upstream.refine_scale_with_silhouette = capture_mesh
    initialization_errors=[]
    original_candidate=upstream.evaluate_init_candidate
    def record_candidate(*args,**kwargs):
        try:
            return original_candidate(*args,**kwargs)
        except Exception as e:
            initialization_errors.append(repr(e))
            raise
    upstream.evaluate_init_candidate=record_candidate
    started = time.monotonic()
    report = dict(status='RUNNING', pid=os.getpid(), training_allowed=False,
                  upstream_sha256=sha(source), script_sha256=sha(Path(__file__)),
                  runtime_adapter_sha256={name:sha(Path(__file__).with_name(name))
                                           for name in ['upstream_runtime.py','object_geometry.py']},
                  config=vars(cfg), seconds_budget=a.seconds,
                  compatibility_changes=['local imports', 'sm86 build architecture',
                                         'exact bounded estimator diameter',
                                         'capture final mesh for native schema',
                                         'PyOpenGL3.1.7 / system GLdispatch preload',
                                         'budget-exhausted log: 10 NaNs -> 9',
                                         'debug video fallback: same-size blank mask panel',
                                         'original score input transform in32-pose batches'],
                  library_preload=os.environ.get('LD_PRELOAD'),
                  core='original process_single_object / choose_best_init / Tracker')
    manifest = a.output / 'upstream_run_manifest.json'
    manifest.write_text(json.dumps(report, indent=2)+'\n')
    def deadline(signum, frame):
        raise RunDeadlineExceeded('original ObjectForesight bounded run deadline')
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(a.seconds)
    try:
        seq = a.output / 'clip'
        upstream.process_single_object(seq / 'objects' / 'object_0',
                                       str(seq / 'action.mp4'), cfg)
        summary = seq / 'objects' / 'object_0' / 'foundationpose10' / 'run_summary.json'
        report['status'] = ('COMPLETED' if summary.exists() else
                            'INITIALIZATION_ERROR' if initialization_errors else 'NO_INITIALIZATION')
        if summary.exists():
            report['summary'] = json.loads(summary.read_text())
    except BaseException as e:
        report.update(status='TIMED_OUT' if isinstance(e,RunDeadlineExceeded) else 'FAILED',
                      error=repr(e))
        raise
    finally:
        signal.alarm(0)
        report['initialization_errors']=initialization_errors
        report['elapsed_s'] = time.monotonic() - started
        manifest.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['stage', 'run'])
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    for name in ['video', 'hdf5', 'mesh', 'masks', 'depth']:
        p.add_argument('--'+name, type=Path)
    p.add_argument('--clean-frames')
    # Stock morphology/area thresholds are in pixels. Preserve native RGB
    # resolution by default so input adaptation does not amplify those gates.
    p.add_argument('--width', type=int, default=1920)
    p.add_argument('--height', type=int, default=1080)
    p.add_argument('--seconds', type=int, default=900)
    p.add_argument('--egl-device', default='3')
    a = p.parse_args()
    a.output = a.output.resolve()
    if a.mode == 'stage':
        if any(getattr(a, k) is None for k in
               ['video', 'hdf5', 'mesh', 'masks', 'depth', 'clean_frames']):
            p.error('stage requires video/hdf5/mesh/masks/depth/clean-frames')
        if not 28 <= a.width <= 1920 or not 28 <= a.height <= 1080:
            p.error('invalid spatial size')
        stage(a)
    else:
        if not 1 <= a.seconds <= 1800:
            p.error('run budget must be 1..1800 seconds')
        run(a)


if __name__ == '__main__':
    main()
