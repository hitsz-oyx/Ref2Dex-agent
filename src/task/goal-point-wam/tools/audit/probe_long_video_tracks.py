#!/usr/bin/env python3
"""Full-context tracking feasibility with point births and fixed shared depth."""
import argparse
from functools import partial
import json
from pathlib import Path
import subprocess
import sys
import time

import h5py
import numpy as np
import torch

TASK = Path(__file__).resolve().parents[2]
CM = TASK.parent / 'cm-pointflow-effect-pretrain'
sys.path.insert(0, str(CM / 'src'))
sys.path.insert(0, str(CM / 'tools/run'))
from oakink_wm.pointwam_data import lift_tracks, project_world, sha256
from prepare_pointwam_pilot import decode_clip
from trajectory_geometry import shared_trajectory_depth


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('geometry', 'tracker-vendor', 'tracker-checkpoint', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--query-batch', type=int, choices=(128, 256, 512, 1024), default=256)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    vendor_sha = subprocess.check_output(['git', '-C', str(args.tracker_vendor), 'rev-parse', 'HEAD'], text=True).strip()
    if vendor_sha != '82e02e8029753ad4ef13cf06be7f4fc5facdda4d':
        raise ValueError('unexpected CoTracker source')
    checkpoint_sha = sha256(args.tracker_checkpoint)
    if checkpoint_sha != '2670d4562ed69326dda775a26e54883925cd11b6fc9b24cb7aa9f8078bce7834':
        raise ValueError('unexpected tracker weights')
    metadata = json.loads(args.geometry.with_name('result.json').read_text())
    frames = np.asarray(metadata['frames'])
    video_path = Path(metadata['video'])
    with h5py.File(video_path.with_suffix('.hdf5'), 'r') as raw:
        camera = raw['transforms/camera'][:]
    if sha256(video_path.with_suffix('.hdf5')) != metadata['raw_hdf5_sha256']:
        raise ValueError('native source drift')
    with np.load(args.geometry, allow_pickle=False) as data:
        intrinsic = data['intrinsic'].copy()
    depth, _, provenance = shared_trajectory_depth(args.geometry, frames, intrinsic, camera, video_path)
    if not np.array_equal(frames, np.arange(0, len(camera), 3)) or len(frames) != 402:
        raise ValueError('fixed complete-record clock required')
    args.output.mkdir(parents=True)
    started = time.monotonic()
    torch.set_num_threads(2)

    def progress(stage, **fields):
        event = dict(stage=stage, seconds=time.monotonic() - started, **fields)
        part = args.output / 'progress.part'
        part.write_text(json.dumps(event, allow_nan=False) + '\n')
        part.replace(args.output / 'progress.json')
        print(json.dumps(event), flush=True)

    progress('DECODE')
    rgb, _ = decode_clip(video_path, frames, len(camera))
    tensor_video = torch.from_numpy(rgb).permute(0, 3, 1, 2).float()[None].cuda()
    del rgb
    # Identical current-only pixel grids, born at two declared source times.
    x, y = np.meshgrid(np.linspace(4, 635, 32), np.linspace(4, 355, 16))
    pixels = np.stack((x.ravel(), y.ravel()), -1).astype('float32')
    query_time = np.repeat([0, 200], len(pixels)).astype('int64')
    queries = np.concatenate((query_time[:, None], np.tile(pixels, (2, 1))), -1).astype('float32')
    sys.path.insert(0, str(args.tracker_vendor))
    from cotracker.predictor import CoTrackerPredictor
    predictor = CoTrackerPredictor(checkpoint=str(args.tracker_checkpoint), offline=True, window_len=60).eval().cuda()
    predictor.model.forward = partial(predictor.model.forward, fmaps_chunk_size=32)
    hooks = []
    feature_chunks = [0]

    def after_features(module, inputs, output):
        feature_chunks[0] += 1
        torch.cuda.synchronize()
        progress('FEATURES', chunks=feature_chunks[0], expected_chunks=13*(1024//args.query_batch),
                 allocated_bytes=torch.cuda.memory_allocated(), peak_allocated_bytes=torch.cuda.max_memory_allocated())
    hooks.append(predictor.model.fnet.register_forward_hook(after_features))
    forward_started = time.monotonic()
    progress('TRACKING', frames=len(frames), queries=len(queries), feature_chunk_size=32, query_batch=args.query_batch)
    all_tracks, all_visibility = [], []
    for lo in range(0, len(queries), args.query_batch):
        with torch.inference_mode():
            xy, visible = predictor(tensor_video, queries=torch.from_numpy(queries[lo:lo+args.query_batch])[None].cuda(), backward_tracking=False)
        torch.cuda.synchronize()
        all_tracks.append(xy[0].float().cpu().numpy())
        all_visibility.append(visible[0].cpu().numpy().astype(bool))
        del xy, visible
        progress('QUERY_BATCH', processed_queries=min(lo+args.query_batch,len(queries)),queries=len(queries),
                 forward_seconds=time.monotonic()-forward_started, peak_allocated_bytes=torch.cuda.max_memory_allocated())
    torch.cuda.synchronize()
    forward_seconds = time.monotonic() - forward_started
    tracks = np.concatenate(all_tracks,axis=1)
    visibility = np.concatenate(all_visibility,axis=1)
    peak_allocated = torch.cuda.max_memory_allocated()
    for hook in hooks:
        hook.remove()
    del tensor_video, predictor
    torch.cuda.empty_cache()
    visibility &= np.arange(len(frames))[:, None] >= query_time[None]
    if not np.isfinite(tracks[visibility]).all():
        raise ValueError('nonfinite visible tracks')
    world, valid = lift_tracks(tracks, visibility, depth, intrinsic, camera[frames])
    progress('AUDIT', forward_seconds=forward_seconds, peak_allocated_bytes=peak_allocated)
    coverage = []
    for tick in (0, 10, 50, 100, 199, 200, 210, 250, 300, 350, 401):
        coverage.append(dict(raw_frame=int(frames[tick]),seconds=float(frames[tick]/30),
            groups={str(birth):dict(visible=int(visibility[tick, query_time == birth].sum()),
                valid3d=int(valid[tick, query_time == birth].sum()), queries=int((query_time == birth).sum()))
                for birth in (0, 200)}))
    windows = []
    for tick in (0, 50, 100, 150, 200, 250, 300, 350, 391):
        span = slice(tick, tick + 11)
        initial = np.broadcast_to(world[tick], world[span].shape)
        projected, _ = project_world(initial[:, None], intrinsic, camera[frames[span]])
        residual = np.linalg.norm(projected[:, 0] - tracks[span], axis=-1)
        candidate = valid[span].all(0) & np.isfinite(residual).all(0) & (residual.max(0) <= 2.)
        displacement = np.linalg.norm(world[span] - world[tick], axis=-1)*1000
        windows.append(dict(current_raw_frame=int(frames[tick]),current_valid3d=int(valid[tick].sum()),
            mean_future_valid_fraction=float(valid[span][1:].mean()),candidate_count=int(candidate.sum()),
            future_valid_fraction_among_current=float(valid[span][1:,valid[tick]].mean()) if valid[tick].any() else None,
            maximum_candidate_median_mm=float(np.median(displacement[:,candidate],axis=1).max()) if candidate.any() else None,
            maximum_candidate_p90_mm=float(np.percentile(displacement[:,candidate],90,axis=1).max()) if candidate.any() else None))
    np.savez_compressed(args.output / 'tracks.npz', source_frame_ids=frames, tracks_uv=tracks,
        visibility=visibility, query_time=query_time, query_pixels=np.tile(pixels,(2,1)), world_xyz=world, world_valid=valid)
    report = dict(status='COMPLETED', engineering_only=True, training_allowed=False,
        raw_duration_seconds=metadata['raw_duration_seconds'], frames=len(frames), queries=len(queries),
        query_batch=args.query_batch,
        forward_seconds=forward_seconds, seconds=time.monotonic()-started, peak_allocated_bytes=peak_allocated,
        coverage=coverage, local_window_audit=windows, geometry_provenance=provenance,
        tracker_vendor_sha256=vendor_sha, tracker_checkpoint_sha256=checkpoint_sha,
        tracks_sha256=sha256(args.output/'tracks.npz'), code_sha256=sha256(Path(__file__)),
        limitation='Fixed grid point births; full offline temporal context. Query batches change spatial support context; no exact invariance claimed. Birth positions/visibility are overridden by predictor. Reconstructed low-image-residual strata are not physical static ground truth. No scientific admission.')
    (args.output/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    progress('COMPLETED', forward_seconds=forward_seconds)


if __name__ == '__main__':
    main()
