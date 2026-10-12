#!/usr/bin/env python3
"""Restart short-clip queries while reusing immutable whole-record geometry."""
import argparse
from functools import partial
import importlib.util
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
sys.path.insert(0, str(CM/'src'))
sys.path.insert(0, str(CM/'tools/run'))
from oakink_wm.pointwam_data import lift_tracks, resample_valid, select_scene, sha256
from prepare_pointwam_pilot import decode_clip
from prepare_egodex_actor import JOINTS
from trajectory_geometry import shared_trajectory_depth
from video_clock import cached_window_clock

_TRACKER_CACHE = {}
_TRACKER_IDENTITIES = {}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('geometry', 'tracker-vendor', 'tracker-checkpoint', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--baseline-tracks', type=Path, help='Optional frozen whole-record tracking comparison')
    p.add_argument('--starts', type=int, nargs='+', default=[0, 150, 300, 450, 600, 900, 1050, 1170])
    p.add_argument('--all-windows', action='store_true', help='0.5s stride with a complete tail; at most80 clips')
    p.add_argument('--max-queries', type=int, choices=(1024, 2048, 4096), default=2048)
    p.add_argument('--seconds', type=int, default=600)
    p.add_argument('--max-windows', type=int, default=80)
    p.add_argument('--compact-teachers', action='store_true', help='Reference the identical shared dense geometry instead of copying it into every teacher')
    p.add_argument('--reuse-model', action='store_true')
    a = p.parse_args()
    if a.output.exists() or not 1 <= len(a.starts) <= 12 or len(set(a.starts)) != len(a.starts):
        raise ValueError('new bounded output and unique clip starts required')
    if not 1 <= a.seconds <= 900 or not 1 <= a.max_windows <= 100 or not torch.cuda.is_available():
        raise ValueError('bounded CUDA inference required')
    vendor_sha = subprocess.check_output(['git', '-C', str(a.tracker_vendor), 'rev-parse', 'HEAD'], text=True).strip()
    if vendor_sha != '82e02e8029753ad4ef13cf06be7f4fc5facdda4d':
        raise ValueError('tracker vendor drift')
    tracker_key = str(a.tracker_checkpoint.resolve())
    checkpoint_sha = _TRACKER_IDENTITIES.get(tracker_key) if a.reuse_model else None
    if checkpoint_sha is None: checkpoint_sha = sha256(a.tracker_checkpoint)
    if checkpoint_sha != '2670d4562ed69326dda775a26e54883925cd11b6fc9b24cb7aa9f8078bce7834':
        raise ValueError('tracker weights drift')
    if a.reuse_model: _TRACKER_IDENTITIES[tracker_key] = checkpoint_sha
    meta = json.loads(a.geometry.with_name('result.json').read_text())
    video = Path(meta['video']); raw = video.with_suffix('.hdf5')
    if sha256(raw) != meta['raw_hdf5_sha256']:
        raise ValueError('native annotations drift')
    with h5py.File(raw, 'r') as f:
        camera = f['transforms/camera'][:]
        hand = np.stack([np.stack([f['transforms'][side+j][:, :3, 3] for j in JOINTS], 1) for side in ('right', 'left')], 1)
        hv = np.stack([np.stack([f['confidences'][side+j][:] >= .5 for j in JOINTS], 1) for side in ('right', 'left')], 1)
        session = str(f.attrs['session_name'])
    hv &= np.isfinite(hand).all(-1)
    frames = np.asarray(meta['frames'])
    if a.all_windows:
        # Keep global source IDs and shift the final window to cover the tail.
        a.starts = list(range(int(frames[0]), int(frames[-1])-29, 15))
        tail = int(frames[-1])-30
        if not a.starts or a.starts[-1] != tail:
            a.starts.append(tail)
        if len(a.starts) > a.max_windows:
            raise ValueError('full-record extraction exceeds declared window budget')
    with np.load(a.geometry, allow_pickle=False) as f:
        intrinsic = f['intrinsic'].copy()
    depth, calibration, provenance = shared_trajectory_depth(a.geometry, frames, intrinsic, camera, video)
    baseline, bmeta = None, None
    if a.baseline_tracks is not None:
        bmeta = json.loads(a.baseline_tracks.with_name('result.json').read_text())
        if sha256(a.baseline_tracks) != bmeta['tracks_sha256'] or bmeta['geometry_provenance']['trajectory_depth_sha256'] != provenance['trajectory_depth_sha256']:
            raise ValueError('baseline identity drift')
        with np.load(a.baseline_tracks, allow_pickle=False) as f:
            baseline = {key: f[key].copy() for key in ('source_frame_ids', 'world_xyz', 'world_valid')}
        if not np.array_equal(baseline['source_frame_ids'], frames):
            raise ValueError('baseline source clock mismatch')
    spans = []
    for start in a.starts:
        ids = np.arange(start, start+31, 3)
        current, offset, history = cached_window_clock(ids, 3)
        k = np.searchsorted(frames, ids)
        if (k >= len(frames)).any() or not np.array_equal(frames[k], ids):
            raise ValueError('clip outside shared geometry')
        spans.append((ids, current, offset, history, k))
    a.output.mkdir(); started = time.monotonic(); torch.set_num_threads(2)

    def progress(stage, **fields):
        event = dict(stage=stage, elapsed_seconds=time.monotonic()-started, **fields)
        path = a.output/'progress.part'; path.write_text(json.dumps(event, allow_nan=False)+'\n'); path.replace(a.output/'progress.json')
        print(json.dumps(event), flush=True)
        if time.monotonic()-started > a.seconds:
            raise TimeoutError('window query restart deadline')

    progress('DECODE')
    rgb, _ = decode_clip(video, frames, len(camera))
    sys.path.insert(0, str(a.tracker_vendor))
    from cotracker.predictor import CoTrackerPredictor
    predictor = _TRACKER_CACHE.get(tracker_key) if a.reuse_model else None
    if predictor is None:
        predictor = CoTrackerPredictor(checkpoint=str(a.tracker_checkpoint), offline=True, window_len=60).eval().cuda()
        predictor.model.forward = partial(predictor.model.forward, fmaps_chunk_size=32)
        if a.reuse_model: _TRACKER_CACHE[tracker_key] = predictor
    yy, xx = np.mgrid[0:360:2, 0:640:2]
    dense = np.stack((xx.ravel(), yy.ravel()), -1).astype('float32')
    records = []; summaries = []; forward_seconds = 0.
    for ids, current, offset, history, k in spans:
        folder = a.output/('f%06d'%ids[0]); folder.mkdir()
        first, first_valid = lift_tracks(dense[None], np.ones((1, len(dense)), bool), depth[k[:1]], intrinsic, camera[ids[:1]])
        # Budget adaptation: choose initial geometry BEFORE tracking, never future visibility.
        grid_ids = select_scene(first, first_valid, a.max_queries)
        pixels = dense[grid_ids]; queries = np.column_stack((np.zeros(len(pixels), 'float32'), pixels))
        tensor = torch.from_numpy(rgb[k]).permute(0, 3, 1, 2).float()[None].cuda()
        xy_all = []; visible_all = []; clip_forward = time.monotonic()
        for lo in range(0, len(queries), 256):
            with torch.inference_mode():
                xy, visible = predictor(tensor, queries=torch.from_numpy(queries[lo:lo+256])[None].cuda(), backward_tracking=False)
            torch.cuda.synchronize()
            xy_all.append(xy[0].float().cpu().numpy()); visible_all.append(visible[0].cpu().numpy().astype(bool))
            del xy, visible
            progress('QUERY_BATCH', raw_start=int(ids[0]), processed_queries=min(lo+256,len(queries)), queries=len(queries), peak_allocated_bytes=torch.cuda.max_memory_allocated())
        elapsed = time.monotonic()-clip_forward; forward_seconds += elapsed
        uv = np.concatenate(xy_all, 1); visible = np.concatenate(visible_all, 1)
        del tensor
        world, valid = lift_tracks(uv, visible, depth[k], intrinsic, camera[ids])
        C = np.linalg.inv(camera[current])
        def anchor(x):
            return (x@C[:3,:3].T+C[:3,3]).astype('float32')
        observed = anchor(world[offset:offset+1]); good = valid[offset:offset+1] & (np.abs(observed)<=1.99).all(-1)
        chosen = select_scene(observed, good, 2048)
        measured, known = resample_valid(world[offset:, chosen], valid[offset:, chosen], 3*(len(ids)-offset-1)+1)
        query = observed[0, chosen]; error = np.linalg.norm(anchor(measured[0])-query, axis=-1)
        if error.max() != 0:
            raise ValueError('shared frame position mismatch')
        sample = dict(query_xyz=query, query_valid=good[0, chosen], track_ids=chosen.astype('int64'),
            point_target=anchor(measured)[1:25], point_target_valid=known[1:25]&known[0][None],
            hand=anchor(hand[current-3:current+25]), hand_valid=hv[current-3:current+25],
            start_object_points=np.zeros((32,3),'float32'), goal_object_points=np.zeros((32,3),'float32'), object_goal_valid=np.zeros(32,bool))
        clip = folder/'clip.npz'; teacher = folder/'teacher.npz'
        np.savez_compressed(clip, **sample)
        dense_teacher = {} if a.compact_teachers else dict(depth_m=depth[k])
        np.savez_compressed(teacher, **dense_teacher, source_frame_ids=ids, tracks_uv=uv, visibility=visible, intrinsic=intrinsic,
            camera_to_origin=camera[ids], query_pixels=pixels, query_grid_ids=grid_ids)
        records.append(dict(path=str(clip.relative_to(a.output)), sha256=sha256(clip), split=meta['source_split'], session=session,
            source_split='official_train', source_identity=meta['source_identity'], raw_hdf5_sha256=meta['raw_hdf5_sha256'],
            teacher_path=str(teacher.resolve()), teacher_sha256=sha256(teacher), teacher_frame_ids=ids[offset:].tolist(),
            source_clip_frame_ids=ids.tolist(), source_current_frame=current, observation_frame_ids=history.tolist(), window_offset=3,
            teacher_depth_storage='shared_lossless_cache' if a.compact_teachers else 'embedded', **provenance))
        old_count = None
        if baseline is not None:
            old = anchor(baseline['world_xyz'][k[offset]:k[offset]+1]); old_valid = baseline['world_valid'][k[offset]:k[offset]+1] & (np.abs(old)<=1.99).all(-1)
            old_count = len(select_scene(old, old_valid, 2048))
        summaries.append(dict(raw_current=current, seconds=current/30, tracked_queries=len(pixels), baseline_selected_points=old_count,
            selected_points=len(chosen), future_valid_fraction=float(sample['point_target_valid'].mean()), current_alignment_max_mm=float(error.max()*1000), forward_seconds=elapsed))
        progress('WINDOW_COMPLETED', **summaries[-1])
    peak = torch.cuda.max_memory_allocated(); del predictor; torch.cuda.empty_cache()
    manifest = dict(schema='goal-point-wam.video-points.v1', status='COMPLETED', engineering_only=True, training_allowed=False,
        causal_observation=False, observation_policy='offline_clip_reconstruction', offline_pretraining_only=True,
        label_depth_backend='offline-teacher', fps=30, history=4, horizon=24, records=records, calibration=calibration,
        label_calibrations=[calibration], depth_context_frame_ids=frames.tolist(), geometry_context='whole_recording',
        query_policy='Every-second-pixel candidate grid; initial-only1.5cm voxel subsampling before tracking, max2048; 256-query batches; fresh local IDs each1s clip',
        limitation='Budget adaptation, not exact author pipeline: pre-tracking subsampling changes spatial support. Offline full-record depth; 10Hz labels interpolated30Hz. Birth forced by tracker; no physical accuracy claim.')
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    progress('QUALITY_AUDIT')
    subprocess.run([sys.executable, str(Path(__file__).with_name('audit_video_tracks.py')), '--data', str(a.output), '--output', str(a.output/'quality.json')], check=True, timeout=90)
    spec = importlib.util.spec_from_file_location('reference_video_data',TASK/'src/goal_point_wam/video_data.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    loader = module.VideoPointWindows(a.output, meta['source_split'], engineering=True)
    for i in range(len(loader)):
        batch = module.video_collate([loader[i]])
        if batch['point_target'].shape[1] != 24 or batch['action'].shape[1] != 24:
            raise ValueError('native H4/K24 loader contract')
    quality = json.loads((a.output/'quality.json').read_text())
    report = dict(status='COMPLETED', engineering_only=True, training_allowed=False, windows=summaries,
        admission_pass=sum(r['admission_screen_pass'] for r in quality['records']), loader_windows=len(loader),
        geometry_provenance=provenance, baseline_tracks_sha256=bmeta['tracks_sha256'] if bmeta else None, tracker_vendor_sha256=vendor_sha,
        tracker_checkpoint_sha256=checkpoint_sha, script_sha256=sha256(Path(__file__)), audit_sha256=sha256(Path(__file__).with_name('audit_video_tracks.py')),
        manifest_sha256=sha256(a.output/'manifest.json'), forward_seconds=forward_seconds, seconds=time.monotonic()-started,
        peak_allocated_bytes=peak, limitation=manifest['limitation'])
    (a.output/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    progress('COMPLETED', admission_pass=report['admission_pass'], windows=len(summaries), peak_allocated_bytes=peak)


if __name__ == '__main__':
    main()
