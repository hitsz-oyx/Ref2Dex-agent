#!/usr/bin/env python3
"""Bounded preparation of source-video-isolated weak point-track windows."""
import argparse
import hashlib
import json
import sys
import tarfile
import time
from pathlib import Path

import numpy as np

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK / 'src'))
sys.path.insert(0, str(TASK / 'tools/audit'))
from oakink_wm.video_points import sample_window
from convert_epic_scene_flow import (sha, source_clock, read_video, load_object_masks,
                                     load_hand_masks, rigid_valid, camera_to_world,
                                     world_to_camera, backproject, track_object_rgb,
                                     static_convention_diagnostic, infer_extrinsics_convention)


def background_points(depths, Ks, w2cs, masks, hands, frames, max_points=256):
    h, w = depths.shape[1:]
    yy, xx = np.mgrid[2:h - 2:4, 2:w - 2:4]
    keep = (depths[0, yy, xx] > .05) & np.isfinite(depths[0, yy, xx]) & ~masks[0, yy, xx] & ~hands[0, yy, xx]
    uv = np.stack((xx[keep], yy[keep]), -1)
    if len(uv) > max_points:
        uv = uv[np.linspace(0, len(uv) - 1, max_points, dtype=int)]
    world = camera_to_world(backproject(uv, depths[0, uv[:, 1], uv[:, 0]], Ks[0]), w2cs[0])
    points = np.zeros((len(frames), len(uv), 3), 'float32')
    valid = np.zeros(points.shape[:2], bool)
    for j, frame in enumerate(frames):
        camera = world_to_camera(world, w2cs[frame])
        q = camera @ Ks[frame].T
        pixels = q[:, :2] / np.maximum(q[:, 2:], 1e-8)
        u, v = np.rint(pixels).astype(int).T
        good = (camera[:, 2] > .05) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        ids = np.flatnonzero(good)
        depth = depths[frame, v[ids], u[ids]]
        good[ids] &= (np.isfinite(depth) & (depth > .05)
                      & (np.abs(depth - camera[ids, 2]) <= np.maximum(.03, .05 * camera[ids, 2]))
                      & ~masks[frame, v[ids], u[ids]] & ~hands[frame, v[ids], u[ids]])
        ids = np.flatnonzero(good)
        valid[j] = good
        points[j, ids] = camera_to_world(backproject(np.stack((u[ids], v[ids]), -1),
                                           depths[frame, v[ids], u[ids]], Ks[frame]), w2cs[frame])
    return points, valid


def object_points(depths, Ks, w2cs, masks, gray, frames):
    pixels = track_object_rgb(gray, masks, frames, max_corners=256,
                               forward_backward_threshold_px=1.)
    valid = np.isfinite(pixels).all(-1)
    points = np.zeros((*valid.shape, 3), 'float32')
    for j, frame in enumerate(frames):
        ids = np.flatnonzero(valid[j])
        u, v = np.rint(pixels[j, ids]).astype(int).T
        u, v = u.clip(0, depths.shape[2] - 1), v.clip(0, depths.shape[1] - 1)
        depth = depths[frame, v, u]
        good = np.isfinite(depth) & (depth > .05)
        valid[j, ids[~good]] = False
        points[j, ids[good]] = camera_to_world(backproject(pixels[j, ids[good]], depth[good], Ks[frame]), w2cs[frame])
    return points, valid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-seconds', type=float, default=1200)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started, extracted = time.monotonic(), 0
    inventory = json.loads(args.inventory.read_text())['scenes']
    official = set((args.data_root / 'metadata/train.txt').read_text().splitlines())
    manifest = dict(schema='ref2dex.video-point-corpus.v1', status='PREPARING',
                    native_training_allowed=False, sequences=[], skipped=[],
                    protocol=dict(history=4, horizon=24, source_stride=2, points_per_kind=128,
                                  fb_threshold_pixels=1., max_clips_per_video=4,
                                  train_video='P03_03', dev_video='P03_13'),
                    source_inventory_sha256=sha(args.inventory),
                    official_train_sha256=sha(args.data_root / 'metadata/train.txt'),
                    implementation_sha256={str(p.relative_to(TASK)): sha(p) for p in
                        (Path(__file__).resolve(), TASK / 'src/oakink_wm/video_points.py',
                         TASK / 'tools/audit/convert_epic_scene_flow.py')})
    try:
        for video, split in [('P03_03', 'train'), ('P03_13', 'dev')]:
            accepted = []
            with tarfile.open(args.data_root / 'shard' / (video + '.tar')) as tar:
                members = {m.name: m for m in tar.getmembers() if m.isfile()}
                for scene, entry in sorted(inventory.items()):
                    if not scene.startswith(video + '_') or len(accepted) >= 4:
                        continue
                    meta = entry['metadata']
                    lo, hi = int(meta['start_frame']), int(meta['stop_frame'])
                    if not 55 <= hi - lo <= 600 or any(max(lo, a) < min(hi, b) for a, b in accepted):
                        manifest['skipped'].append(dict(scene=scene, reason='length_or_overlap'))
                        continue
                    objects = sorted(k for k in official if k.startswith(scene + '/objects/'))
                    if len(objects) != 1:
                        manifest['skipped'].append(dict(scene=scene, reason='require_one_official_train_object'))
                        continue
                    prefix = video + '/' + scene + '/'
                    names = ['action.meta.json', 'action.mp4', 'spatracker.npz',
                             'egohos/twohands_masks.npz', objects[0].split('/', 1)[1] + '/masks.npz']
                    if any(prefix + name not in members for name in names):
                        manifest['skipped'].append(dict(scene=scene, reason='missing_required_source'))
                        continue
                    raw = args.output / 'raw' / scene
                    source_hashes = {}
                    for name in names:
                        m = members[prefix + name]
                        if extracted + m.size > 2 * 1024 ** 3:
                            raise RuntimeError('raw extraction cap exceeded')
                        p = raw / name
                        p.parent.mkdir(parents=True, exist_ok=True)
                        with tar.extractfile(m) as source, p.open('wb') as output:
                            while True:
                                chunk = source.read(1024 * 1024)
                                if not chunk:
                                    break
                                output.write(chunk)
                        extracted += m.size
                        source_hashes[name] = sha(p)
                    try:
                        with np.load(raw / 'spatracker.npz', allow_pickle=False) as z:
                            depths, Ks, extrinsics = [z[k].astype('float32') for k in ('depths', 'intrinsics', 'extrinsics')]
                        n, h, w = depths.shape
                        start, fps = source_clock(json.loads((raw / 'action.meta.json').read_text()), n)
                        if not rigid_valid(extrinsics).all():
                            raise ValueError('non-rigid camera')
                        diagnostic_frames = np.unique(np.linspace(0, n - 1, min(n, 64), dtype=int))
                        diagnostic = static_convention_diagnostic(depths, Ks, extrinsics, diagnostic_frames)
                        convention = infer_extrinsics_convention(diagnostic, 5.)
                        w2cs = extrinsics if convention['convention'] == 'w2c' else np.linalg.inv(extrinsics)
                        gray = read_video(raw / 'action.mp4', (w, h))
                        if len(gray) != n:
                            raise ValueError('decoded video length differs from depth/clock')
                        masks = load_object_masks(raw / names[-1], (h, w), n)
                        hands = load_hand_masks(raw / 'egohos/twohands_masks.npz', (h, w), n)
                        frames = np.arange(0, n, 2)
                        bg, bv = background_points(depths, Ks, w2cs, masks, hands, frames)
                        ob, ov = object_points(depths, Ks, w2cs, masks, gray, frames)
                        points, valid = np.concatenate((ob, bg), 1), np.concatenate((ov, bv), 1)
                        kind = np.concatenate((np.ones(ob.shape[1], 'int64'), np.zeros(bg.shape[1], 'int64')))
                        timestamps = frames.astype(float) / fps
                        starts = []
                        for tick in range(len(frames) - 27):
                            try:
                                sample_window(points, valid, kind, timestamps, tick)
                                starts.append(tick)
                            except ValueError:
                                pass
                        if not starts:
                            raise ValueError('no history-qualified windows')
                        output = args.output / (scene + '.npz')
                        np.savez_compressed(output, points=points, valid=valid, kind=kind,
                                            timestamps=timestamps, source_frame_ids=start + frames)
                        manifest['sequences'].append(dict(scene=scene, video=video, split=split,
                                  file=output.name, sha256=sha(output), original_split='ObjectForesight train',
                                  source_range=[lo, hi], sampled_fps=fps / 2,
                                  window_starts=starts, camera_diagnostic=diagnostic,
                                  camera_inference=convention, source_sha256=source_hashes,
                                  history_selection_uses_future_validity=False,
                                  object_track_candidates=ob.shape[1], background_track_candidates=bg.shape[1]))
                        accepted.append((lo, hi))
                        print('%s %s: %d windows, %d object / %d background candidates' %
                              (split, scene, len(starts), ob.shape[1], bg.shape[1]), flush=True)
                    except ValueError as exc:
                        manifest['skipped'].append(dict(scene=scene, reason=str(exc)))
                    if time.monotonic() - started > args.max_seconds:
                        raise TimeoutError('video preparation deadline')
        counts = {split: sum(e['split'] == split for e in manifest['sequences']) for split in ('train', 'dev')}
        if min(counts.values()) < 2:
            raise ValueError('fewer than2 qualified clips in a partition')
        manifest['status'] = 'QUALIFIED_VIDEO_PROBE_ONLY'
    except Exception as exc:
        manifest['status'], manifest['error'] = 'FAILED', str(exc)
        raise
    finally:
        import subprocess
        manifest.update(git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                        extracted_bytes=extracted, elapsed_seconds=time.monotonic() - started)
        (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
