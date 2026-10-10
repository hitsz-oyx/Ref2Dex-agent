#!/usr/bin/env python3
"""Rebirth weak point identities at each HISTORY start, never within a window."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK / 'src'))
sys.path.insert(0, str(TASK / 'tools/run'))
from oakink_wm.video_points import sample_window
from prepare_epic_video_points import (sha, source_clock, read_video, load_object_masks,
    load_hand_masks, background_points, object_points)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-seconds', type=float, default=600)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    original = json.loads((args.source / 'manifest.json').read_text())
    if original['status'] != 'QUALIFIED_VIDEO_PROBE_ONLY':
        raise ValueError('unqualified source manifest')
    manifest = dict(schema='ref2dex.video-point-corpus.v2', status='PREPARING',
        native_training_allowed=False, sequences=[], skipped=[],
        protocol=dict(original['protocol'], track_birth='window_history_start',
                      storage='window_blocks', min_h24_object_points_per_clip=16,
                      min_h24_object_supported_clips_per_split=2),
        source_manifest_sha256=sha(args.source / 'manifest.json'),
        source_root=str(args.source.resolve()),
        implementation_sha256={str(p.relative_to(TASK)): sha(p) for p in
            (Path(__file__).resolve(), TASK / 'tools/run/prepare_epic_video_points.py',
             TASK / 'tools/audit/convert_epic_scene_flow.py', TASK / 'src/oakink_wm/video_points.py')})
    try:
        for entry in original['sequences']:
            raw = args.source / 'raw' / entry['scene']
            for name, checksum in entry['source_sha256'].items():
                if sha(raw / name) != checksum:
                    raise ValueError('raw source hash drift: ' + name)
            with np.load(raw / 'spatracker.npz', allow_pickle=False) as z:
                depths, Ks, extrinsics = [z[k].astype('float32') for k in ('depths', 'intrinsics', 'extrinsics')]
            n, h, w = depths.shape
            start, fps = source_clock(json.loads((raw / 'action.meta.json').read_text()), n)
            w2cs = extrinsics if entry['camera_inference']['convention'] == 'w2c' else np.linalg.inv(extrinsics)
            gray = read_video(raw / 'action.mp4', (w, h))
            if len(gray) != n:
                raise ValueError('RGB length mismatch')
            object_file = next(name for name in entry['source_sha256'] if name.startswith('objects/'))
            masks = load_object_masks(raw / object_file, (h, w), n)
            hands = load_hand_masks(raw / 'egohos/twohands_masks.npz', (h, w), n)
            blocks, support, rejected = [], [], []
            # All source-stride2 starts, never filtered using future support.
            for birth in range(0, n - 54, 2):
                if time.monotonic() - started > args.max_seconds:
                    raise TimeoutError('window preparation deadline')
                local = np.arange(0, 55, 2)
                span = slice(birth, birth + 55)
                try:
                    ob, ov = object_points(depths[span], Ks[span], w2cs[span], masks[span], gray[span], local)
                    bg, bv = background_points(depths[span], Ks[span], w2cs[span], masks[span], hands[span], local)
                    points = np.zeros((28, 512, 3), 'float32')
                    valid = np.zeros((28, 512), bool)
                    points[:, :ob.shape[1]], valid[:, :ob.shape[1]] = ob, ov
                    points[:, 256:256 + bg.shape[1]], valid[:, 256:256 + bg.shape[1]] = bg, bv
                    kind = np.repeat([1, 0], 256).astype('int64')
                    timestamps = (birth + local).astype(float) / fps
                    sample = sample_window(points, valid, kind, timestamps, 0)
                    h24 = int((sample['target_valid'][:, 23] & (sample['point_kind'] == 1)).sum())
                    blocks.append((points, valid, kind, timestamps, start + birth + local))
                    support.append(dict(birth_source_frame=start + birth, object_h24=h24))
                except ValueError as exc:
                    rejected.append(dict(birth_source_frame=start + birth, reason=str(exc)))
            if not blocks:
                manifest['skipped'].append(dict(scene=entry['scene'], reason='no history-valid window births'))
                continue
            output = args.output / (entry['scene'] + '.npz')
            np.savez_compressed(output, **{key: np.stack([b[i] for b in blocks]) for i, key in
                enumerate(('points', 'valid', 'kind', 'timestamps', 'source_frame_ids'))})
            manifest['sequences'].append(dict(entry, file=output.name, sha256=sha(output),
                window_starts=list(range(len(blocks))), window_support=support,
                rejected_history_windows=rejected, storage='window_blocks',
                object_h24_support=sum(s['object_h24'] for s in support),
                track_identity_scope='scene/birth_source_frame/initial_corner_id'))
            print('%s %s: %d windows, object h24 support %d' %
                (entry['split'], entry['scene'], len(blocks), sum(s['object_h24'] for s in support)), flush=True)
        coverage = {split: sum(e['split'] == split and e['object_h24_support'] >= 16
                              for e in manifest['sequences']) for split in ('train', 'dev')}
        manifest['h24_supported_clips'] = coverage
        manifest['status'] = ('QUALIFIED_VIDEO_PROBE_ONLY' if min(coverage.values()) >= 2
                              else 'INSUFFICIENT_LONG_HORIZON_SUPPORT')
    except Exception as exc:
        manifest['status'], manifest['error'] = 'FAILED', str(exc)
        raise
    finally:
        manifest.update(git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                        elapsed_seconds=time.monotonic() - started,
                        bytes_written=sum(p.stat().st_size for p in args.output.glob('*.npz')))
        (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
