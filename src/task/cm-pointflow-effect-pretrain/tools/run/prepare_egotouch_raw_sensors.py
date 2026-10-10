#!/usr/bin/env python3
"""Qualify diverse raw sensor/hand windows with explicit pseudo-action support."""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK / 'tools/audit'))
sys.path.insert(0, str(TASK / 'src'))
from audit_egotouch_label_schema import read_rows
from oakink_wm.raw_sensors import sensor_window


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    original = json.loads((args.source / 'manifest.json').read_text())
    if original['status'] != 'COMPLETED_DIVERSE_RAW_SAMPLE':
        raise ValueError('acquisition incomplete')
    for e in original['files']:
        if sha(args.source / 'raw' / e['path']) != e['sha256']:
            raise ValueError('original source hash drift')
    manifest = dict(schema='ref2dex.raw-sensor-corpus.v1', status='PREPARING', records=[],
        native_training_allowed=False, hardware_channel_roles_qualified=False,
        protocol=dict(history=4, horizon=24, scale=255, min_frames=128, min_tasks_per_split=3,
            hand_coordinates='per-frame wrist-relative, HISTORY-median palm span only',
            window_support='requires complete H4+K24 two-hand pseudo-action availability for ALL arms',
            sensor_target_selection='no future sensor-value selection'),
        source_manifest_sha256=sha(args.source / 'manifest.json'),
        source_root=str(args.source.resolve()),
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        implementation_sha256={str(p.relative_to(TASK)):sha(p) for p in
            (Path(__file__).resolve(), TASK / 'src/oakink_wm/raw_sensors.py', TASK / 'tools/audit/audit_egotouch_label_schema.py')})
    try:
        for entry in original['selection']:
            path = args.source / 'raw' / entry['record']
            pressure, hands = [read_rows(path / n) for n in ('jq_pressure.json', 'wilor_hands.json')]
            ids = [r['frame_index'] for r in pressure]
            if ids != list(range(len(pressure))) or ids != [r['frame_index'] for r in hands]:
                raise ValueError('original frame gap or modality mismatch')
            if len(pressure) < 128:
                raise ValueError('record shorter than128 original frames')
            raw = np.asarray([r['sensor_left'] + r['sensor_right'] for r in pressure], dtype='float32')
            if raw.shape != (len(pressure), 512) or not np.isfinite(raw).all() or raw.min() < 0 or raw.max() > 255:
                raise ValueError('original byte sensor schema differs')
            joints = np.full((len(pressure), 2, 21, 3), np.nan, 'float32')
            hand_valid = np.zeros((len(pressure), 2), bool)
            for t, row in enumerate(hands):
                for side, key in enumerate(('left_pos', 'right_pos')):
                    a = np.asarray(row[key], dtype='float32')
                    if a.shape == (21, 3) and np.isfinite(a).all() and np.linalg.norm(a[9] - a[0]) > 1e-6:
                        joints[t, side], hand_valid[t, side] = a, True
            timestamps = np.asarray([r['ts'] for r in pressure], dtype='float64')
            if not np.isfinite(timestamps).all() or not (np.diff(timestamps) > 0).all():
                raise ValueError('nonmonotonic sensor clock')
            starts = [s for s in range(len(raw) - 27) if hand_valid[s:s + 28].all()]
            for start in starts:
                sensor_window(raw, joints, timestamps, start)
            output = args.output / (entry['record'].replace('/', '__') + '.npz')
            np.savez_compressed(output, sensors=raw, joints=joints, timestamps=timestamps,
                                hand_valid=hand_valid, source_frame_ids=np.asarray(ids))
            manifest['records'].append(dict(entry, file=output.name, sha256=sha(output),
                original_frames=len(raw), window_starts=starts, complete_bimanual_frames=int(hand_valid.all(1).sum()),
                original_missing_hand_frames=int((~hand_valid.all(1)).sum()),
                valid_is_shape_finite_only=True, changing_raw_channels=int((np.ptp(raw, 0) >= 2).sum()),
                original_split='official TRAIN'))
            print('%s %s: %d frames, %d complete windows' % (entry['split'], entry['record'], len(raw), len(starts)), flush=True)
        counts = {split: sum(r['split'] == split and bool(r['window_starts']) for r in manifest['records'])
                  for split in ('fit', 'held_task')}
        manifest['qualified_tasks'] = counts
        fit_raw = [np.load(args.output / r['file'], allow_pickle=False)['sensors'] for r in manifest['records'] if r['split'] == 'fit']
        merged = np.concatenate(fit_raw)
        manifest['train_dynamic_channels'] = np.flatnonzero(np.ptp(merged, axis=0) >= 2).tolist()
        manifest['dynamic_channel_selection'] = 'FIT original raw range>=2 counts, includes all FIT frames; no held values'
        manifest['status'] = 'QUALIFIED_RAW_SENSOR_PROBE_ONLY' if min(counts.values()) >= 3 else 'INSUFFICIENT_PSEUDO_ACTION_SUPPORT'
    except Exception as exc:
        manifest['status'], manifest['error'] = 'FAILED', str(exc)
        raise
    finally:
        manifest['elapsed_seconds'] = time.monotonic() - started
        (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
