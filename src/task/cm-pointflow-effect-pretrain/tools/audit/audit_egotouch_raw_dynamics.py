#!/usr/bin/env python3
"""Raw byte sensor predictability; no contact, force or world-geometry claim."""
import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
from audit_egotouch_label_schema import read_rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('output collision')
    started = time.monotonic()
    manifest = json.loads((args.labels / 'manifest.json').read_text())
    if manifest['status'] != 'COMPLETED_SCHEMA_SAMPLE':
        raise ValueError('unqualified acquisition')
    for e in manifest['files']:
        if hashlib.sha256((args.labels / 'raw' / e['path']).read_bytes()).hexdigest() != e['sha256']:
            raise ValueError('original label drift')
    paths = sorted({(args.labels / 'raw' / e['path']).parent for e in manifest['files']})
    if len(paths) != 2:
        raise ValueError('predeclared two-record audit')
    records = []
    for path in paths:
        rows = read_rows(path / 'jq_pressure.json')
        raw = np.asarray([r['sensor_left'] + r['sensor_right'] for r in rows], dtype=float)
        ts = np.asarray([r['ts'] for r in rows], dtype=float)
        ids = [r['frame_index'] for r in rows]
        if raw.shape != (len(rows), 512) or not np.isfinite(raw).all() or raw.min() < 0 or raw.max() > 255:
            raise ValueError('raw sensor byte schema differs')
        if not np.all(np.diff(ts) > 0) or ids != list(range(len(rows))):
            raise ValueError('source frame/clock gap')
        records.append((path, raw, ts))
    # Both are official TRAIN episodes. Stable first record is fitting material;
    # second is a held original TRAIN record, not the official test set.
    active = np.ptp(records[0][1], axis=0) >= 2
    if active.sum() < 16:
        raise ValueError('too few changing TRAIN channels')
    result = dict(schema='ref2dex.egotouch-raw-sensor-dynamics.v1', history=4, horizon=8,
        scale=255, training_allowed=False, records=[], active_train_channels=np.flatnonzero(active).tolist(),
        active_channel_rule='TRAIN raw range>=2 byte counts, frozen for held record',
        label_manifest_sha256=hashlib.sha256((args.labels / 'manifest.json').read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        limitations=['two recordings, overlapping windows, no model or causal inference',
                     'raw sensor channel roles are UNKNOWN; not pressure/contact/force supervision',
                     'TRAIN-only dynamic-channel selection misses newly active held-record channels'])
    for idx, (path, raw, ts) in enumerate(records):
        errors = {name: [] for name in ('persistence', 'last_two_cv', 'history_ols')}
        changes = []
        for start in range(len(raw) - 11):
            history, future = raw[start:start + 4] / 255., raw[start + 4:start + 12] / 255.
            clock = ts[start:start + 4] - ts[start + 3]
            elapsed = ts[start + 4:start + 12] - ts[start + 3]
            centered = clock - clock.mean()
            slope = np.sum(centered[:, None] * history, axis=0) / np.sum(centered ** 2)
            velocities = dict(last_two_cv=(history[-1] - history[-2]) / (clock[-1] - clock[-2]), history_ols=slope)
            pred = dict(persistence=np.broadcast_to(history[-1], future.shape))
            pred.update({name: history[-1] + elapsed[:, None] * v for name, v in velocities.items()})
            # Known byte range only; no future-dependent normalization/clipping.
            for name, prediction in pred.items():
                errors[name].append(np.abs(np.clip(prediction, 0, 1) - future))
            changes.append(np.abs(future[-1] - history[-1]))
        metrics = {}
        for name, value in errors.items():
            e = np.stack(value)
            metrics[name] = dict(all_channel_mae=float(e.mean()),
                train_active_channel_mae=float(e[..., active].mean()),
                train_active_channel_h8_mae=float(e[:, 7, active].mean()),
                left_active_h8_mae=float(e[:, 7, :256][:, active[:256]].mean()),
                right_active_h8_mae=float(e[:, 7, 256:][:, active[256:]].mean()))
        delta = np.stack(changes)
        result['records'].append(dict(record=str(path.relative_to(args.labels / 'raw')),
            split='fit' if idx == 0 else 'held_record', windows=len(changes),
            original_frames=len(raw), channels=512, raw_min=float(raw.min()), raw_max=float(raw.max()),
            dynamic_channels=int((np.ptp(raw, axis=0) >= 2).sum()),
            changing_h8_active_fraction=float((delta[:, active] * 255 >= 2 - 1e-6).mean()),
            metrics=metrics))
    held = result['records'][1]['metrics']
    result['decision_signal'] = ('HISTORY_SIGNAL' if held['history_ols']['train_active_channel_h8_mae'] <=
        .9 * held['persistence']['train_active_channel_h8_mae'] else 'NO_PRESET_HISTORY_SIGNAL')
    result['elapsed_seconds'] = time.monotonic() - started
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'active_train_channels'}, indent=2))


if __name__ == '__main__':
    main()
