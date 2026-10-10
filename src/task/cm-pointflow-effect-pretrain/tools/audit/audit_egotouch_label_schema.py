#!/usr/bin/env python3
"""Inspect verified original labels without inventing a synchronized world frame."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np


def read_rows(path):
    text = Path(path).read_text()
    try:
        rows = json.loads(text)
    except json.JSONDecodeError:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(rows, list) or not rows:
        raise ValueError('expected nonempty original label rows')
    return rows


def clock(rows):
    ids = np.asarray([x['frame_index'] for x in rows], dtype=np.int64)
    ts = np.asarray([x['ts'] for x in rows], dtype=float)
    delta = np.diff(ts)
    return dict(rows=len(rows), frame_ids=ids.tolist(),
                unique_frame_ids=len(set(ids.tolist())) == len(ids),
                monotonic_frame_ids=bool((np.diff(ids) > 0).all()),
                contiguous_from_zero=ids.tolist() == list(range(len(rows))),
                finite_timestamps=bool(np.isfinite(ts).all()),
                strictly_increasing_timestamps=bool((delta > 0).all()),
                first_timestamp=float(ts[0]), last_timestamp=float(ts[-1]),
                dt_min=float(delta.min()) if len(delta) else None,
                dt_median=float(np.median(delta)) if len(delta) else None,
                dt_max=float(delta.max()) if len(delta) else None)


def audit_record(path):
    pressure, hands, vive = [read_rows(path / name) for name in
                             ('jq_pressure.json', 'wilor_hands.json', 'vive_poses.json')]
    clocks = {name: clock(rows) for name, rows in
              [('pressure', pressure), ('wilor', hands), ('vive', vive)]}
    result = dict(record=str(path), clocks=clocks,
                  frame_ids_match=clocks['pressure']['frame_ids'] == clocks['wilor']['frame_ids']
                                  == clocks['vive']['frame_ids'],
                  pressure_vive_timestamps_match=[x['ts'] for x in pressure] == [x['ts'] for x in vive],
                  wilor_aligned_to_vive_values=sorted(set(str(x.get('aligned_to_vive')) for x in hands)),
                  explicit_hand_validity_fields=sorted(k for k in hands[0] if 'valid' in k or 'conf' in k),
                  calibration_file_present=(path / 'camera_matrix.txt').is_file(),
                  manual_contact=json.loads((path / 'manual_contact_annotation.json').read_text()),
                  hand={}, pressure={})
    with np.load(path / 'pressure_grids.npz', allow_pickle=False) as grids:
        for side in ('left', 'right'):
            joints = np.asarray([x[side + '_pos'] for x in hands], dtype=float)
            raw = np.asarray([x['sensor_' + side] for x in pressure], dtype=float)
            grid = grids[side + '_pressure_grid']
            if joints.shape != (len(hands), 21, 3) or raw.shape != (len(pressure), 256):
                raise ValueError('unexpected hand or raw sensor shape')
            if grid.shape != (len(pressure), 21, 21):
                raise ValueError('pressure-grid frame count or layout differs from source rows')
            mask = np.isfinite(grid)
            finite_values = grid[mask]
            result['hand'][side] = dict(shape=list(joints.shape), all_finite=bool(np.isfinite(joints).all()),
                                        z_min=float(joints[..., 2].min()), z_max=float(joints[..., 2].max()),
                                        coordinate_semantics='original Wilor coordinates; world/calibrated metric frame unqualified')
            result['pressure'][side] = dict(raw_shape=list(raw.shape), grid_shape=list(grid.shape),
                                            measured_cells_per_frame=mask.sum((1, 2)).tolist(),
                                            sensor_layout_constant=bool((mask == mask[0]).all()),
                                            finite_min=float(finite_values.min()), finite_max=float(finite_values.max()),
                                            missing_cells_are_nan=bool(np.isnan(grid[~mask]).all()))
        result['normalization_metadata'] = {k: grids[k].item() for k in grids if grids[k].ndim == 0}
    result['training_allowed'] = False
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('output already exists')
    manifest = json.loads((args.sample_dir / 'manifest.json').read_text())
    if manifest['status'] != 'COMPLETED_SCHEMA_SAMPLE':
        raise ValueError('sample acquisition is incomplete')
    for entry in manifest['files']:
        if hashlib.sha256((args.sample_dir / 'raw' / entry['path']).read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError('verified sample file changed')
    paths = sorted({(args.sample_dir / 'raw' / entry['path']).parent for entry in manifest['files']})
    result = dict(schema='ref2dex.egotouch-label-audit.v1', training_allowed=False,
                  git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  acquisition_manifest_sha256=hashlib.sha256((args.sample_dir / 'manifest.json').read_bytes()).hexdigest(),
                  records=[audit_record(p) for p in paths],
                  limitations=['label identity agreement does not prove video synchronization',
                               'no RGB, object point tracks or calibrated camera poses audited',
                               'missing confidence and calibration require further qualification',
                               'normalize pressure and bend separately; preserve unmeasured-cell masks'])
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print('Audited %d original label bundles; training_allowed=false' % len(paths))


if __name__ == '__main__':
    main()
