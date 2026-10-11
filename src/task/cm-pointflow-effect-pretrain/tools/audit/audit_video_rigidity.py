#!/usr/bin/env python3
"""Oracle SE(3) consistency of weak tracks; never a predictive baseline."""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK / 'src'))
from oakink_wm.video_points import sample_window


def rigid_residual(source, target):
    if len(source) < 8:
        return None
    a, b = source - source.mean(0), target - target.mean(0)
    if np.linalg.svd(a, compute_uv=False)[1] < 1e-5:
        return None
    u, _, vh = np.linalg.svd(a.T @ b)
    sign = np.ones(3)
    sign[-1] = np.linalg.det(u @ vh)
    rotation = (u * sign) @ vh
    fitted = a @ rotation + target.mean(0)
    return np.linalg.norm(fitted - target, axis=-1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('output collision')
    started = time.monotonic()
    manifest = json.loads((args.data / 'manifest.json').read_text())
    result = dict(schema='ref2dex.video-rigid-consistency.v1', horizon=8, clips=[],
        data_manifest_sha256=hashlib.sha256((args.data / 'manifest.json').read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        interpretation='oracle fit to scored labels; cannot establish GT accuracy or forecasting',
        min_points_per_window=8, min_second_singular_value_m=1e-5)
    for e in manifest['sequences']:
        path = args.data / e['file']
        if hashlib.sha256(path.read_bytes()).hexdigest() != e['sha256']:
            raise ValueError('source drift')
        details, residuals, motions = [], [], []
        with np.load(path, allow_pickle=False) as z:
            for block in e['window_starts']:
                s = sample_window(z['points'][block], z['valid'][block], z['kind'][block], z['timestamps'][block], 0)
                ids = s['selected_track_ids'][(s['point_kind'] == 1) & s['target_valid'][:, 7]]
                src, dst = z['points'][block, 3, ids], z['points'][block, 11, ids]
                residual = rigid_residual(src.astype(float), dst.astype(float))
                motion = np.linalg.norm(dst - src, axis=-1)
                if residual is None:
                    details.append(dict(block=block, supported_points=len(ids), qualified=False))
                    continue
                residuals.extend(residual.tolist())
                motions.extend(motion.tolist())
                details.append(dict(block=block, supported_points=len(ids), qualified=True,
                    rigid_residual_mean_m=float(residual.mean()), static_epe_m=float(motion.mean()),
                    residual_to_motion_ratio=float(residual.mean() / max(motion.mean(), 1e-9))))
        eligible = [w for w in details if w['qualified']]
        ratio = float(np.mean(residuals) / max(np.mean(motions), 1e-9)) if residuals else None
        result['clips'].append(dict(scene=e['scene'], split=e['split'], windows=details,
            qualified_windows=len(eligible), supported_point_windows=len(residuals),
            rigid_residual_mean_m=float(np.mean(residuals)) if residuals else None,
            static_epe_m=float(np.mean(motions)) if motions else None,
            residual_to_motion_ratio=ratio))
        if time.monotonic() - started > 30:
            raise TimeoutError('30s CPU audit budget')
    dev = [c for c in result['clips'] if c['split'] == 'dev' and c['supported_point_windows'] >= 16]
    ratios = [c['residual_to_motion_ratio'] for c in dev]
    result['decision_signal'] = ('RIGID_INCONSISTENCY' if len(ratios) >= 2 and all(r >= .5 for r in ratios)
        else 'MOSTLY_RIGID' if len(ratios) >= 2 and all(r <= .25 for r in ratios) else 'UNCLEAR')
    result['elapsed_seconds'] = time.monotonic() - started
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'clips'}, indent=2))
    for c in result['clips']:
        print(json.dumps({k: v for k, v in c.items() if k != 'windows'}))


if __name__ == '__main__':
    main()
