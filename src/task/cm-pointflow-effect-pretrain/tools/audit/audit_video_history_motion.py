#!/usr/bin/env python3
"""Compare observed-only velocity estimators before more weak-video training."""
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
from oakink_wm.video_points import VideoWindows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('output collision')
    started = time.monotonic()
    manifest = json.loads((args.data / 'manifest.json').read_text())
    for e in manifest['sequences']:
        if hashlib.sha256((args.data / e['file']).read_bytes()).hexdigest() != e['sha256']:
            raise ValueError('source hash drift')
    totals = {}
    for split in ('train', 'dev'):
        dataset = VideoWindows(args.data, split)
        for i in range(len(dataset)):
            s = dataset[i]
            scene, block = dataset.index[i]
            times = dataset.sequences[scene]['timestamps'][block][:4]
            dt = np.diff(times)
            if not np.allclose(dt, dt[0], atol=1e-8, rtol=1e-6):
                raise ValueError('closed-form estimator requires uniform HISTORY clock')
            # Exact least-squares slope for four equally spaced HISTORY points:
            # [3*(p3-p0)+(p2-p1)]/(10*dt). Uses observed features only.
            displacement = s['features'][:, 9:12]
            prev = s['velocity'] - s['features'][:, 6:9] * dt[0]
            velocities = dict(last_two=s['velocity'], history_mean=displacement / (3 * dt[0]),
                              history_ols=.3 * displacement / dt[0] + .1 * prev)
            predictions = {name: v * s['elapsed'][7] for name, v in velocities.items()}
            predictions['static'] = np.zeros_like(s['velocity'])
            for kind, group in ((1, 'object'), (0, 'background')):
                support = s['target_valid'][:, 7] & (s['point_kind'] == kind)
                for name, pred in predictions.items():
                    key = split + '/' + scene + '/' + group
                    pair = totals.setdefault(key, {}).setdefault(name, [0., 0])
                    pair[0] += float(np.linalg.norm(pred[support] - s['target_flow'][support, 7], axis=-1).sum())
                    pair[1] += int(support.sum())
            if time.monotonic() - started > 30:
                raise TimeoutError('history-motion audit30s CPU budget')
    metrics = {key: {name: dict(epe_m=total / count if count else None, support=count)
                    for name, (total, count) in values.items()} for key, values in totals.items()}
    qualified = [v for k, v in metrics.items() if k.startswith('dev/') and k.endswith('/object') and v['static']['support'] >= 16]
    positive = len(qualified) >= 2 and all(v['history_ols']['epe_m'] <= .9 * v['static']['epe_m'] for v in qualified)
    result = dict(schema='ref2dex.video-history-motion-audit.v1', horizon=8,
        data_manifest_sha256=hashlib.sha256((args.data / 'manifest.json').read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        metrics=metrics, decision_signal='OBSERVED_MOTION_SIGNAL' if positive else 'NO_PRESET_SIGNAL',
        limitations=['pseudo-label errors conflate physical motion and reconstruction noise',
                     'no sensor GT, model fit or action causal claim'], elapsed_seconds=time.monotonic() - started)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
