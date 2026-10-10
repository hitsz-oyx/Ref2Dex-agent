#!/usr/bin/env python3
"""Same identities/visibility, image-coordinate HISTORY extrapolation diagnostic."""
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


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def project_world(points, w2c, intrinsic):
    camera = points @ w2c[:3, :3].T + w2c[:3, 3]
    pixels = camera @ intrinsic.T
    if not np.isfinite(pixels).all() or (camera[:, 2] <= 0).any():
        raise ValueError('scored camera projection invalid; do not silently change denominator')
    return pixels[:, :2] / pixels[:, 2:]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--raw', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('output collision')
    manifest = json.loads((args.data / 'manifest.json').read_text())
    if manifest['protocol']['supervised_horizon'] != 8:
        raise ValueError('fixed h8 audit only')
    started, metrics = time.monotonic(), {}
    for e in manifest['sequences']:
        pack = args.data / e['file']
        raw = args.raw / e['scene'] / 'spatracker.npz'
        if sha(pack) != e['sha256'] or sha(raw) != e['source_sha256']['spatracker.npz']:
            raise ValueError('source hash drift')
        with np.load(raw, allow_pickle=False) as z:
            intrinsics, extrinsics = z['intrinsics'].astype(float), z['extrinsics'].astype(float)
        w2cs = extrinsics if e['camera_inference']['convention'] == 'w2c' else np.linalg.inv(extrinsics)
        totals = {n: [0., 0] for n in ('static', 'last_two', 'history_ols')}
        with np.load(pack, allow_pickle=False) as z:
            for block in e['window_starts']:
                s = sample_window(z['points'][block], z['valid'][block], z['kind'][block], z['timestamps'][block], 0)
                ids = s['selected_track_ids'][(s['point_kind'] == 1) & s['target_valid'][:, 7]]
                if not len(ids):
                    continue
                rows = z['source_frame_ids'][block] - e['source_range'][0]
                if rows.min() < 0 or rows.max() >= len(w2cs):
                    raise ValueError('original frame to camera-row correspondence differs')
                ticks = (0, 1, 2, 3, 11)
                uv = np.stack([project_world(z['points'][block, t, ids], w2cs[rows[t]], intrinsics[rows[t]]) for t in ticks])
                ts = z['timestamps'][block]
                dt = np.diff(ts[:4])
                if not np.allclose(dt, dt[0], rtol=1e-6, atol=1e-8):
                    raise ValueError('nonuniform HISTORY clock')
                horizon = ts[11] - ts[3]
                last_velocity = (uv[3] - uv[2]) / dt[0]
                ols_velocity = (.3 * (uv[3] - uv[0]) + .1 * (uv[2] - uv[1])) / dt[0]
                predictions = dict(static=uv[3], last_two=uv[3]+last_velocity*horizon,
                                   history_ols=uv[3]+ols_velocity*horizon)
                for name, pred in predictions.items():
                    totals[name][0] += float(np.linalg.norm(pred - uv[4], axis=-1).sum())
                    totals[name][1] += len(ids)
        metrics[e['split']+'/'+e['scene']] = {name: dict(epe_px=total/count if count else None, support=count)
                                              for name, (total, count) in totals.items()}
        if time.monotonic()-started > 30:
            raise TimeoutError('30s CPU audit cap')
    dev = [v for k,v in metrics.items() if k.startswith('dev/') and v['static']['support'] >= 16]
    passed = len(dev) >= 2 and all(v['history_ols']['epe_px'] <= .9*v['static']['epe_px'] for v in dev)
    result = dict(schema='ref2dex.video-image-motion-audit.v1', horizon=8, metrics=metrics,
        decision_signal='IMAGE_HISTORY_SIGNAL' if passed else 'NO_PRESET_IMAGE_HISTORY_SIGNAL',
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        data_manifest_sha256=sha(args.data/'manifest.json'), script_sha256=sha(__file__),
        raw_root=str(args.raw.resolve()), elapsed_seconds=time.monotonic()-started,
        limitations=['same history-selected identities and h8 mask as 3D audit; no additional filtering',
            'image coordinates contain camera motion, whereas 3D targets are inferred world coordinates',
            'future camera is used ONLY to reconstruct endpoint label, never the predictor',
            'a 2D/3D discrepancy cannot alone isolate depth error, camera error or label accuracy',
            'no physical GT, learned-model or action causal conclusion'])
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
