"""Read-only native BC cache audit; no network inference or GPU initialization."""
import argparse
import json
import subprocess
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np


@lru_cache(maxsize=128)
def source_manifest(root):
    m = json.loads((Path(root) / 'trajectory-manifest.json').read_text())
    return m, {(r['episode'], r['env']): r for r in m['trajectories']}


@lru_cache(maxsize=4096)
def geometry_check(path):
    with np.load(path, allow_pickle=False) as z:
        points, normals = z['surface'], z['normals']
        if points.shape != normals.shape or points.ndim != 2 or points.shape[1] != 3:
            raise ValueError('CAD point/normal shape')
        if not np.isfinite(points).all() or not np.isfinite(normals).all():
            raise ValueError('nonfinite CAD')
        if (np.linalg.norm(normals, axis=-1) > 1e-6).sum() < 32:
            raise ValueError('fewer than32 usable CAD points')
    return True


def inspect(cache, row):
    raw = Path(row['raw_episode'])
    manifest, sources = source_manifest(str(raw.parent))
    source = sources[int(raw.stem.split('-')[-1]), row['env']]
    for key in ('object_code', 'scale', 'split'):
        if source[key] != row[key]:
            raise ValueError('raw manifest identity mismatch: ' + key)
    if not source['endpoint_within_5cm']:
        raise ValueError('source not a terminal-success candidate')
    objid = next(i for i, obj in enumerate(manifest['objects'])
                 if obj['object_code'] == row['object_code'])
    if Path(row['geometry']).resolve() != (raw.parent / 'geometry' / ('object-%03d.npz' % objid)).resolve():
        raise ValueError('geometry identity mismatch')
    geometry_check(row['geometry'])
    with np.load(cache / row['path'], allow_pickle=False) as z:
        d = {key: z[key] for key in z.files}
    n = len(d['hand'])
    expected = dict(hand=(n, 11, 3), poses=(n, 4, 4), goal=(n, 3),
                    observation=(n, 300), action=(n-1, 24), time=(n,))
    for key, shape in expected.items():
        if d[key].shape != shape or not np.isfinite(d[key]).all():
            raise ValueError('shape/nonfinite: ' + key)
    if not np.allclose(np.diff(d['time']), 1/60, atol=1e-7, rtol=0):
        raise ValueError('60Hz state clock')
    if np.abs(d['action']).max() > 1.00001:
        raise ValueError('canonical command range')
    previous_error = float(np.abs(d['observation'][1:, 167:191] - d['action']).max())
    if previous_error != 0:
        raise ValueError('observation[t+1] previous action != command[t]')
    if not np.allclose(d['goal'], d['goal'][0], atol=1e-7, rtol=0):
        raise ValueError('decision-known task goal changes inside episode')
    r = d['poses'][:, :3, :3]
    so3_error = float(np.abs(r.transpose(0, 2, 1) @ r - np.eye(3)).max())
    if so3_error > 1e-3 or np.abs(np.linalg.det(r) - 1).max() > 1e-3:
        raise ValueError('object rotation is not SO(3)')
    if not np.allclose(d['poses'][:, 3], [0, 0, 0, 1], atol=1e-7, rtol=0):
        raise ValueError('homogeneous object pose')
    ticks = np.asarray(row['ticks'], dtype=np.int64)
    if not len(ticks) or np.any(np.diff(ticks) <= 0) or ticks[0] < 0:
        raise ValueError('nonempty sorted unique query ticks')
    if ticks[-1]+48 >= n or ticks[-1]+8 > len(d['action']):
        raise ValueError('incomplete forecast/action support')
    speed = np.linalg.norm(np.diff(d['poses'][:, :3, 3], axis=0), axis=-1)*60
    hand_speed = np.linalg.norm(np.diff(d['hand'], axis=0), axis=-1).max(-1)*60
    rel = r[:-1].transpose(0, 2, 1) @ r[1:]
    angular = np.arccos(np.clip((np.trace(rel, axis1=1, axis2=2)-1)/2, -1, 1))*60
    bad = np.flatnonzero((speed > 5) | (hand_speed > 5) | (angular > 40))+1
    onset = int(bad[0]) if len(bad) else n
    if row['unsafe_onset'] != onset or np.any(ticks+48 >= onset):
        raise ValueError('window reaches first unsafe transition or later tail')
    local = np.einsum('tpi,tij->tpj', d['hand']-d['poses'][:, None, :3, 3], r)
    allowed = np.arange(max(0, min(n-48, onset-48)))
    allowed = allowed[(local[allowed] >= -2).all((1, 2))]
    if not np.array_equal(ticks, allowed):
        raise ValueError('admitted tick set differs from declared policy')
    return dict(previous_command_max_error=previous_error, rotation_max_error=so3_error,
                admitted_max_object_speed_mps=float(speed[:ticks[-1]+48].max()),
                admitted_max_hand_speed_mps=float(hand_speed[:ticks[-1]+48].max()),
                admitted_max_angular_speed_radps=float(angular[:ticks[-1]+48].max()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--budget-seconds', type=int, default=600)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[5]
    if a.output.exists() or root/'outputs/shadow-finetune' not in a.output.resolve().parents:
        raise ValueError('fresh Task output required')
    m = json.loads((a.cache/'manifest.json').read_text())
    if m.get('schema') != 'ref2dex.shadow-native-policy.v1' or m.get('status') != 'COMPLETED':
        raise ValueError('completed native cache required')
    a.output.mkdir(parents=True)
    started = time.monotonic()
    failures, maxima, identities, objects, counts, windows = [], {}, set(), {}, Counter(), Counter()
    for i, row in enumerate(m['records']):
        if time.monotonic()-started > a.budget_seconds:
            raise TimeoutError('bounded cache audit')
        try:
            identity = (str(Path(row['raw_episode']).resolve()), row['env'])
            if identity in identities:
                raise ValueError('duplicate raw trajectory')
            identities.add(identity)
            for key, value in inspect(a.cache, row).items():
                maxima[key] = max(maxima.get(key, 0.), value)
            objects.setdefault(row['split'], set()).add(row['object_code'])
            counts[row['split']] += 1
            windows[row['split']] += len(row['ticks'])
        except Exception as error:
            failures.append(dict(row=i, path=row['path'], error=repr(error)))
        if (i+1) % 1000 == 0:
            print(json.dumps(dict(reviewed=i+1, failures=len(failures), elapsed_s=time.monotonic()-started)), flush=True)
    overlap = sorted(objects.get('train', set()) & objects.get('dev', set()))
    report = dict(status='PASS' if not failures and not overlap else 'FAIL',
                  git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
                  cache=str(a.cache.resolve()), sequences=len(m['records']), windows=dict(windows),
                  split_sequences=dict(counts), split_objects={k: len(v) for k, v in objects.items()},
                  object_overlap=overlap, maxima=maxima, failures=failures,
                  elapsed_s=time.monotonic()-started,
                  scope='CPU NumPy cache and raw-manifest audit; all saved previous commands checked. '
                        'No full teacher-network recomputation, stable-contact certification, or GPU execution.')
    (a.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)
    if report['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
