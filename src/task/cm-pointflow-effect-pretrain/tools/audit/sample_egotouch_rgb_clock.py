#!/usr/bin/env python3
"""Acquire chest RGB for previously verified small labels; audit frame clocks."""
import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from urllib.parse import quote

import cv2
import numpy as np
import requests

from sample_egotouch_labels import REVISION, DATASET, verify_bytes
from audit_egotouch_label_schema import read_rows, clock


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inventory', type=Path, required=True)
    p.add_argument('--labels', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--total-cap-mib', type=float, default=2)
    p.add_argument('--per-video-cap-mib', type=float, default=1)
    p.add_argument('--deadline-seconds', type=float, default=90)
    p.add_argument('--contact-sheet', action='store_true', help='fixed early/middle/late RGB frames; no label selection')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    files = {x['path']: x for x in json.loads(args.inventory.read_text()) if x['type'] == 'file'}
    original = json.loads((args.labels / 'manifest.json').read_text())
    if original['status'] not in ('COMPLETED_SCHEMA_SAMPLE', 'COMPLETED_DIVERSE_RAW_SAMPLE') or original['revision'] != REVISION:
        raise ValueError('unqualified original acquisition')
    for entry in original['files']:
        if hashlib.sha256((args.labels / 'raw' / entry['path']).read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError('original label drift')
    records = sorted({str(Path(e['path']).parent) for e in original['files']})
    selected = [files[r + '/chest.mp4'] for r in records]
    expected = sum(e['size'] for e in selected)
    if args.total_cap_mib <= 0 or args.per_video_cap_mib <= 0 or args.deadline_seconds <= 0:
        raise ValueError('budgets must be positive')
    if expected > args.total_cap_mib * 1024 ** 2 or any(e['size'] > args.per_video_cap_mib * 1024 ** 2 for e in selected):
        raise ValueError('selected RGB exceeds declared byte caps before network')
    started, total = time.monotonic(), 0
    sheet_rows = []
    session = requests.Session()
    session.trust_env = False
    mirror_failed = False
    result = dict(schema='ref2dex.egotouch-rgb-clock.v1', status='RUNNING', dataset=DATASET,
        revision=REVISION, training_allowed=False, records=[], attempts=[],
        expected_bytes=expected, budget=dict(total_cap_mib=args.total_cap_mib, per_video_cap_mib=args.per_video_cap_mib, deadline_seconds=args.deadline_seconds),
        source_label_manifest_sha256=hashlib.sha256((args.labels / 'manifest.json').read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        limitations=['frame count and relative timing agreement is engineering consistency, not synchronization proof',
                     'no camera calibration, physical hand/world coordinate or pressure-force calibration qualified',
                     'manual contact flag semantics remain unknown'])
    try:
        for record in records:
            entry = files[record + '/chest.mp4']
            if time.monotonic() - started > args.deadline_seconds:
                raise RuntimeError('RGB total deadline')
            routes = [] if mirror_failed else [('hf-mirror.com', {}, 2)]
            routes.append(('huggingface.co', {'http': 'http://127.0.0.1:7897', 'https': 'http://127.0.0.1:7897'}, 15))
            data = None
            for host, proxies, timeout in routes:
                attempt = dict(record=record, host=host)
                result['attempts'].append(attempt)
                try:
                    url = 'https://' + host + '/datasets/' + DATASET + '/resolve/' + REVISION + '/' + quote(entry['path'])
                    chunks = []
                    with session.get(url, proxies=proxies, timeout=timeout, stream=True) as response:
                        response.raise_for_status()
                        for chunk in response.iter_content(65536):
                            total += len(chunk)
                            if total > args.total_cap_mib * 1024 ** 2 or time.monotonic() - started > args.deadline_seconds:
                                raise RuntimeError('RGB byte/time budget')
                            chunks.append(chunk)
                    data = b''.join(chunks)
                    verify_bytes(data, entry)
                    attempt['status'] = 'VERIFIED'
                    break
                except requests.RequestException as exc:
                    attempt['status'] = type(exc).__name__
                    if host == 'hf-mirror.com':
                        mirror_failed = True
            if data is None:
                raise RuntimeError('RGB download failed')
            video = args.output / 'raw' / entry['path']
            video.parent.mkdir(parents=True, exist_ok=True)
            video.write_bytes(data)
            pressure = read_rows(args.labels / 'raw' / record / 'jq_pressure.json')
            snapshot_ids = sorted(set(int((len(pressure) - 1) * q) for q in (.1, .5, .9))) if args.contact_sheet else []
            snapshots = []
            cap = cv2.VideoCapture(str(video))
            fps, frames, unique = cap.get(cv2.CAP_PROP_FPS), 0, set()
            shape = None
            while True:
                ok, image = cap.read()
                if not ok:
                    break
                if time.monotonic() - started > args.deadline_seconds:
                    cap.release()
                    raise RuntimeError('RGB decode deadline')
                if frames in snapshot_ids:
                    thumb = cv2.resize(image, (400, 225), interpolation=cv2.INTER_AREA)
                    cv2.putText(thumb, 'frame %d' % frames, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 255), 2)
                    snapshots.append(thumb)
                frames += 1
                shape = list(image.shape)
                unique.add(hashlib.sha256(image.tobytes()).hexdigest())
            cap.release()
            probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                '-show_frames', '-show_entries', 'frame=best_effort_timestamp_time', '-of', 'json', str(video)],
                text=True, timeout=10))
            pts = np.asarray([float(x['best_effort_timestamp_time']) for x in probe['frames']])
            ts = np.asarray([x['ts'] for x in pressure])
            ids = np.asarray([x['frame_index'] for x in pressure])
            grid = np.load(args.labels / 'raw' / record / 'pressure_grids.npz', allow_pickle=False)
            temporal = {}
            for side in ('left', 'right'):
                values = grid[side + '_pressure_grid']
                valid = np.isfinite(values)
                delta = np.abs(np.nan_to_num(values[1:]) - np.nan_to_num(values[:-1]))
                paired = valid[1:] & valid[:-1]
                temporal[side] = dict(measured_cells=int(valid[0].sum()),
                    finite_nonzero_fraction=float((values[valid] != 0).mean()),
                    mean_absolute_frame_change=float(delta[paired].mean()),
                    semantics='normalized sensor grid; tactile/bend channel separation not independently audited')
            grid.close()
            agree = frames == len(pressure) and ids.tolist() == list(range(frames)) and len(pts) == frames
            drift = float(np.max(np.abs((ts - ts[0]) - (pts - pts[0])))) if agree else None
            result['records'].append(dict(record=record, bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
                decoded_frames=frames, distinct_decoded_frames=len(unique), fps=fps, shape=shape,
                pressure_clock=clock(pressure), video_pts_seconds=pts.tolist(),
                frame_identity_consistent=agree, relative_timestamp_max_deviation_seconds=drift,
                synchronized_qualified=False, sensor_grid_temporal=temporal,
                snapshot_frame_ids=snapshot_ids))
            if args.contact_sheet and len(snapshots) == 3:
                title = np.full((34, 1200, 3), 245, dtype='uint8')
                cv2.putText(title, record + ' (' + str(frames) + ' frames)', (8, 24), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 1)
                sheet_rows.append(np.concatenate((title, np.concatenate(snapshots, axis=1)), axis=0))
            print('Audited RGB ' + record, flush=True)
        if sheet_rows:
            sheet = args.output / 'fixed-temporal-contact-sheet.jpg'
            if not cv2.imwrite(str(sheet), np.concatenate(sheet_rows, axis=0)):
                raise RuntimeError('contact sheet write failed')
            result['contact_sheet'] = dict(file=sheet.name, sha256=hashlib.sha256(sheet.read_bytes()).hexdigest(),
                selection='fixed 10%,50%,90% original-frame positions; no hand/contact/sensor label selection')
        result['status'] = 'COMPLETED_RGB_CLOCK_AUDIT'
    except Exception as exc:
        result['status'], result['error'] = 'FAILED', str(exc)
        raise
    finally:
        result.update(elapsed_seconds=time.monotonic() - started, transferred_bytes=total)
        (args.output / 'manifest.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        print(json.dumps({k: v for k, v in result.items() if k not in ('records', 'attempts')}, indent=2))


if __name__ == '__main__':
    main()
