#!/usr/bin/env python3
"""Ten task-separated TRAIN label bundles; no RGB/model/full-corpus download."""
import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from urllib.parse import quote

import requests
from sample_egotouch_labels import REVISION, DATASET, verify_bytes

NAMES = ('jq_pressure.json', 'wilor_hands.json', 'pressure_grids.npz', 'manual_contact_annotation.json')


def select(files, train):
    scenarios = ('Home', 'Office', 'Outdoor', 'Retail', 'Workbench')
    chosen = []
    for scenario in scenarios:
        candidates = []
        for path in train:
            record = '/'.join(path.split('/')[-3:])[:-5]
            if record.split('/')[0] != scenario or not all(record + '/' + n in files for n in NAMES):
                continue
            size = sum(files[record + '/' + n]['size'] for n in NAMES)
            pressure_size = files[record + '/jq_pressure.json']['size']
            if 400000 <= pressure_size <= 2000000 and size <= 5000000:
                candidates.append((size, record))
        tasks = set()
        for _, record in sorted(candidates):
            task = '/'.join(record.split('/')[:2])
            if task in tasks:
                continue
            tasks.add(task)
            chosen.append(dict(record=record, split='fit' if len(tasks) == 1 else 'held_task'))
            if len(tasks) == 2:
                break
        if len(tasks) != 2:
            raise ValueError('two distinct eligible tasks absent in ' + scenario)
    return chosen


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inventory', type=Path, required=True)
    p.add_argument('--split', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    inventory = json.loads(args.inventory.read_text())
    files = {x['path']: x for x in inventory if x['type'] == 'file'}
    chosen = select(files, json.loads(args.split.read_text())['train'])
    expected = sum(files[e['record'] + '/' + n]['size'] for e in chosen for n in NAMES)
    if expected > 16 * 1024 ** 2:
        raise ValueError('selected label bytes exceed16MiB before any download')
    args.output.mkdir(parents=True, exist_ok=False)
    started, total, mirror_failed = time.monotonic(), 0, False
    result = dict(schema='ref2dex.egotouch-diverse-acquisition.v1', status='RUNNING',
        dataset=DATASET, revision=REVISION, training_allowed=False, selection=chosen,
        expected_bytes=expected, files=[], attempts=[],
        original_split='all from official TRAIN; held_task is local development only',
        modelscope_discovery='2026-10-10 initial primary-source search found no same release',
        inventory_sha256=hashlib.sha256(args.inventory.read_bytes()).hexdigest(),
        split_sha256=hashlib.sha256(args.split.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
    session = requests.Session()
    session.trust_env = False
    try:
        for record in chosen:
            for name in NAMES:
                entry = files[record['record'] + '/' + name]
                routes = [] if mirror_failed else [('hf-mirror.com', {}, 2)]
                routes.append(('huggingface.co', {'http': 'http://127.0.0.1:7897', 'https': 'http://127.0.0.1:7897'}, 15))
                verified = False
                for host, proxies, timeout in routes:
                    attempt = dict(path=entry['path'], host=host)
                    result['attempts'].append(attempt)
                    try:
                        if time.monotonic() - started > 240:
                            raise RuntimeError('acquisition240s deadline')
                        url = 'https://' + host + '/datasets/' + DATASET + '/resolve/' + REVISION + '/' + quote(entry['path'])
                        chunks, size = [], 0
                        with session.get(url, proxies=proxies, timeout=timeout, stream=True) as response:
                            response.raise_for_status()
                            for chunk in response.iter_content(65536):
                                total += len(chunk)
                                size += len(chunk)
                                if total > 16 * 1024 ** 2 or size > entry['size'] or time.monotonic() - started > 240:
                                    raise RuntimeError('download byte/time cap')
                                chunks.append(chunk)
                        data = b''.join(chunks)
                        verify_bytes(data, entry)
                        output = args.output / 'raw' / entry['path']
                        output.parent.mkdir(parents=True, exist_ok=True)
                        output.write_bytes(data)
                        result['files'].append(dict(path=entry['path'], bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
                        attempt['status'] = 'VERIFIED'
                        verified = True
                        break
                    except requests.RequestException as exc:
                        attempt['status'] = type(exc).__name__
                        if host == 'hf-mirror.com':
                            mirror_failed = True
                if not verified:
                    raise RuntimeError('all routes failed: ' + entry['path'])
            print('Verified ' + record['record'], flush=True)
        result['status'] = 'COMPLETED_DIVERSE_RAW_SAMPLE'
    except Exception as exc:
        result['status'], result['error'] = 'FAILED', str(exc)
        raise
    finally:
        result.update(transferred_bytes=total, elapsed_seconds=time.monotonic() - started)
        (args.output / 'manifest.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        print(json.dumps({k: v for k, v in result.items() if k not in ('files', 'attempts')}, indent=2))


if __name__ == '__main__':
    main()
