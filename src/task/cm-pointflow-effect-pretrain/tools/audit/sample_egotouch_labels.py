#!/usr/bin/env python3
"""Acquire two small TRAIN label bundles, with pinned hashes and byte/time caps.

No RGB, model, full dataset, or training manifest is downloaded. Before using
this tool check ModelScope for the same release; use the domestic HF mirror
first and the project proxy as fallback. Inventory must be from REVISION.
"""
import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from urllib.parse import quote

import numpy as np
import requests

REVISION = 'cfdbb0ac31cc2af4247943820aa250575e7e6637'
DATASET = 'zhouzhoujy/EgoTouch'
LABELS = ('jq_pressure.json', 'wilor_hands.json', 'vive_poses.json',
          'pressure_grids.npz', 'manual_contact_annotation.json')


def select(inventory, splits):
    files = {x['path']: x for x in inventory if x['type'] == 'file'}
    candidates = []
    for path in splits['train']:
        record = '/'.join(path.split('/')[-3:])
        if not record.endswith('.hdf5'):
            raise ValueError('unexpected split recording path')
        record = record[:-5]
        if all(record + '/' + n in files for n in LABELS):
            candidates.append((sum(files[record + '/' + n]['size'] for n in LABELS), record))
    selected, tasks = [], set()
    for _, record in sorted(candidates):
        task = '/'.join(record.split('/')[:2])
        if task not in tasks:
            selected.append(record)
            tasks.add(task)
        if len(selected) == 2:
            break
    if len(selected) != 2:
        raise ValueError('two distinct train tasks with all required labels not found')
    return [files[record + '/' + name] for record in selected for name in LABELS]


def verify_bytes(data, entry):
    if len(data) != entry['size']:
        raise ValueError('download length differs from pinned inventory')
    if 'lfs' in entry:
        actual = hashlib.sha256(data).hexdigest()
        expected = entry['lfs']['oid']
    else:
        actual = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        expected = entry['oid']
    if actual != expected:
        raise ValueError('download checksum differs from pinned inventory')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inventory', type=Path, required=True)
    p.add_argument('--split', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-bytes', type=int, default=16 * 1024 * 1024)
    p.add_argument('--max-seconds', type=float, default=180)
    args = p.parse_args()
    entries = select(json.loads(args.inventory.read_text()), json.loads(args.split.read_text()))
    if sum(x['size'] for x in entries) > args.max_bytes:
        raise ValueError('selected label files exceed the byte cap')
    args.output.mkdir(parents=True, exist_ok=False)
    started, transferred = time.monotonic(), 0
    result = dict(dataset=DATASET, revision=REVISION, status='RUNNING',
                  training_allowed=False, split='train',
                  selection='smallest complete bundles from two distinct train tasks; biased schema sample',
                  git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  files=[], attempts=[], max_bytes=args.max_bytes, max_seconds=args.max_seconds,
                  input_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in (args.inventory, args.split)},
                  limits=['no video synchronization verified', 'no object pose labels',
                          'sample size and selection cannot support a corpus-wide quality claim'])
    session = requests.Session()
    session.trust_env = False
    try:
        for entry in entries:
            data = None
            for host, proxies in [('hf-mirror.com', {}), ('huggingface.co',
                                   {'http': 'http://127.0.0.1:7897', 'https': 'http://127.0.0.1:7897'})]:
                url = 'https://' + host + '/datasets/' + DATASET + '/resolve/' + REVISION + '/' + quote(entry['path'])
                attempt = dict(path=entry['path'], host=host)
                result['attempts'].append(attempt)
                try:
                    remaining = args.max_seconds - (time.monotonic() - started)
                    if remaining <= 0:
                        raise TimeoutError('label acquisition wall-time cap reached')
                    chunks = []
                    with session.get(url, proxies=proxies, timeout=min(12, remaining), stream=True) as response:
                        attempt['http_status'] = response.status_code
                        response.raise_for_status()
                        size = 0
                        for chunk in response.iter_content(65536):
                            size += len(chunk)
                            transferred += len(chunk)
                            if size > entry['size'] or transferred > args.max_bytes:
                                raise ValueError('download byte cap reached')
                            if time.monotonic() - started > args.max_seconds:
                                raise TimeoutError('label acquisition wall-time cap reached')
                            chunks.append(chunk)
                    candidate = b''.join(chunks)
                    verify_bytes(candidate, entry)
                    data = candidate
                    attempt['status'] = 'VERIFIED'
                    break
                except requests.RequestException as exc:
                    attempt['status'] = type(exc).__name__
            if data is None:
                raise RuntimeError('both download routes failed for ' + entry['path'])
            path = args.output / 'raw' / entry['path']
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            record = dict(path=entry['path'], bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
            if path.suffix == '.npz':
                with np.load(path, allow_pickle=False) as arrays:
                    record['arrays'] = {k: dict(shape=list(arrays[k].shape), dtype=str(arrays[k].dtype),
                                                finite_fraction=float(np.isfinite(arrays[k]).mean()))
                                        for k in arrays}
            else:
                text = data.decode('utf-8')
                try:
                    obj = json.loads(text)
                    record['format'] = 'json'
                except json.JSONDecodeError:
                    obj = [json.loads(line) for line in text.splitlines() if line.strip()]
                    record['format'] = 'jsonl'
                record['top_level_type'] = type(obj).__name__
                record['top_level_count'] = len(obj) if isinstance(obj, (dict, list)) else None
                record['example'] = repr(obj[0] if isinstance(obj, list) and obj else obj)[:1000]
            result['files'].append(record)
        result['status'] = 'COMPLETED_SCHEMA_SAMPLE'
    except Exception as exc:
        result['status'] = 'FAILED'
        result['error'] = str(exc)
        raise
    finally:
        result['elapsed_seconds'] = time.monotonic() - started
        result['transferred_bytes'] = transferred
        (args.output / 'manifest.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        print(json.dumps({k: v for k, v in result.items() if k not in ('files', 'attempts')}, indent=2))


if __name__ == '__main__':
    main()
