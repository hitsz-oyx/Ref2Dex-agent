#!/usr/bin/env python3
"""Find an annotated contact-positive TRAIN bundle and check chest frame identity."""
import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from urllib.parse import quote

import cv2
import requests

from sample_egotouch_labels import REVISION, DATASET, LABELS, verify_bytes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inventory', type=Path, required=True)
    p.add_argument('--split', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    files = {x['path']: x for x in json.loads(args.inventory.read_text()) if x['type'] == 'file'}
    train = json.loads(args.split.read_text())['train']
    names = LABELS + ('chest.mp4',)
    candidates = []
    for path in train:
        record = '/'.join(path.split('/')[-3:])[:-5]
        if not all(record + '/' + name in files for name in names):
            continue
        size = sum(files[record + '/' + name]['size'] for name in names)
        if size > 16 * 1024 ** 2 - 65536:
            continue
        # Favor manipulation task names, but actual annotation must be true.
        priority = 0 if any(word in record for word in ('squeeze', 'pick_up', 'hold', 'grip')) else 1
        candidates.append((priority, size, record))
    args.output.mkdir(parents=True, exist_ok=False)
    result = dict(dataset=DATASET, revision=REVISION, status='RUNNING', training_allowed=False,
                  selection='one candidate per distinct task, manipulation names first, then smallest bundle',
                  split='train', annotations=[], files=[], attempts=[],
                  git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  inventory_sha256=hashlib.sha256(args.inventory.read_bytes()).hexdigest(),
                  split_sha256=hashlib.sha256(args.split.read_bytes()).hexdigest())
    started, total = time.monotonic(), 0
    session = requests.Session()
    session.trust_env = False
    mirror_failed = [False]
    def get(entry):
        nonlocal total
        routes = [] if mirror_failed[0] else [('hf-mirror.com', {}, 2)]
        routes.append(('huggingface.co', {'http': 'http://127.0.0.1:7897', 'https': 'http://127.0.0.1:7897'}, 15))
        for host, proxies, timeout in routes:
            attempt = dict(path=entry['path'], host=host)
            result['attempts'].append(attempt)
            try:
                url = 'https://' + host + '/datasets/' + DATASET + '/resolve/' + REVISION + '/' + quote(entry['path'])
                chunks, size = [], 0
                with session.get(url, proxies=proxies, timeout=timeout, stream=True) as response:
                    response.raise_for_status()
                    for chunk in response.iter_content(65536):
                        total += len(chunk)
                        size += len(chunk)
                        if total > 16 * 1024 ** 2 or size > entry['size'] or time.monotonic() - started > 300:
                            raise RuntimeError('contact sample byte/time budget reached')
                        chunks.append(chunk)
                data = b''.join(chunks)
                verify_bytes(data, entry)
                attempt['status'] = 'VERIFIED'
                return data
            except requests.RequestException as exc:
                attempt['status'] = type(exc).__name__
                if host == 'hf-mirror.com':
                    mirror_failed[0] = True
        raise RuntimeError('download routes failed for ' + entry['path'])
    try:
        tasks, selected = set(), None
        for _, _, record in sorted(candidates):
            task = '/'.join(record.split('/')[:2])
            if task in tasks:
                continue
            tasks.add(task)
            entry = files[record + '/manual_contact_annotation.json']
            data = get(entry)
            annotation = json.loads(data)
            positive = annotation.get('left_contact') is True or annotation.get('right_contact') is True
            result['annotations'].append(dict(record=record, positive=positive,
                                             sha256=hashlib.sha256(data).hexdigest()))
            local = args.output / 'annotation-search' / (record.replace('/', '__') + '.json')
            local.parent.mkdir(exist_ok=True)
            local.write_bytes(data)
            if positive:
                selected = record
                break
            if len(tasks) >= 40:
                break
        if selected is None:
            result['status'] = 'NO_POSITIVE_IN_BOUNDED_SEARCH'
            return
        result['selected_record'] = selected
        for name in names:
            entry = files[selected + '/' + name]
            data = get(entry)
            path = args.output / 'raw' / entry['path']
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            result['files'].append(dict(path=entry['path'], bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
        cap = cv2.VideoCapture(str(args.output / 'raw' / selected / 'chest.mp4'))
        fps, frames = cap.get(cv2.CAP_PROP_FPS), 0
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            frames += 1
        cap.release()
        result['chest_video'] = dict(decoded_frames=frames, fps=fps,
                                     synchronized_labels_qualified=False)
        result['status'] = 'COMPLETED_CONTACT_POSITIVE_SAMPLE'
    except Exception as exc:
        result['status'], result['error'] = 'FAILED', str(exc)
        raise
    finally:
        result.update(elapsed_seconds=time.monotonic() - started, transferred_bytes=total)
        (args.output / 'manifest.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        print(json.dumps({k: v for k, v in result.items() if k not in ('files', 'attempts', 'annotations')}, indent=2))


if __name__ == '__main__':
    main()
