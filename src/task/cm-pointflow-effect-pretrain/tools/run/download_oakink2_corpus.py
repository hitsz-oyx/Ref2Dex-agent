#!/usr/bin/env python3
"""Acquire pinned corpus with verified hardlink reuse of the earlier audit."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--reuse-dir', type=Path, required=True)
    p.add_argument('--seconds', type=int, default=7200)
    args = p.parse_args()
    root, reuse = args.run_dir.resolve(), args.reuse_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    for name in ('dataset_info.json', 'anno_tree.json', 'root_tree.json'):
        if not (root / name).exists():
            shutil.copyfile(reuse / name, root / name)
    revision = json.loads((root / 'dataset_info.json').read_text())['sha']
    ann = sorted(json.loads((root / 'anno_tree.json').read_text()), key=lambda x: x['path'])
    assets = [x for x in json.loads((root / 'root_tree.json').read_text()) if x['path'] in
              {'object_raw.tar', 'object_repair.tar', 'object_affordance.tar', 'program.tar', 'program_extension.tar'}]
    assert len(ann) == 627 and len(assets) == 5
    jobs = assets + ann
    expected = sum(x['size'] for x in jobs)
    assert expected < 40 * 1024**3
    out = root / 'download'
    out.mkdir(exist_ok=True)
    if not (root / 'dataset').exists():
        (root / 'dataset').symlink_to(reuse / 'dataset', target_is_directory=True)
    missing = sum(x['size'] for x in jobs if not (out / x['path']).exists() and not (reuse / 'download' / x['path']).exists())
    if shutil.disk_usage(root).free < missing + 10 * 1024**3:
        raise ValueError('insufficient free disk reserve')
    start = time.monotonic()
    record = dict(status='RUNNING', revision=revision, selection=[x['path'] for x in ann],
                  files={}, expected_bytes=expected, budget_seconds=args.seconds,
                  script_sha256=digest(Path(__file__)),
                  git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
    manifest = root / 'download_manifest.json'
    if manifest.exists():
        old = json.loads(manifest.read_text())
        if old['revision'] != revision: raise ValueError('revision drift')
        shutil.copyfile(manifest, root / ('download_manifest.previous.%d.json' % time.time_ns()))

    def fetch(item):
        name = item['path']
        path, cached = out / name, reuse / 'download' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists() and cached.exists():
            if digest(cached) != item['lfs']['oid']: raise ValueError('reused checksum mismatch: ' + name)
            os.link(cached, path)
        if not path.exists():
            remaining = args.seconds - (time.monotonic() - start)
            if remaining <= 0: raise TimeoutError('acquisition deadline')
            part = path.with_name(path.name + '.part')
            url = 'https://huggingface.co/datasets/kelvin34501/OakInk-v2/resolve/' + revision + '/' + name
            for attempt in range(3):
                remaining = args.seconds - (time.monotonic() - start)
                if remaining <= 0: raise TimeoutError('acquisition deadline')
                result = subprocess.run(['curl', '--fail', '--silent', '--show-error', '--location', '--retry', '2',
                                         '--connect-timeout', '15', '--max-time', str(min(600, int(remaining))),
                                         '--max-filesize', str(item['size']), '--output', str(part), url],
                                        timeout=remaining, capture_output=True, text=True)
                if result.returncode == 0 and part.stat().st_size == item['size'] and digest(part) == item['lfs']['oid']:
                    break
                if attempt == 2: raise RuntimeError(name + ': ' + result.stderr[-500:])
            if part.stat().st_size != item['size'] or digest(part) != item['lfs']['oid']:
                raise ValueError('download checksum mismatch: ' + name)
            part.rename(path)
        if path.stat().st_size != item['size'] or digest(path) != item['lfs']['oid']:
            raise ValueError('existing checksum mismatch: ' + name)
        return name, dict(size=item['size'], sha256=item['lfs']['oid'])

    try:
        with ThreadPoolExecutor(max_workers=6) as pool:
            errors = []
            for future in as_completed([pool.submit(fetch, job) for job in jobs]):
                try:
                    name, info = future.result()
                except Exception as exc:
                    errors.append(str(exc))
                    continue
                record['files'][name] = info
                record['elapsed_seconds'] = time.monotonic() - start
                manifest.write_text(json.dumps(record, indent=2) + '\n')
                print(json.dumps({'completed': len(record['files']), 'total': len(jobs), 'elapsed_seconds': record['elapsed_seconds']}), flush=True)
        if errors:
            raise RuntimeError(str(len(errors)) + ' acquisition errors: ' + '; '.join(errors[:5]))
        record['status'] = 'COMPLETED'
    except BaseException as exc:
        record.update(status='FAILED', error=str(exc))
        raise
    finally:
        record['elapsed_seconds'] = time.monotonic() - start
        manifest.write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()
