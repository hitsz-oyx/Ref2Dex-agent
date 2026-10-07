"""Bounded, CRC-checked native EgoDex test subset from a remote ZIP.

Probe the domestic mirror first, then use the user-authorized proxy. Only
selected paired HDF5/MP4 members are transferred; no full archive is materialized.
"""
import argparse
import hashlib
import io
import json
import re
import time
import zipfile
from pathlib import Path

import requests

OFFICIAL = 'https://ml-site.cdn-apple.com/datasets/egodex/test.zip'
MIRROR = 'https://hf-mirror.com/datasets/zhenyuxie-zhzh/egodex/resolve/main/test.zip'
PROXY = 'http://127.0.0.1:7897'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


class RemoteZip(io.RawIOBase):
    def __init__(self, url, session, cap, deadline):
        self.url, self.session, self.cap, self.deadline = url, session, cap, deadline
        r = session.head(url, allow_redirects=True, timeout=25)
        r.raise_for_status()
        self.size = int(r.headers['Content-Length'])
        self.etag = r.headers.get('ETag')
        self.pos, self.transferred = 0, 0

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset + (self.pos if whence == 1 else self.size if whence == 2 else 0)
        if self.pos < 0:
            raise ValueError('negative ZIP seek')
        return self.pos

    def read(self, length=-1):
        length = self.size - self.pos if length < 0 else min(length, self.size - self.pos)
        if length <= 0:
            return b''
        if self.transferred + length > self.cap or time.monotonic() > self.deadline:
            raise RuntimeError('transfer/time budget reached')
        start, end = self.pos, self.pos + length - 1
        # A range-specific URL avoids intermediary caches serving a different range.
        url = self.url + ('&' if '?' in self.url else '?') + f'ref5_range={start}-{end}'
        r = self.session.get(url, headers={'Range': f'bytes={start}-{end}'},
                             timeout=(20, 90), stream=True)
        try:
            if r.status_code != 206 or r.headers.get('Content-Range') != f'bytes {start}-{end}/{self.size}':
                raise RuntimeError('server did not honor the exact byte range')
            if self.etag and r.headers.get('ETag') != self.etag:
                raise RuntimeError('remote ZIP identity changed')
            chunks, actual = [], 0
            for chunk in r.iter_content(1 << 20):
                actual += len(chunk)
                self.transferred += len(chunk)
                if actual > length or self.transferred > self.cap or time.monotonic() > self.deadline:
                    raise RuntimeError('transfer/time budget reached')
                chunks.append(chunk)
            if actual != length:
                raise RuntimeError('truncated byte range')
        finally:
            r.close()
        self.pos += length
        return b''.join(chunks)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--clips', type=int, default=20)
    p.add_argument('--seconds', type=int, default=1800)
    p.add_argument('--max-gib', type=float, default=2)
    p.add_argument('--selection-file', type=Path,
                   help='JSON list of exact HDF5 archive paths; at most50 unique pairs')
    p.add_argument('--metadata-only', action='store_true',
                   help='Acquire HDF5 only for object metadata curation; no paired clip claim')
    a = p.parse_args()
    if not 1 <= a.clips <= 50:
        p.error('clips must be in [1,50]')
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'download_manifest.json').exists():
        raise FileExistsError('use a fresh output directory; never overwrite a manifest')
    manifest = {'status': 'RUNNING', 'split': 'official_test', 'training_allowed': False,
                'files': [], 'transport_probes': [], 'max_bytes': int(a.max_gib * 2**30),
                'metadata_only': a.metadata_only, 'paired_video_available': not a.metadata_only}
    start = time.monotonic()
    direct = requests.Session()
    direct.trust_env = False
    try:
        r = direct.head(MIRROR, timeout=15, allow_redirects=True)
        manifest['transport_probes'].append({'url': MIRROR, 'status': r.status_code})
        mirror_ok = r.status_code == 200 and int(r.headers.get('Content-Length', 0)) == 17304529397
    except requests.RequestException as e:
        manifest['transport_probes'].append({'url': MIRROR, 'error': str(e)})
        mirror_ok = False
    session = direct if mirror_ok else requests.Session()
    session.trust_env = False
    if not mirror_ok:
        session.proxies = {'http': PROXY, 'https': PROXY}
    url = MIRROR if mirror_ok else OFFICIAL
    manifest['url'], manifest['proxy_used'] = url, not mirror_ok
    remote = None
    try:
        remote = RemoteZip(url, session, manifest['max_bytes'], start + a.seconds)
        manifest.update(archive_size=remote.size, archive_etag=remote.etag)
        with zipfile.ZipFile(remote) as z:
            names = set(z.namelist())
            (out / 'zip_inventory.json').write_text(json.dumps([
                {'name': i.filename, 'bytes': i.file_size, 'compressed_bytes': i.compress_size,
                 'crc32': i.CRC} for i in z.infolist()], indent=2))
            # Candidate manipulation tasks; per-clip metadata and visual review
            # must still exclude articulated/deformable targets before 6DoF use.
            tasks = ['basic_pick_place', 'vertical_pick_place', 'stack_unstack',
                     'sort_objects', 'add_remove_lid', 'clean_table', 'put_object_in_container']
            pools = {}
            for name in sorted(names):
                if name.endswith('.hdf5') and name[:-5] + '.mp4' in names:
                    task = name.split('/')[-2]
                    if task in tasks:
                        pools.setdefault(task, []).append(name)
            selected = []
            while pools and len(selected) < a.clips:
                for task in tasks:
                    if task in pools:
                        selected.append(pools[task].pop(0))
                        if not pools[task]:
                            del pools[task]
                        if len(selected) == a.clips:
                            break
            if len(selected) != a.clips:
                raise RuntimeError(f'only {len(selected)} matching pairs; inspect zip_inventory.json')
            if a.selection_file:
                selected = json.loads(a.selection_file.read_text())
                if (not isinstance(selected,list) or not 1 <= len(selected) <= 50
                        or not all(isinstance(n,str) for n in selected)
                        or len(set(selected)) != len(selected)):
                    raise ValueError('selection must contain1..50 unique archive HDF5 paths')
                if any(not n.endswith('.hdf5') or n not in names or n[:-5]+'.mp4' not in names
                       for n in selected):
                    raise ValueError('selection does not identify existing paired archive members')
                manifest['selection_file_sha256'] = sha(a.selection_file)
            manifest['selection'] = selected
            total = sum(z.getinfo(n).file_size + (0 if a.metadata_only else
                        z.getinfo(n[:-5] + '.mp4').file_size) for n in selected)
            if total > manifest['max_bytes']:
                raise RuntimeError('selected uncompressed files exceed disk budget')
            for count, name in enumerate(selected,1):
                members = (name,) if a.metadata_only else (name,name[:-5]+'.mp4')
                for member in members:
                    # Preserve source identity, exclude path traversal.
                    relative = Path(member)
                    if relative.is_absolute() or '..' in relative.parts:
                        raise ValueError('unsafe archive path')
                    dest = out / 'raw' / relative
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    partial = dest.with_suffix(dest.suffix + '.part')
                    if dest.exists() or partial.exists():
                        raise FileExistsError(dest)
                    with z.open(member) as f, partial.open('xb') as target:
                        for chunk in iter(lambda: f.read(1 << 20), b''):
                            if time.monotonic() > start + a.seconds:
                                raise TimeoutError('download deadline')
                            target.write(chunk)
                    # ZipExtFile verifies CRC at EOF.
                    partial.rename(dest)
                    manifest['files'].append({'member': member, 'bytes': dest.stat().st_size,
                                              'sha256': sha(dest), 'crc32': z.getinfo(member).CRC})
                print(json.dumps({'downloaded_clips': 0 if a.metadata_only else count,
                                  'downloaded_metadata': count if a.metadata_only else 0,
                                  'clip': name, 'transferred_bytes': remote.transferred}), flush=True)
        manifest['status'] = 'COMPLETED'
    except Exception as e:
        manifest.update(status='FAILED', error=repr(e))
        raise
    finally:
        manifest.update(elapsed_s=time.monotonic() - start,
                        transferred_bytes=remote.transferred if remote else 0)
        (out / 'download_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
