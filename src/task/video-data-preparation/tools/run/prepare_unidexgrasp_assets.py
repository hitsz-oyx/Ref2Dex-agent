#!/usr/bin/env python3
"""Fetch bounded official policy inputs and prepare an object-disjoint pilot."""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import threading
import time
import urllib.request
import zipfile

PIN = '6c9e710b04769bf5d9765e3233820101cb3d718f'
BASE = 'https://mirrors.pku.edu.cn/files/dl-release/UniDexGrasp_CVPR2023/dexgrasp_policy/assets/'
PACKAGES = {
    'meshdatav3_scaled.tar.xz': (BASE+'meshdatav3_scaled.tar.xz', 441222940),
    'meshdatav3_pc_feat.zip': (BASE+'meshdatav3_pc_feat.zip', 26844762),
    'datasetv4.1_posedata.npy': (
        'https://drive.usercontent.google.com/download?id=1DajtOFyTPC5YhsO-Fd3Gv17x7eAysI1b&export=download&confirm=t',
        140597661),
}
SCALES = (.06, .08, .10, .12, .15)


def write_json(path, value):
    partial = path.with_suffix(path.suffix+'.partial')
    partial.write_text(json.dumps(value, indent=2)+'\n')
    partial.replace(path)


def safe_name(name):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or '\\' in name:
        raise ValueError('unsafe archive/source path: '+name)
    return path


def download(url, dest, size, deadline, workers):
    """Resume verified-length HTTP ranges; publish only when every range arrived."""
    side = dest.with_suffix(dest.suffix+'.download.json')
    chunk_size = 8*2**20
    count = (size+chunk_size-1)//chunk_size
    if side.exists():
        state = json.loads(side.read_text())
        if (state['url'], state['bytes'], state['chunk_size']) != (url, size, chunk_size):
            raise ValueError('resume input changed')
    else:
        state = dict(url=url, bytes=size, chunk_size=chunk_size, completed=[], status='DOWNLOADING')
    if dest.exists():
        if dest.stat().st_size != size or state['status'] != 'COMPLETE':
            raise ValueError('unverified published download')
        return state
    partial = dest.with_suffix(dest.suffix+'.partial')
    if state['completed'] and (not partial.exists() or partial.stat().st_size != size):
        raise ValueError('resume ranges missing')
    fd = os.open(str(partial), os.O_CREAT | os.O_RDWR, 0o644)
    os.ftruncate(fd, size)
    lock = threading.Lock()

    def one(index):
        lo = index*chunk_size
        hi = min(size, lo+chunk_size)-1
        for attempt in range(3):
            if time.monotonic() >= deadline:
                raise TimeoutError('download wall budget')
            try:
                req = urllib.request.Request(url, headers={'Range': 'bytes=%d-%d' % (lo, hi)})
                with urllib.request.urlopen(req, timeout=20) as response:
                    expected = 'bytes %d-%d/%d' % (lo, hi, size)
                    if response.status != 206 or response.headers.get('Content-Range') != expected:
                        raise ValueError('server did not honor exact bounded range')
                    with lock:
                        identity = response.headers.get('ETag') or response.headers.get('Last-Modified')
                        if state.get('remote_identity', identity) != identity:
                            raise ValueError('remote source changed during resume')
                        state['remote_identity'] = identity
                    data = bytearray()
                    while len(data) <= hi-lo:
                        if time.monotonic() >= deadline:
                            raise TimeoutError('download wall budget')
                        part = response.read(min(2**20, hi-lo+1-len(data)))
                        if not part:
                            break
                        data.extend(part)
                    if len(data) != hi-lo+1:
                        raise IOError('incomplete range')
                offset = 0
                while offset < len(data):
                    offset += os.pwrite(fd, data[offset:], lo+offset)
                with lock:
                    state['completed'].append(index)
                    write_json(side, state)
                    print(dest.name, 'ranges', len(state['completed']), '/', count, flush=True)
                return
            except (OSError, TimeoutError):
                if attempt == 2:
                    raise
        raise RuntimeError('unreachable download retry')

    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(one, i) for i in range(count) if i not in state['completed']]
            for future in as_completed(futures):
                future.result()
        os.fsync(fd)
    finally:
        os.close(fd)
    partial.replace(dest)
    state['status'] = 'COMPLETE'
    write_json(side, state)
    return state


def split_file(root, name, deadline):
    path = root/name
    if not path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError('input preparation budget')
        url = 'https://raw.githubusercontent.com/PKU-EPIC/UniDexGrasp2/'+PIN+'/dexgrasp/cfg/'+name
        with urllib.request.urlopen(url, timeout=20) as response:
            body = response.read(1024*1024+1)
        if len(body) > 1024*1024:
            raise ValueError('oversized split file')
        path.write_bytes(body)
    # Upstream files are Python-style dictionary entries without enclosing braces.
    objects = ast.literal_eval('{'+path.read_text()+'}')
    for code, scales in objects.items():
        if len(safe_name(code).parts) != 2 or not scales or any(s not in SCALES for s in scales):
            raise ValueError('invalid official object/scale')
    return objects


def prepare(args):
    import numpy as np
    begin = time.monotonic()
    deadline = begin+args.seconds
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root/'manifest.json').exists():
        raise ValueError('prepared pilot is immutable; use its manifest or a fresh output')
    inputs = root/'inputs'
    inputs.mkdir(exist_ok=True)
    train = split_file(inputs, 'train_set.yaml', deadline)
    seen = split_file(inputs, 'test_set_seen_cat.yaml', deadline)
    unseen = split_file(inputs, 'test_set_unseen_cat.yaml', deadline)
    if set(train) & (set(seen) | set(unseen)) or set(seen) & set(unseen):
        raise ValueError('official train/test objects overlap')
    if (len(train), len(seen), len(unseen)) != (3200, 141, 100):
        raise ValueError('official policy split coverage changed')
    chosen = []
    categories = set()
    for code in sorted(train):
        category = code.rsplit('-', 1)[0]
        if category not in categories:
            chosen.append(dict(object_code=code, scale=float(train[code][0]), official_split='TRAIN',
                               split='dev' if (len(chosen)+1) % 5 == 0 else 'train'))
            categories.add(category)
        if len(chosen) == args.objects:
            break
    if len(chosen) != args.objects:
        raise ValueError('not enough distinct official train categories')
    write_json(root/'selection.json', dict(objects=chosen, selection='sorted first distinct categories; no rollout outcomes',
                                          official_counts=dict(train=len(train),test_seen=len(seen),test_unseen=len(unseen))))
    downloads = []
    for name, (url, size) in PACKAGES.items():
        downloads.append(download(url, inputs/name, size, deadline, args.workers))

    assets = root/'assets'
    if assets.exists():
        # Preparation resumes only task-owned unpublished files; existing files must agree.
        pass
    else:
        assets.mkdir()
        for name in ('mjcf', 'stls', 'textures'):
            shutil.copytree(args.vendor/'assets'/name, assets/name)
    selected = {r['object_code'] for r in chosen}
    extracted = 0
    scanned = 0

    def publish(name, data):
        nonlocal extracted
        path = assets/safe_name(name)
        extracted += len(data)
        if extracted > 128*2**20:
            raise ValueError('selected asset budget')
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if path.read_bytes() != data:
                raise ValueError('resume asset changed')
        else:
            path.write_bytes(data)

    with tarfile.open(inputs/'meshdatav3_scaled.tar.xz', mode='r|xz') as archive:
        for entry in archive:
            if time.monotonic() >= deadline:
                raise TimeoutError('archive scan budget')
            path = safe_name(entry.name)
            scanned += entry.size
            if scanned > 8*2**30:
                raise ValueError('archive expanded beyond declared scan budget')
            parts = path.parts
            if not parts or parts[0] != 'meshdatav3_scaled':
                continue
            if len(parts) >= 4 and '/'.join(parts[1:3]) in selected and entry.isfile():
                if entry.size > 16*2**20:
                    raise ValueError('oversized selected mesh asset')
                publish(str(path), archive.extractfile(entry).read())
    with zipfile.ZipFile(inputs/'meshdatav3_pc_feat.zip') as archive:
        for row in chosen:
            name = 'meshdatav3_pc_feat/'+row['object_code']+'/pc_feat_%03d.npy' % round(row['scale']*100)
            if archive.getinfo(name).file_size > 4096:
                raise ValueError('unexpected shape feature size')
            publish(name, archive.read(name))
    # This is the official trusted-source pickle provided by the authors, not arbitrary user input.
    all_priors = np.load(inputs/'datasetv4.1_posedata.npy', allow_pickle=True).item()
    priors = {}
    shapes = dict(target_qpos=22, target_hand_pos=3, target_hand_rot=4, object_euler_xy=2, object_init_z=1)
    for row in chosen:
        code, scale = row['object_code'], row['scale']
        values = {k: np.asarray(all_priors[code][scale][k], dtype=np.float32) for k in shapes}
        count = len(values['object_init_z'])
        if not count or any(v.shape != (count, shapes[k]) or not np.isfinite(v).all() for k, v in values.items()):
            raise ValueError('official initialization prior contract mismatch: '+code)
        if np.max(abs(np.linalg.norm(values['target_hand_rot'], axis=1)-1)) > 1e-4:
            raise ValueError('official target rotation is not unit')
        mesh = assets/'meshdatav3_scaled'/code/'coacd'
        for name in ('coacd_%03d.urdf', 'decomposed_%03d.obj'):
            if not (mesh/(name % round(scale*100))).is_file():
                raise ValueError('selected mesh/scale missing: '+code)
        feature = np.load(assets/'meshdatav3_pc_feat'/code/('pc_feat_%03d.npy' % round(scale*100)))
        if feature.shape != (64,) or not np.isfinite(feature).all():
            raise ValueError('official 64D feature contract mismatch')
        priors[code] = {scale: values}
        row['initialization_priors'] = count
    np.save(assets/'datasetv4.1_posedata.npy', priors)
    manifest = dict(status='COMPLETE', upstream_commit=PIN, objects=chosen, asset_root=str(assets),
                    priors_source=PACKAGES['datasetv4.1_posedata.npy'][0], official_split_counts=dict(train=3200,test_seen=141,test_unseen=100),
                    downloads=downloads, compressed_input_bytes=sum(s for _,s in PACKAGES.values()),
                    selected_assets_bytes=sum(p.stat().st_size for p in assets.rglob('*') if p.is_file()),
                    scanned_mesh_expanded_bytes=scanned, elapsed_s=time.monotonic()-begin, training_allowed=False)
    write_json(root/'manifest.json', manifest)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--vendor', type=Path, required=True)
    parser.add_argument('--objects', type=int, default=20)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--seconds', type=int, default=1800)
    args = parser.parse_args()
    if not 5 <= args.objects <= 50 or not 1 <= args.workers <= 8 or not 60 <= args.seconds <= 3600:
        raise ValueError('bounded pilot arguments required')
    args.vendor = args.vendor.resolve()
    prepare(args)
