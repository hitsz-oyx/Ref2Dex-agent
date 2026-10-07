#!/usr/bin/env python3
"""Bounded official-split GRAB/ARCTIC conversion, or zero-MANO native reindex."""
import sys
sys.dont_write_bytecode = True

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path.insert(0, str(TASK / 'src'))
sys.path.insert(0, str(Path(__file__).parent / 'ref5_data_expansion'))
from oakink_wm.native_splits import arctic_protocol, official_split


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def write_json(path, data):
    temporary = Path(path).with_suffix('.json.part')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def owned_size(path):
    return sum(p.stat().st_size for p in Path(path).rglob('*') if p.is_file() and not p.is_symlink())


def contained(path, parent):
    try:
        Path(path).relative_to(parent)
        return True
    except ValueError:
        return False


def guard_output(output, inputs=()):
    output = Path(output).resolve()
    if not contained(output, (ROOT / 'outputs').resolve()):
        raise ValueError('output must be inside this repository outputs/')
    if output.exists():
        raise FileExistsError(output)
    for source in inputs:
        source = Path(source).resolve()
        if output == source or contained(source, output) or contained(output, source):
            raise ValueError('output and input boundaries overlap')
    if shutil.disk_usage(ROOT).free < 20 * 1024**3:
        raise RuntimeError('less than 20 GiB free disk')
    return output


def check_budget(output, start, seconds, max_gib):
    if time.monotonic() - start >= seconds:
        raise TimeoutError('native conversion deadline')
    if owned_size(output) > max_gib * 1024**3:
        raise RuntimeError('native output size budget exceeded')
    if shutil.disk_usage(output).free < 20 * 1024**3:
        raise RuntimeError('native disk reserve violated')


def gpu_guard(index):
    result = subprocess.check_output(['nvidia-smi', '-i', str(index),
        '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
    if result:
        raise RuntimeError('GPU %s occupied by compute PID(s): %s' % (index, result))
    memory = int(subprocess.check_output(['nvidia-smi', '-i', str(index),
        '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).strip())
    if memory > 512:
        raise RuntimeError('GPU memory exceeds idle guard: %s MiB' % memory)


def code_identity():
    files = [Path(__file__), TASK / 'src/oakink_wm/native_splits.py',
             TASK / 'src/oakink_wm/data.py']
    files += [Path(__file__).parent / 'ref5_data_expansion' / x
              for x in ('prepare_native.py', 'native_data.py')]
    return {str(p.resolve()): sha(p) for p in files}


def manifest_base(args, protocol):
    from native_data import SEMANTICS
    return dict(schema='ref2dex.native-wm30.v1', status='RUNNING', fps=30,
                horizon=24, history=4, hand_order=['right', 'left'],
                hand_semantics=SEMANTICS, units='m', sequences=[], records=[],
                splits={s: [] for s in ('train', 'val', 'test')},
                training_allowed=False, purpose='native3d_official_full',
                base_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                code_sha256=code_identity(), protocol_path=str(args.protocol.resolve()),
                protocol_sha256=sha(args.protocol), split_protocol='GRAB author object holdouts; ARCTIC protocol_p1',
                category_semantics={'1': 'moving_near_hand', '2': 'static_near_hand'},
                program_available=False, device='cpu_metadata' if args.reindex else 'cuda:0',
                physical_gpu=None if args.reindex else args.gpu,
                cpu_reason='reindex immutable native arrays and audit contracts' if args.reindex else None,
                bounds=dict(max_sequences=args.max_sequences, seconds=args.seconds,
                            max_output_gib=args.max_output_gib, min_free_gib=20),
                invocation=list(sys.argv), source_read_only=True)


def add_record(meta, rows_by_split, record, rows, protocol):
    split = official_split(record['source'], record['source_sequence'], protocol)
    record = dict(record, split=split)
    name = record['sequence']
    if name in meta['sequences']:
        raise ValueError('duplicate native sequence ' + name)
    seqidx = len(meta['sequences'])
    meta['sequences'].append(name)
    meta['records'].append(record)
    meta['splits'][split].append(name)
    rows_by_split[split].extend([[seqidx, *map(int, row)] for row in rows])
    return record


def verify_inputs(meta):
    for record in meta['records']:
        if sha(record['source_path']) != record['source_sha256']:
            raise ValueError('source changed during conversion: ' + record['source_path'])
        for path, digest in record.get('dependencies', {}).items():
            if sha(path) != digest:
                raise ValueError('dependency changed during conversion: ' + path)
    for path, digest in meta['code_sha256'].items():
        if sha(path) != digest:
            raise ValueError('implementation changed during conversion: ' + path)
    if sha(meta['protocol_path']) != meta['protocol_sha256']:
        raise ValueError('official protocol changed during conversion')
    for path, digest in meta.get('mano_sha256', {}).items():
        if sha(path) != digest:
            raise ValueError('MANO dependency changed during conversion')


def reindex(args, meta, out, rows_by_split, protocol):
    import numpy as np
    pack = args.reindex.resolve()
    original = json.loads((pack / 'processed/manifest.json').read_text())
    if original['schema'] != 'ref2dex.native-wm30.v1' or original['status'] != 'COMPLETED':
        raise ValueError('reindex input must be a completed native pack')
    if original.get('training_allowed') is not True:
        raise ValueError('reindex input does not permit training')
    if len(original['sequences']) > args.max_sequences:
        raise ValueError('input exceeds sequence budget')
    audit = json.loads((pack / 'audit.json').read_text())
    if audit.get('verdict') != 'ENGINEERING_PASS':
        raise ValueError('reindex source engineering audit is required')
    meta['reindex_source'] = str(pack)
    meta['reindex_source_manifest_sha256'] = sha(pack / 'processed/manifest.json')
    meta['reindex_source_audit_sha256'] = sha(pack / 'audit.json')
    meta['selection'] = 'all previously converted sequences, official split reindex without MANO reconstruction'
    indices = [np.load(pack / f'processed/index_{s}.npy') for s in ('train', 'val', 'test')]
    allrows = np.concatenate(indices)
    if len({tuple(map(int, row[:3])) for row in allrows}) != len(allrows):
        raise ValueError('duplicate input window identity')
    (out / 'canonical').symlink_to(pack / 'processed/canonical', target_is_directory=True)
    for sid, record in enumerate(original['records']):
        if record['sequence'] != original['sequences'][sid]:
            raise ValueError('source record and sequence identities disagree')
        dest = out / 'sequences' / record['sequence']
        dest.mkdir()
        arrays_sha = {}
        for source in sorted((pack / 'processed/sequences' / record['sequence']).glob('*.npy')):
            arrays_sha[source.name] = sha(source)
            (dest / source.name).symlink_to(source.resolve())
        rows = allrows[allrows[:, 0] == sid, 1:]
        # Preserve all supervised inputs; classification is independently checked below.
        record = add_record(meta, rows_by_split, dict(record, array_sha256=arrays_sha), rows, protocol)
        write_json(dest / 'meta.json', record)


def convert(args, meta, out, rows_by_split, protocol, start):
    import numpy as np
    import torch
    from smplx import MANO
    import prepare_native as native
    from native_data import eligible_rows, rigid_valid
    from scipy.spatial import cKDTree
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError('GPU conversion requested but CUDA unavailable')
    args.device = 'cuda:0'
    meta['mano_sha256'] = {str((args.mano_root / f'MANO_{side.upper()}.pkl').resolve()):
                         sha(args.mano_root / f'MANO_{side.upper()}.pkl') for side in ('left', 'right')}
    models = {side: MANO(str(args.mano_root), is_rhand=side == 'right', use_pca=False,
                        flat_hand_mean=False).to(args.device) for side in ('right', 'left')}
    inventory = [('grab', p) for p in sorted((args.grab_root / 'grab').glob('*/*.npz'))]
    inventory += [('arctic', p) for p in sorted((args.arctic_root / 'raw_seqs').glob('*/*.mano.npy'))]
    if not inventory or len(inventory) > args.max_sequences:
        raise ValueError('full inventory empty or exceeds sequence budget')
    meta['selection'] = 'all locally available native sequences; no pilot subsampling'
    meta['inventory_sequences'] = len(inventory)
    cache = {}
    (out / 'canonical').mkdir()
    for source, path in inventory:
        check_budget(out.parent, start, args.seconds, args.max_output_gib)
        source_sequence = path.parent.name + '/' + path.name
        official_split(source, source_sequence, protocol)
        before = sha(path)
        hand, poses, ids, timestamps, meshes, dependencies = (
            native.grab(path, args) if source == 'grab' else native.arctic(path, args, models))
        valid = np.isfinite(hand).all((-1, -2))
        hand[~valid] = 0
        pose_valid = rigid_valid(poses)
        centers, objects, clouds = [], [], []
        for key, mesh, meshpath in meshes:
            if key not in cache:
                cache[key] = native.sample_surface(mesh, key)
                np.savez_compressed(out / 'canonical' / (key + '.npz'), **cache[key],
                                    mesh_path=str(meshpath.resolve()), mesh_sha256=sha(meshpath))
            clouds.append(cache[key]['points'])
            centers.append(cache[key]['center'])
            objects.append(key)
        centers = np.asarray(centers, 'float32')
        near = np.zeros((len(hand), len(objects)), bool)
        for j, cloud in enumerate(clouds):
            for t in range(len(hand)):
                world = cloud @ poses[t, j, :3, :3].T + poses[t, j, :3, 3]
                distances = cKDTree(world).query(hand[t].reshape(22, 3))[0].reshape(2, 11)
                near[t, j] = np.min(np.where(valid[t, :, None], distances, np.inf)) < .05
        rows = eligible_rows(hand, valid, poses, pose_valid, near, timestamps, centers)
        name = f'{source}_{path.parent.name}_{path.name.split(".")[0]}'
        dest = out / 'sequences' / name
        dest.mkdir()
        arrays = dict(hand=hand, hand_valid=valid, poses=poses, pose_valid=pose_valid,
                      program=np.zeros_like(near), near=near, frame_ids=ids,
                      source_frame_ids=ids, timestamps=timestamps, centers=centers)
        for key, value in arrays.items():
            np.save(dest / (key + '.npy'), value)
        dependencies.extend(meshpath for _, _, meshpath in meshes)
        record = dict(source=source, sequence=name, source_sequence=source_sequence,
            source_path=str(path.resolve()), source_sha256=before, frames=len(ids),
            object_instances=len(objects), eligible_windows=len(rows), objects=objects,
            hand_label='native_MANO_parameters', effect_label='native_object_parameters',
            program_available=False, articulated_parts=source == 'arctic',
            dependencies={str(p.resolve()): sha(p) for p in dependencies},
            motion_windows=int((rows[:, 2] == 1).sum()),
            array_sha256={p.name: sha(p) for p in dest.glob('*.npy')})
        record = add_record(meta, rows_by_split, record, rows, protocol)
        write_json(dest / 'meta.json', record)
        print(json.dumps(dict(completed=len(meta['records']), total=len(inventory),
             sequence=name, windows=len(rows), elapsed_seconds=time.monotonic() - start)), flush=True)


def audit_pack(pack, protocol):
    """Full motion-category/clock/split audit plus per-sequence tensor correspondence."""
    import numpy as np
    from native_data import NativeWindows, rigid_valid
    from oakink_wm.data import transform_points
    root = Path(pack) / 'processed'
    meta = json.loads((root / 'manifest.json').read_text())
    indices = {s: np.load(root / f'index_{s}.npy') for s in ('train', 'val', 'test')}
    identities, maximum = set(), 0.
    counts = {}
    for split, rows in indices.items():
        dataset = NativeWindows(pack, split) if len(rows) else None
        counts[split] = len(rows)
        for sid, record in enumerate(meta['records']):
            selected = np.flatnonzero(rows[:, 0] == sid)
            if official_split(record['source'], record['source_sequence'], protocol) != record['split']:
                raise ValueError('record official split mismatch')
            if len(selected) and record['split'] != split:
                raise ValueError('window assigned to wrong official split')
            if not len(selected):
                continue
            arrays = root / 'sequences' / record['sequence']
            poses = np.load(arrays / 'poses.npy', mmap_mode='r')
            timestamps = np.load(arrays / 'timestamps.npy', mmap_mode='r')
            if not np.allclose(np.diff(timestamps), 1 / 30, atol=1e-5, rtol=0):
                raise ValueError('invalid native clock')
            if not rigid_valid(poses).all():
                raise ValueError('nonrigid native object pose')
            for name, digest in record['array_sha256'].items():
                if sha(arrays / name) != digest:
                    raise ValueError('array changed during conversion')
            local = rows[selected]
            for chunk in np.array_split(local, max(1, (len(local) + 1023) // 1024)):
                anchor, tick, category = chunk[:, 1:].T
                future = poses[tick[:, None] + np.arange(1, 25), anchor[:, None]]
                now = poses[tick, anchor]
                delta = np.linalg.norm(future[..., :3, 3] - now[:, None, :3, 3], axis=-1)
                rotation = future[..., :3, :3] @ now[:, None, :3, :3].swapaxes(-1, -2)
                angle = np.arccos(np.clip((np.trace(rotation, axis1=-2, axis2=-1) - 1) / 2, -1, 1))
                expected = np.where((delta > .002).any(1) | (angle > .02).any(1), 1, 2)
                if not np.array_equal(category, expected):
                    raise ValueError('full-future native motion category mismatch')
                for row in chunk:
                    identity = (record['source'], record['source_sequence'], int(row[1]), int(row[2]))
                    if identity in identities:
                        raise ValueError('duplicate or cross-split window')
                    identities.add(identity)
            for idx in np.unique([selected[0], selected[-1]]):
                sample = dataset[int(idx)]
                if not all(np.isfinite(sample[k]).all() for k in ('xyz', 'features', 'action', 'points', 'effect')):
                    raise ValueError('nonfinite tensor sample')
                row = rows[int(idx)]
                current = np.linalg.inv(poses[int(row[2]), int(row[1])])
                # Reconstruct selected object point futures independently of effect application.
                d = dataset.sequence(record['sequence'])
                from oakink_wm.data import local_objects
                objects = local_objects(poses[int(row[2])], d['pose_valid'][int(row[2])], int(row[1]), d['centers'])
                for slot, obj in enumerate(objects):
                    cloud = dataset.canonical(record['objects'][obj])[0]
                    expected = transform_points(cloud[None], current @ poses[int(row[2])+1:int(row[2])+25, obj])
                    actual = transform_points(sample['points'][slot][None], sample['effect'][slot])
                    maximum = max(maximum, float(np.linalg.norm(expected - actual, axis=-1).max()))
    if maximum > 1e-5:
        raise ValueError('rigid target correspondence mismatch')
    return dict(status='COMPLETED', verdict='ENGINEERING_PASS', official_split=True,
                windows=counts, checked_motion_categories=len(identities),
                maximum_rigid_correspondence_error_m=maximum,
                absent_source_test=['arctic'] if not any(r['source']=='arctic' and r['split']=='test' for r in meta['records']) else [])


def worker(args):
    import numpy as np
    # Compatibility aliases must precede licensed chumpy/SMPL imports.
    for name, value in [('bool', bool), ('int', int), ('float', float), ('complex', complex),
                        ('object', object), ('unicode', str), ('str', str)]:
        if name not in np.__dict__:
            setattr(np, name, value)
    protocol = arctic_protocol(args.protocol)
    out = args.output / 'processed'
    out.mkdir()
    (out / 'sequences').mkdir()
    meta = manifest_base(args, protocol)
    rows = {s: [] for s in ('train', 'val', 'test')}
    start = time.monotonic()
    write_json(out / 'manifest.json', meta)
    try:
        if args.reindex:
            reindex(args, meta, out, rows, protocol)
        else:
            convert(args, meta, out, rows, protocol, start)
        verify_inputs(meta)
        for split, values in rows.items():
            np.save(out / f'index_{split}.npy', np.asarray(values, np.int64).reshape(-1, 4))
        meta.update(status='COMPLETED', windows={s: len(v) for s, v in rows.items()})
        write_json(out / 'manifest.json', meta)
        report = audit_pack(args.output, protocol)
        check_budget(args.output, start, args.seconds, args.max_output_gib)
        meta['canonical_sha256'] = {p.name: sha(p) for p in (out / 'canonical').glob('*.npz')}
        meta.update(training_allowed=True, audit_sha256=None)
        write_json(args.output / 'audit.json', report)
        meta['audit_sha256'] = sha(args.output / 'audit.json')
    except BaseException as error:
        meta.update(status='FAILED', training_allowed=False, error=repr(error))
        raise
    finally:
        meta['elapsed_s'] = time.monotonic() - start
        write_json(out / 'manifest.json', meta)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reindex', type=Path, help='completed audited native pack; no GPU/MANO work')
    parser.add_argument('--grab-root', type=Path, default=ROOT / 'data/raw_data/GRAB')
    parser.add_argument('--arctic-root', type=Path, default=ROOT / 'data/raw_data/ARCTIC/arctic/data/arctic_data/data')
    parser.add_argument('--mano-root', type=Path, default=ROOT / 'data/raw_data/ARCTIC/arctic/data/body_models/mano')
    parser.add_argument('--protocol', type=Path, default=ROOT / 'data/raw_data/ARCTIC/arctic/data/arctic_data/data/splits_json/protocol_p1.json')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--max-sequences', type=int, default=2000)
    parser.add_argument('--seconds', type=int, default=3600)
    parser.add_argument('--max-output-gib', type=float, default=5)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not (1 <= args.max_sequences <= 2000 and 1 <= args.seconds <= 3600 and 0 < args.max_output_gib <= 5 and 0 <= args.gpu <= 7):
        parser.error('bounds: sequences 1..2000, seconds 1..3600, output (0,5] GiB, GPU 0..7')
    args.output = args.output.resolve()
    if args.worker:
        worker(args)
        return
    inputs = [args.reindex] if args.reindex else [args.grab_root, args.arctic_root, args.mano_root]
    guard_output(args.output, inputs)
    arctic_protocol(args.protocol)  # fail before creating output or importing torch
    if not args.reindex:
        gpu_guard(args.gpu)
    args.output.mkdir(parents=True)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(ROOT / 'tmp'))
    if not args.reindex:
        env['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    command = [sys.executable, '-B', str(Path(__file__).resolve()), *sys.argv[1:], '--worker']
    started = time.monotonic()
    status = dict(status='RUNNING', command=command, child_pid=None, physical_gpu=None if args.reindex else args.gpu,
                  seconds=args.seconds, max_output_gib=args.max_output_gib)
    process = None
    try:
        with (args.output / 'console.log').open('w') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=ROOT, start_new_session=True)
            status['child_pid'] = process.pid
            write_json(args.output / 'run_manifest.json', status)
            while process.poll() is None:
                check_budget(args.output, started, args.seconds, args.max_output_gib)
                time.sleep(.5)
            if process.returncode:
                raise RuntimeError('conversion worker exited ' + str(process.returncode))
        status['status'] = 'COMPLETED'
    except BaseException as error:
        status.update(status='TIMED_OUT' if isinstance(error, TimeoutError) else 'FAILED', error=repr(error))
        if process and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
        manifest = args.output / 'processed/manifest.json'
        if manifest.exists():
            data = json.loads(manifest.read_text())
            data.update(status=status['status'], training_allowed=False, supervisor_error=repr(error))
            write_json(manifest, data)
        raise
    finally:
        status.update(elapsed_s=time.monotonic() - started, returncode=process.returncode if process else None)
        write_json(args.output / 'run_manifest.json', status)


if __name__ == '__main__':
    main()
