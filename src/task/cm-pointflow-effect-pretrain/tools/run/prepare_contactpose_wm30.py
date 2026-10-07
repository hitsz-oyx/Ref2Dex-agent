#!/usr/bin/env python3
"""Bounded CPU native ContactPose -> 30Hz WM packs with subject-held-out splits."""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK / 'src'))
from oakink_wm.contactpose import (SEMANTICS, OPENPOSE_11, native_trajectory,
                                  resample_segments, sha, subject_split, window_rows)


def canonical(meshpath, object_name):
    # Grasp-local thermal contact meshes are already in metres; the separate
    # 3Dmodels.zip/contactpose_ply_files_mm meshes are in millimetres.
    mesh = trimesh.load(meshpath, process=False, force='mesh')
    triangles = np.asarray(mesh.triangles)
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    areas = np.linalg.norm(cross, axis=-1)
    if not np.isfinite(triangles).all() or areas.sum() <= 0:
        raise ValueError('invalid ContactPose mesh ' + str(meshpath))
    diameter = float(np.linalg.norm(mesh.extents))
    if not .01 < diameter < 1.5:
        raise ValueError('mesh metre-scale contract failed ' + str(meshpath))
    rng = np.random.default_rng(int(hashlib.sha256(('contactpose:' + object_name).encode()).hexdigest()[:8], 16))
    faces = rng.choice(len(areas), 512, p=areas / areas.sum())
    u, v = np.sqrt(rng.random(512)), rng.random(512)
    bary = np.stack((1-u, u*(1-v), u*v), axis=-1)
    points = (triangles[faces] * bary[:, :, None]).sum(1)
    radius = np.sqrt(np.mean(np.sum((points - points.mean(0))**2, axis=-1)))
    return dict(points=points.astype('float32'), normals=(cross[faces]/areas[faces, None]).astype('float32'),
                radius=radius, center=np.asarray(mesh.centroid, 'float32'),
                face_indices=faces, barycentric=bary, mesh_path=str(meshpath.resolve()), mesh_sha256=sha(meshpath))


def select_paths(root, limit, subjects=None):
    paths = [p for p in sorted((root / 'data/contactpose_data').glob('*/*/annotations.json'))
             if p.parent.name not in ('hands', 'palm_print')]
    if subjects:
        paths = [p for p in paths if p.parent.parent.name.split('_')[0] in subjects]
    # Spread bounded pilots across subjects/intents; no nested duplicate copies.
    return [paths[i] for i in np.linspace(0, len(paths)-1, min(len(paths), limit), dtype=int)] if paths else []


def directory_bytes(path):
    return sum(p.stat().st_size for p in path.rglob('*') if p.is_file())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contactpose-root', type=Path, default=Path('data/raw_data/ContactPose/ContactPose'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=20)
    parser.add_argument('--subjects', help='comma-separated fullN subject IDs')
    parser.add_argument('--seconds', type=int, default=3600)
    parser.add_argument('--max-output-gib', type=float, default=5.)
    parser.add_argument('--max-gap-s', type=float, default=.075)
    parser.add_argument('--max-translation-speed', type=float, default=3.)
    parser.add_argument('--max-angular-speed', type=float, default=15.)
    args = parser.parse_args()
    if not (1 <= args.limit <= 1600 and 1 <= args.seconds <= 3600 and 0 < args.max_output_gib <= 5
            and 1/30 <= args.max_gap_s <= .1 and 0 < args.max_translation_speed <= 10
            and 0 < args.max_angular_speed <= 30):
        parser.error('conversion bounds exceeded')
    root, output = args.contactpose_root.resolve(), args.output.resolve()
    repo = TASK.parents[2].resolve()
    if repo / 'outputs' not in output.parents:
        parser.error('output must be inside this repository outputs/')
    if output.exists():
        raise FileExistsError(output)
    if shutil.disk_usage(repo).free < 20 * 1024**3:
        raise RuntimeError('less than 20 GiB free')
    paths = select_paths(root, args.limit, set(args.subjects.split(',')) if args.subjects else None)
    if not paths:
        raise ValueError('no canonical top-level ContactPose annotations')
    output.mkdir(parents=True)
    processed = output / 'processed'
    (processed / 'sequences').mkdir(parents=True)
    (processed / 'canonical').mkdir()
    start = time.monotonic()
    manifest = dict(schema='ref2dex.native-wm30.v1', source='contactpose', status='RUNNING', fps=30,
                    history=4, horizon=24, units='m', coordinate_frame='native_mocap_world',
                    hand_order=['right', 'left'], hand_semantics=SEMANTICS, openpose_joint_indices=OPENPOSE_11,
                    sequences=[], records=[], splits={s: [] for s in ('train', 'val', 'test')},
                    training_allowed=False, program_available=False,
                    hand_label='fixed_articulation_native_21_joints_with_per_frame_rigid_transforms',
                    supervision_scope='rigid_grasp_transport; no dynamic finger or time-varying contact GT',
                    split_rule='sha256(contactpose:fullN) modulo 100; <80 train, <90 val, otherwise test',
                    source_root=str(root), selected_annotations=[str(p) for p in paths],
                    resource=dict(device='cpu', reason='file parsing, rigid interpolation and point-distance audit; no neural model',
                                  seconds=args.seconds, max_output_gib=args.max_output_gib),
                    masks=dict(max_gap_s=args.max_gap_s, max_translation_speed=args.max_translation_speed,
                               max_angular_speed=args.max_angular_speed, min_contiguous_frames=28),
                    git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    code_sha256={str(p): sha(p) for p in (Path(__file__), TASK/'src/oakink_wm/contactpose.py')},
                    inputs_sha256={}, skipped=[], audits=[])
    canonical_cache = {}
    rows = {s: [] for s in manifest['splits']}
    try:
        for path in paths:
            if time.monotonic() - start > args.seconds:
                raise TimeoutError('ContactPose conversion deadline')
            if directory_bytes(output) > args.max_output_gib * 1024**3:
                raise RuntimeError('ContactPose output budget exceeded')
            annotation_sha = sha(path)
            manifest['inputs_sha256'][str(path)] = annotation_sha
            annotation = json.loads(path.read_text())
            raw = native_trajectory(annotation)
            packs, audit = resample_segments(raw, args.max_gap_s, args.max_translation_speed, args.max_angular_speed)
            audit.update(source_path=str(path), segments=len(packs))
            manifest['audits'].append(audit)
            if not packs:
                manifest['skipped'].append(dict(source_path=str(path), reason='no >=28-frame contiguous valid segment'))
                continue
            obj = path.parent.name
            key = 'contactpose_' + obj
            if key not in canonical_cache:
                # Pick a deterministic source mesh, shared by every grasp of this object.
                meshpath = sorted((root/'data/contactpose_data').glob('*/' + obj + '/' + obj + '.ply'))[0]
                cloud = canonical(meshpath, obj)
                canonical_cache[key] = cloud
                np.savez_compressed(processed / 'canonical' / (key + '.npz'), **cloud)
                manifest['inputs_sha256'][str(meshpath)] = cloud['mesh_sha256']
            cloud = canonical_cache[key]
            subject = path.parent.parent.name.split('_')[0]
            split = subject_split(subject)
            for segment, pack in enumerate(packs):
                near = np.zeros((len(pack['hand']), 1), bool)
                for tick, pose in enumerate(pack['poses'][:, 0]):
                    world = cloud['points'] @ pose[:3, :3].T + pose[:3, 3]
                    distance = cKDTree(world).query(pack['hand'][tick].reshape(22, 3))[0].reshape(2, 11)
                    near[tick, 0] = np.min(np.where(pack['hand_valid'][tick, :, None], distance, np.inf)) < .05
                selected = window_rows(pack, near)
                if not len(selected):
                    continue
                name = 'contactpose_' + path.parent.parent.name + '_' + obj + '_seg' + str(segment)
                dest = processed / 'sequences' / name
                dest.mkdir()
                pack.update(near=near, program=np.zeros_like(near), centers=cloud['center'][None])
                for field, value in pack.items():
                    if field != 'raw_range':
                        np.save(dest / (field + '.npy'), value)
                record = dict(source='contactpose', dataset_name='contactpose', sequence=name, objects=[key],
                              subject=subject, source_sequence=path.parent.parent.name+'/'+obj,
                              source_path=str(path), source_sha256=annotation_sha, split=split,
                              frames=len(pack['hand']), eligible_windows=len(selected), object_instances=1,
                              motion_windows=int((selected[:, 2] == 1).sum()), source_frame_range=pack['raw_range'],
                              timestamp_camera=raw['camera'], timestamp_origin_ns=raw['timestamp_origin_ns'],
                              hand_label=manifest['hand_label'], effect_label='native_oTw_Optitrack_or_optimized_pose',
                              program_available=False, articulated_parts=False)
                (dest / 'meta.json').write_text(json.dumps(record, indent=2)+'\n')
                seqidx = len(manifest['sequences'])
                rows[split].extend([[seqidx, *r] for r in selected.tolist()])
                manifest['sequences'].append(name)
                manifest['records'].append(record)
                manifest['splits'][split].append(name)
            print(json.dumps(dict(source_sequence=path.parent.parent.name+'/'+obj, split=split,
                                  raw_frames=len(raw['times']), segments=len(packs), elapsed_s=time.monotonic()-start)), flush=True)
        for source, expected in manifest['inputs_sha256'].items():
            if sha(source) != expected:
                raise RuntimeError('source changed during conversion: ' + source)
        if directory_bytes(output) > args.max_output_gib * 1024**3:
            raise RuntimeError('ContactPose output budget exceeded')
        if shutil.disk_usage(repo).free < 20 * 1024**3:
            raise RuntimeError('less than 20 GiB free after conversion')
        for split, values in rows.items():
            np.save(processed / ('index_' + split + '.npy'), np.asarray(values, dtype=np.int64).reshape(-1, 4))
        manifest.update(status='COMPLETED', training_allowed=bool(rows['train']),
                        windows={s: len(values) for s, values in rows.items()})
    except Exception as error:
        manifest.update(status='FAILED', error=repr(error), training_allowed=False)
        raise
    finally:
        manifest['elapsed_s'] = time.monotonic() - start
        manifest['output_bytes'] = directory_bytes(output)
        (processed / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
