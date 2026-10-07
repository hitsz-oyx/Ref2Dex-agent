"""Independent official pose parity, source integrity, splits and WM tensor checks."""
import sys
# Protect all imports, including dynamically loaded external source modules.
# This must work even when the caller omits -B / PYTHONDONTWRITEBYTECODE.
sys.dont_write_bytecode = True

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
import trimesh

from native_data import NativeWindows, rigid_valid, sha
from oakink_wm.data import collate, transform_points


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--external-repo', type=Path, default=Path('/home2/wyy/oyx_ws/Ref2Dex'))
    a = p.parse_args()
    reportpath = a.input/'audit.json'
    if reportpath.exists():
        raise FileExistsError(reportpath)
    torch.set_num_threads(2)
    root = a.input/'processed'
    meta = json.loads((root/'manifest.json').read_text())
    if meta['status'] != 'COMPLETED':
        raise ValueError('conversion incomplete')
    ompath = a.external_repo/'dataset/GRAB/tools/objectmodel.py'
    rotpath = a.external_repo/'dataset/arctic/common/rot.py'
    objpath = a.external_repo/'dataset/arctic/common/object_tensors.py'
    om = module(ompath, 'official_grab_objectmodel')
    rot = module(rotpath, 'official_arctic_rotation')
    report = dict(status='RUNNING', source_integrity=True, subject_overlap=[],
                  official_source_hashes={str(p): sha(p) for p in (ompath, rotpath, objpath)},
                  records=[], interface_checks=[], maximum_rigid_correspondence_error_m=0)
    subjects = {s: set() for s in ('train', 'val', 'test')}
    verified_parts = set()
    for record in meta['records']:
        path = Path(record['source_path'])
        if sha(path) != record['source_sha256']:
            raise ValueError('source drift ' + str(path))
        for p, checksum in record['dependencies'].items():
            if sha(Path(p)) != checksum:
                raise ValueError('dependency drift ' + p)
        subjects[record['split']].add((record['source'], record['source_sequence'].split('/')[0]))
        d = root/'sequences'/record['sequence']
        poses = np.load(d/'poses.npy'); ids = np.load(d/'frame_ids.npy')
        ticks = np.unique(np.linspace(0, len(ids)-1, min(len(ids), 12), dtype=int))
        if not rigid_valid(poses).all():
            raise ValueError('nonrigid pose')
        if record['source'] == 'grab':
            with np.load(path, allow_pickle=True) as f:
                raw = f['object'].item()
            rawroot = path.parents[2]
            mesh = trimesh.load(rawroot/raw['object_mesh'], process=False)
            points = np.asarray(mesh.vertices)[::max(1, len(mesh.vertices)//64)]
            model = om.ObjectModel(points, batch_size=len(ticks), dtype=torch.float64)
            reference = model(global_orient=torch.as_tensor(raw['params']['global_orient'][ids[ticks]], dtype=torch.float64),
                              transl=torch.as_tensor(raw['params']['transl'][ids[ticks]], dtype=torch.float64)).vertices.detach().numpy()
            actual = transform_points(points, poses[ticks, 0].astype('float64'))
            error = float(np.max(np.linalg.norm(actual-reference, axis=-1)))
        else:
            raw = np.load(path.with_name(path.name.replace('.mano.npy', '.object.npy')))[ticks]
            quat_global = rot.axis_angle_to_quaternion(torch.as_tensor(raw[:, 1:4], dtype=torch.float64))
            quat_top = rot.axis_angle_to_quaternion(torch.as_tensor(np.column_stack((np.zeros((len(ticks), 2)), -raw[:, 0])), dtype=torch.float64))
            error = 0.
            for part, obj in enumerate(record['objects']):
                with np.load(root/'canonical'/(obj+'.npz')) as f:
                    points = torch.as_tensor(f['points'], dtype=torch.float64)[None].expand(len(ticks), -1, -1)
                local = rot.quaternion_apply(quat_top[:, None], points) if part == 1 else points
                reference = rot.quaternion_apply(quat_global[:, None], local).numpy() + raw[:, None, 4:7]/1000
                actual = transform_points(points[0].numpy(), poses[ticks, part].astype('float64'))
                error = max(error, float(np.max(np.linalg.norm(actual-reference, axis=-1))))
            # Independent original part labels, rather than the older local adapter.
            with np.load(root/'canonical'/(record['objects'][1]+'.npz')) as f:
                meshpath = Path(str(f['mesh_path']))
            full = trimesh.load(meshpath, process=False)
            labels = np.asarray(json.loads(meshpath.with_name('parts.json').read_text()))
            for part, obj in enumerate(record['objects']):
                if obj in verified_parts:
                    continue
                accepted = 1 if part == 0 else 0
                original_tri = full.vertices[full.faces[(labels[full.faces] == accepted).all(-1)]]/1000
                with np.load(root/'canonical'/(obj+'.npz')) as f:
                    cloud = f['points']
                    faces, barycentric = f['face_indices'], f['barycentric']
                if not (barycentric >= 0).all() or not np.allclose(barycentric.sum(-1), 1):
                    raise ValueError('invalid barycentric coordinates')
                # Exact original-triangle provenance avoids nearest-point numerical
                # instability for ARCTIC's extremely thin/degenerate triangles.
                expected = (original_tri[faces] * barycentric[:, :, None]).sum(1)
                distance = np.linalg.norm(cloud-expected, axis=-1)
                if distance.max() > 1e-6:
                    raise ValueError(f'ARCTIC part assignment mismatch: {obj}: {distance.max()}m')
                verified_parts.add(obj)
        if error > 2e-6:
            raise ValueError('official object parity failed ' + record['sequence'])
        report['records'].append(dict(sequence=record['sequence'], source=record['source'],
                                      official_pose_geometry_max_error_m=error,
                                      eligible_windows=record['eligible_windows']))
    for s in subjects:
        for t in subjects:
            if s != t and subjects[s] & subjects[t]:
                raise ValueError('subject leakage')
    report['subjects'] = {s: sorted(v) for s, v in subjects.items()}
    for split in subjects:
        dataset = NativeWindows(a.input, split)
        # Audit every indexed category independently of the conversion helper.
        checked_categories = 0
        for seqidx in np.unique(dataset.rows[:, 0]):
            poses = dataset.sequence(dataset.sequences[seqidx])['poses']
            for _, anchor, tick, category in dataset.rows[dataset.rows[:,0] == seqidx]:
                future = poses[tick+1:tick+25, anchor]
                displacement = np.linalg.norm(future[:,:3,3]-poses[tick,anchor,:3,3],axis=-1)
                rotation = future[:,:3,:3] @ poses[tick,anchor,:3,:3].T
                angle = np.arccos(np.clip((np.trace(rotation,axis1=1,axis2=2)-1)/2,-1,1))
                expected = 1 if (displacement>.002).any() or (angle>.02).any() else 2
                if category != expected:
                    raise ValueError(f'full-horizon motion category mismatch: {split} '
                                     f'{dataset.sequences[seqidx]} anchor{anchor} tick{tick}')
                checked_categories += 1
        # Deterministic stratification by sequence; no stochastic training or seed.
        selected = []
        for seqidx in np.unique(dataset.rows[:, 0]):
            candidates = np.flatnonzero(dataset.rows[:, 0] == seqidx)
            selected.extend(candidates[np.linspace(0, len(candidates)-1, min(4, len(candidates)), dtype=int)])
        samples = []
        error = 0.
        for index in selected:
            sample = dataset[int(index)]
            if not all(np.isfinite(v).all() for v in sample.values() if hasattr(v, 'dtype')):
                raise ValueError('nonfinite WM input')
            seqidx, anchor, tick = sample['sample_id']
            data = dataset.sequence(dataset.sequences[seqidx])
            from oakink_wm.data import local_objects
            objects = local_objects(data['poses'][tick], data['pose_valid'][tick], anchor, data['centers'])
            C = np.linalg.inv(data['poses'][tick, anchor])
            for slot, objidx in enumerate(objects):
                canonical = dataset.canonical(data['objects'][objidx])[0]
                for h in (0, 7, 23):
                    actual = transform_points(sample['points'][slot], sample['effect'][slot, h])
                    reference = transform_points(canonical, C @ data['poses'][tick+h+1, objidx])
                    error = max(error, float(np.max(np.linalg.norm(actual-reference, axis=-1))))
            samples.append(sample)
        collate(samples[:4])
        if error > 2e-6:
            raise ValueError('fixed-current-frame effect mismatch')
        report['maximum_rigid_correspondence_error_m'] = max(report['maximum_rigid_correspondence_error_m'], error)
        report['interface_checks'].append(dict(split=split, windows=len(dataset), checked_windows=len(selected),
                                               checked_motion_categories=checked_categories,
                                               correspondence_max_error_m=error,
                                               category_counts=[len(g) for g in dataset.groups]))
    report.update(status='COMPLETED', verdict='ENGINEERING_PASS', windows=meta['windows'],
                  source_windows={s: sum(r['eligible_windows'] for r in meta['records'] if r['source']==s) for s in ('grab','arctic')})
    reportpath.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('verdict','windows','source_windows','maximum_rigid_correspondence_error_m')}))


if __name__ == '__main__':
    main()
