"""Convert bounded read-only GRAB / articulated ARCTIC into WM30-compatible packs."""
import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
for name, value in [('bool', bool), ('int', int), ('float', float), ('complex', complex),
                    ('object', object), ('unicode', str), ('str', str)]:
    if name not in np.__dict__:
        setattr(np, name, value)
import torch
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from smplx import MANO

from native_data import MANO_MCP, MANO_TIPS, SEMANTICS, arctic_poses, eligible_rows, rigid_valid, sha


def sample_surface(mesh, key):
    tri = np.asarray(mesh.triangles)
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = np.linalg.norm(cross, axis=-1)
    if not np.isfinite(tri).all() or area.sum() <= 0:
        raise ValueError('invalid mesh')
    import hashlib
    rng = np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest()[:8], 16))
    faces = rng.choice(len(tri), 512, p=area / area.sum())
    u, v = np.sqrt(rng.random(512)), rng.random(512)
    points = (tri[faces, 0] * (1-u[:, None]) + tri[faces, 1] * (u*(1-v))[:, None]
              + tri[faces, 2] * (u*v)[:, None])
    return dict(points=points.astype('float32'), normals=(cross[faces]/area[faces, None]).astype('float32'),
                face_indices=faces, barycentric=np.stack((1-u, u*(1-v), u*v), axis=-1),
                center=np.asarray(mesh.centroid, 'float32'),
                radius=np.sqrt(np.mean(np.sum((points-points.mean(0))**2, axis=-1))))


def reconstruct(fields, model, device):
    values = []
    with torch.no_grad():
        for start in range(0, len(fields['global_orient']), 128):
            stop = start + 128
            batch = {k: torch.as_tensor(v[start:stop], dtype=torch.float32, device=device)
                     for k, v in fields.items()}
            if 'betas' not in batch:
                batch['betas'] = model.betas.expand(len(batch['global_orient']), -1)
            result = model(**batch)
            values.append(torch.cat((result.joints[:, [0] + MANO_MCP],
                                     result.vertices[:, MANO_TIPS]), dim=1).cpu().numpy())
    return np.concatenate(values).astype('float32')


def grab(path, a):
    with np.load(path, allow_pickle=True) as f:
        raw = {k: f[k].item() if f[k].shape == () else f[k] for k in f.files}
    fps = float(raw['framerate'])
    if fps != 120:
        raise ValueError('expected native GRAB 120Hz')
    ids = np.arange(0, int(raw['n_frames']), 4)
    hands, dependencies = [], []
    for side, key in [('right', 'rhand'), ('left', 'lhand')]:
        template = a.grab_root / raw[key]['vtemp']
        dependencies.append(template)
        model = MANO(str(a.mano_root), is_rhand=side == 'right', use_pca=False,
                     flat_hand_mean=True, v_template=np.asarray(
                         trimesh.load(template, process=False).vertices, 'float32')).to(a.device)
        params = raw[key]['params']
        hands.append(reconstruct({'global_orient': params['global_orient'][ids],
                                  'hand_pose': params['fullpose'][ids], 'transl': params['transl'][ids]},
                                 model, a.device))
    object_data = raw['object']
    pose = np.broadcast_to(np.eye(4), (len(ids), 1, 4, 4)).copy()
    # Official GRAB ObjectModel uses row-vector v @ R, so column-vector R is transposed.
    pose[:, 0, :3, :3] = Rotation.from_rotvec(object_data['params']['global_orient'][ids]).as_matrix().swapaxes(-1, -2)
    pose[:, 0, :3, 3] = object_data['params']['transl'][ids]
    meshpath = a.grab_root / object_data['object_mesh']
    meshes = [(f'grab_{raw["obj_name"]}', trimesh.load(meshpath, process=False), meshpath)]
    return np.stack(hands, axis=1), pose.astype('float32'), ids, ids / fps, meshes, dependencies


def arctic(path, a, models):
    hand_params = np.load(path, allow_pickle=True).item()
    objectpath = path.with_name(path.name.replace('.mano.npy', '.object.npy'))
    object_params = np.load(objectpath)
    ids = np.arange(len(object_params))
    hands = []
    for side in ('right', 'left'):
        fields = hand_params[side]
        hands.append(reconstruct({'global_orient': fields['rot'], 'hand_pose': fields['pose'],
                                  'transl': fields['trans'], 'betas': np.broadcast_to(fields['shape'], (len(ids), 10))},
                                 models[side], a.device))
    obj = path.name.split('_')[0]
    assets = a.arctic_root / 'meta/object_vtemplates' / obj
    full = trimesh.load(assets / 'mesh.obj', process=False)
    # Raw ARCTIC object templates and object translations are explicitly mm.
    full.vertices = np.asarray(full.vertices) / 1000
    labels = np.asarray(json.loads((assets / 'parts.json').read_text()), bool)
    meshes = []
    # Use original mesh coordinates and unambiguous per-vertex part assignments.
    # Never silently assign a mixed triangle to one part.
    # Official ARCTIC adds one to raw parts.json and articulates parts_ids==1:
    # raw 0 is top; raw 1 is bottom. The older local adapter reverses these.
    for part, label in [('bottom', True), ('top', False)]:
        faces = full.faces[(labels[full.faces] == label).all(-1)]
        mesh = trimesh.Trimesh(vertices=full.vertices, faces=faces, process=False)
        mesh.remove_unreferenced_vertices()
        meshes.append((f'arctic_{obj}_{part}', mesh, assets / 'mesh.obj'))
    return (np.stack(hands, axis=1), arctic_poses(object_params), ids, ids / 30, meshes,
            [objectpath, assets / 'parts.json'])


def choose(paths, count):
    paths = sorted(paths)
    return [paths[i] for i in np.linspace(0, len(paths)-1, min(len(paths), count), dtype=int)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--grab-root', type=Path, default=Path('data/raw_data/GRAB'))
    p.add_argument('--arctic-root', type=Path, default=Path('data/raw_data/ARCTIC/arctic/data/arctic_data/data'))
    p.add_argument('--mano-root', type=Path, default=Path('data/raw_data/ARCTIC/arctic/data/body_models/mano'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--per-source', type=int, default=20)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--seconds', type=int, default=1200)
    a = p.parse_args()
    if not 1 <= a.per_source <= 20:
        p.error('per-source must be 1..20')
    torch.set_num_threads(2)
    out = a.output.resolve() / 'processed'
    out.mkdir(parents=True, exist_ok=False)
    for name in ('sequences', 'canonical'):
        (out / name).mkdir()
    models = {side: MANO(str(a.mano_root), is_rhand=side == 'right', use_pca=False,
                         flat_hand_mean=False).to(a.device) for side in ('right', 'left')}
    start = time.monotonic()
    meta = dict(schema='ref2dex.native-wm30.v1', status='RUNNING', fps=30, horizon=24, history=4,
                hand_order=['right', 'left'], hand_semantics=SEMANTICS, units='m', sequences=[],
                records=[], splits={'train': [], 'val': [], 'test': []},
                base_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                code_sha256={p.name: sha(p) for p in Path(__file__).parent.glob('*.py')},
                mano_sha256={side: sha(a.mano_root / f'MANO_{side.upper()}.pkl') for side in ('left', 'right')},
                purpose='native3d_engineering_pilot', training_allowed=True,
                selection='20 evenly spaced source sequences; subject-disjoint pilot splits')
    rows_by_split = {k: [] for k in meta['splits']}
    canonical_cache = {}
    try:
        for source, paths in [('grab', choose((a.grab_root/'grab').glob('*/*.npz'), a.per_source)),
                              ('arctic', choose((a.arctic_root/'raw_seqs').glob('*/*.mano.npy'), a.per_source))]:
            for path in paths:
                if time.monotonic() - start > a.seconds:
                    raise TimeoutError('conversion deadline')
                hand, poses, ids, timestamps, meshes, dependencies = (
                    grab(path, a) if source == 'grab' else arctic(path, a, models))
                hand_valid = np.isfinite(hand).all((-1, -2))
                hand[~hand_valid] = 0
                pose_valid = rigid_valid(poses)
                points, centers, objects = [], [], []
                for key, mesh, meshpath in meshes:
                    if key not in canonical_cache:
                        canonical_cache[key] = sample_surface(mesh, key)
                        np.savez_compressed(out/'canonical'/(key+'.npz'), **canonical_cache[key],
                                            mesh_path=str(meshpath.resolve()), mesh_sha256=sha(meshpath))
                    cloud = canonical_cache[key]
                    points.append(cloud['points']); centers.append(cloud['center']); objects.append(key)
                centers = np.asarray(centers, 'float32')
                near = np.zeros((len(hand), len(objects)), bool)
                for j, cloud in enumerate(points):
                    for t in range(len(hand)):
                        world = cloud @ poses[t, j, :3, :3].T + poses[t, j, :3, 3]
                        distances = cKDTree(world).query(hand[t].reshape(22, 3))[0].reshape(2, 11)
                        near[t, j] = np.min(np.where(hand_valid[t, :, None], distances, np.inf)) < .05
                rows = eligible_rows(hand, hand_valid, poses, pose_valid, near, timestamps, centers)
                subject = path.parent.name
                subject_number = int(subject[1:])
                split = 'val' if subject_number == 8 else 'test' if subject_number >= 9 else 'train'
                name = f'{source}_{subject}_{path.name.split(".")[0]}'
                dest = out/'sequences'/name; dest.mkdir()
                arrays = dict(hand=hand, hand_valid=hand_valid, poses=poses, pose_valid=pose_valid,
                              program=np.zeros_like(near), near=near, frame_ids=ids,
                              source_frame_ids=ids, timestamps=timestamps, centers=centers)
                for key, value in arrays.items():
                    np.save(dest/(key+'.npy'), value)
                record = dict(source=source, sequence=name, source_sequence=f'{subject}/{path.name}',
                              source_path=str(path.resolve()), source_sha256=sha(path), frames=len(ids),
                              object_instances=len(objects), eligible_windows=len(rows), split=split,
                              objects=objects, hand_label='native_MANO_parameters', effect_label='native_object_parameters',
                              program_available=False, articulated_parts=source == 'arctic',
                              dependencies={str(p.resolve()): sha(p) for p in dependencies},
                              motion_windows=int((rows[:, 2] == 1).sum()))
                (dest/'meta.json').write_text(json.dumps(record, indent=2)+'\n')
                seqidx = len(meta['sequences']); meta['sequences'].append(name)
                meta['records'].append(record); meta['splits'][split].append(name)
                rows_by_split[split].extend([[seqidx, *r] for r in rows.tolist()])
                print(json.dumps({k: record[k] for k in ('source', 'sequence', 'frames', 'eligible_windows', 'motion_windows', 'split')}), flush=True)
        for split, rows in rows_by_split.items():
            np.save(out/f'index_{split}.npy', np.asarray(rows, dtype=np.int64).reshape(-1, 4))
        meta.update(status='COMPLETED', windows={s: len(r) for s, r in rows_by_split.items()})
    except Exception as e:
        meta.update(status='FAILED', error=repr(e))
        raise
    finally:
        meta['elapsed_s'] = time.monotonic() - start
        (out/'manifest.json').write_text(json.dumps(meta, indent=2)+'\n')


if __name__ == '__main__':
    main()
