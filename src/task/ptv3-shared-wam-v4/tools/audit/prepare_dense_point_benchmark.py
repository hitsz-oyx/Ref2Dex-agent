"""Prepare a bounded real-GRAB geometry fixture; never edit the source corpus."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np


def export(args):
    sys.path.insert(0, str(args.code_root / 'src/task/ptv3-shared-wam-v4/src'))
    from ptv3_shared_wam_v4 import paths
    from ptv3_shared_wam_v4.data import NineWindows
    recipe = json.loads((args.recipe / 'recipe.json').read_text())
    ds = NineWindows(recipe['main3_root'], recipe['native_root'], 'train', recipe)
    indices = ds.panel(4, 20261019)['1']
    records, arrays = [], {}
    source = ds.main.base.sources[1]
    for i, index in enumerate(indices):
        sample = ds[index][0]
        local = int(index - ds.main.base.offsets[1])
        seq_id, anchor, tick, _ = map(int, source.rows[local])
        seq_name = source.sequences[seq_id]
        folder = source.root / 'processed/sequences' / seq_name
        meta = json.loads((folder / 'meta.json').read_text())
        seq = source.sequence(seq_name)
        raw_ids = np.load(folder / 'source_frame_ids.npy')[tick-3:tick+25]
        poses = seq['poses'][tick-3:tick+25, anchor]
        arrays[f'{i}__poses'] = poses
        arrays[f'{i}__frame_ids'] = raw_ids
        for key, value in sample.items():
            if isinstance(value, np.ndarray) and value.dtype.kind not in 'OU':
                arrays[f'{i}__base__{key}'] = value
        records.append(dict(sequence=seq_name, source_path=meta['source_path'],
                            index=int(index), tick=tick, anchor=anchor))
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output / 'baseline.npz', **arrays)
    (args.output / 'metadata.json').write_text(json.dumps(dict(records=records,
        recipe=str(args.recipe), scope='four real TRAIN GRAB windows; performance only'), indent=2)+'\n')
    print(json.dumps(dict(output=str(args.output), records=records)), flush=True)


def reconstruct(args):
    # Pure geometric MANO skinning on CPU; no neural training/inference here.
    for name, value in [('bool', bool), ('int', int), ('float', float), ('complex', complex),
                        ('object', object), ('unicode', str), ('str', str)]:
        if name not in np.__dict__:
            setattr(np, name, value)
    import torch
    import trimesh
    from smplx import MANO
    torch.set_num_threads(2)
    started = time.monotonic()
    meta = json.loads((args.input / 'metadata.json').read_text())
    arrays = dict(np.load(args.input / 'baseline.npz', allow_pickle=False))
    rng = np.random.default_rng(176)
    errors = []
    for i, record in enumerate(meta['records']):
        raw_path = Path(record['source_path'])
        grab_root = raw_path.parents[2]
        with np.load(raw_path, allow_pickle=True) as f:
            raw = {k:f[k].item() if f[k].shape == () else f[k] for k in f.files}
        ids, poses = arrays[f'{i}__frame_ids'], arrays[f'{i}__poses']
        anchor = np.linalg.inv(poses[3])
        verts, keypoints = [], []
        for side, key in [('right', 'rhand'), ('left', 'lhand')]:
            template = trimesh.load(grab_root / raw[key]['vtemp'], process=False)
            model = MANO(str(args.mano_root), is_rhand=side == 'right', use_pca=False,
                         flat_hand_mean=True, v_template=np.asarray(template.vertices, np.float32))
            fields = raw[key]['params']
            with torch.no_grad():
                result = model(global_orient=torch.tensor(fields['global_orient'][ids], dtype=torch.float32),
                    hand_pose=torch.tensor(fields['fullpose'][ids], dtype=torch.float32),
                    transl=torch.tensor(fields['transl'][ids], dtype=torch.float32),
                    betas=model.betas.expand(len(ids), -1))
            v = result.vertices.numpy() @ anchor[:3, :3].T + anchor[:3, 3]
            kp = torch.cat((result.joints[:, [0,13,1,4,10,7]],
                            result.vertices[:, [744,320,443,554,671]]), dim=1).numpy()
            kp = kp @ anchor[:3, :3].T + anchor[:3, 3]
            verts.append(v.astype(np.float32)); keypoints.append(kp.astype(np.float32))
        vertices = np.stack(verts, axis=1)
        kp = np.stack(keypoints, axis=1)
        base_hand = arrays[f'{i}__base__start_hand']
        max_error = float(abs(kp[3] - base_hand).max())
        if max_error > 2e-5:
            raise ValueError(f'reconstructed hand/anchor mismatch: {max_error}')
        errors.append(max_error)
        arrays[f'{i}__dense_hand'] = vertices[3]
        arrays[f'{i}__dense_history'] = np.concatenate(((vertices[3]-vertices[2])*30,
            (vertices[3]-2*vertices[2]+vertices[1])*900, vertices[3]-vertices[0]), axis=-1)
        arrays[f'{i}__dense_trajectory'] = vertices[4:] - vertices[3]
        mesh = trimesh.load(grab_root / raw['object']['object_mesh'], process=False)
        triangles = np.asarray(mesh.triangles)
        cross = np.cross(triangles[:,1]-triangles[:,0], triangles[:,2]-triangles[:,0])
        area = np.linalg.norm(cross, axis=-1)
        face = rng.choice(len(area), 1536, p=area/area.sum())
        u, v = np.sqrt(rng.random(1536)), rng.random(1536)
        extra = (triangles[face,0]*(1-u[:,None]) + triangles[face,1]*(u*(1-v))[:,None]
                 + triangles[face,2]*(u*v)[:,None])
        points = np.concatenate((arrays[f'{i}__base__xyz'], extra)).astype(np.float32)
        if points.shape != (2048,3) or len(np.unique(points, axis=0)) != 2048:
            raise ValueError('2048 distinct on-CAD surface samples required')
        normals = np.concatenate((arrays[f'{i}__base__features'][:,:3], cross[face]/area[face,None]))
        relative = anchor @ poses[:4]
        history = np.einsum('tij,pj->tpi', relative[:,:3,:3], points) + relative[:,None,:3,3]
        feature = np.zeros((2048,18), np.float32)
        feature[:,:3] = normals
        feature[:,3:6] = (history[3]-history[2])*30
        feature[:,6:9] = (history[3]-2*history[2]+history[1])*900
        feature[:,9:12] = history[3]-history[0]
        feature[:,12] = 1.
        feature[:,13] = np.linalg.norm(feature[:,9:12],axis=-1) < .002
        # Preserve the original512 rows exactly in the matched input comparison.
        feature[:512] = arrays[f'{i}__base__features']
        arrays[f'{i}__dense_xyz'] = points
        arrays[f'{i}__dense_features'] = feature
        arrays[f'{i}__dense_normals'] = normals.astype(np.float32)
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output / 'fixture.npz', **arrays)
    meta.update(hand_vertices_per_side=778, object_surface_points=2048, seed=176,
        max_reconstructed_keypoint_error_m=max(errors), preparation_seconds=time.monotonic()-started,
        label_scope='Original128-query C labels only; added C queries are masked, no contact-quality claim',
        environment=dict(python=sys.version, torch=torch.__version__, device='CPU geometric skinning'))
    (args.output / 'metadata.json').write_text(json.dumps(meta, indent=2)+'\n')
    print(json.dumps({k:v for k,v in meta.items() if k!='records'}), flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['export','reconstruct'])
    p.add_argument('--code-root', type=Path)
    p.add_argument('--recipe', type=Path)
    p.add_argument('--input', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mano-root', type=Path)
    args=p.parse_args()
    (export if args.mode=='export' else reconstruct)(args)


if __name__=='__main__':
    main()
