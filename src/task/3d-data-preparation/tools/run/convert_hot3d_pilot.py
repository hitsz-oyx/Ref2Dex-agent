"""Bounded native HOT3D conversion; no RGB, learned estimator, or fake labels."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from pose_mesh_data.windows import SCHEMA, validate_sequence, xyzquat_to_matrix


def masks(path, stream):
    with path.open() as f:
        return {int(r['timestamp[ns]']): r['mask'] == 'True'
                for r in csv.DictReader(f) if r['stream_id'] == stream}


def mano_models(root):
    # Legacy licensed MANO pickles import chumpy's old NumPy aliases.
    for name, value in [('bool', bool), ('int', int), ('float', float),
                        ('complex', complex), ('object', object), ('str', str), ('unicode', str)]:
        if name not in np.__dict__:
            setattr(np, name, value)
    import smplx
    import torch
    models = [smplx.create(str(root / ('MANO_' + side.upper() + '.pkl')), 'mano',
                          use_pca=True, num_pca_comps=15, is_rhand=side == 'right')
              for side in ('right', 'left')]
    if torch.sum(torch.abs(models[1].shapedirs[:, 0] - models[0].shapedirs[:, 0])) < 1:
        models[1].shapedirs[:, 0] *= -1
    return models


def geometry(path, scale=1.):
    import trimesh
    mesh = trimesh.load(str(path), process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    vertices = np.asarray(mesh.vertices)*scale
    _, ids = np.unique(vertices, axis=0, return_index=True)
    ids = np.sort(ids)
    ids = ids[np.linspace(0, len(ids)-1, min(512, len(ids)), dtype=int)]
    surface = vertices[ids].astype(np.float32)
    normals = np.asarray(mesh.vertex_normals)[ids].astype(np.float32)
    if len(surface) < 32 or not np.isfinite(surface).all():
        raise ValueError('insufficient finite unique CAD vertices')
    return dict(surface=surface, normals=normals,
                query=surface[np.linspace(0, len(surface)-1, 32, dtype=int)])


def convert(path, models, frames):
    import torch
    metadata = json.loads((path / 'metadata.json').read_text())
    stream = '214-1' if metadata['headset'] == 'Aria' else '1201-1'
    qa = masks(path / 'masks/mask_qa_pass.csv', stream)
    hand_available = masks(path / 'masks/mask_hand_pose_available.csv', stream)
    object_available = masks(path / 'masks/mask_object_pose_available.csv', stream)
    timestamps = np.array(sorted(t for t in qa if t > 0), np.int64)
    good = [t for t in timestamps if qa[t] and hand_available.get(t) and object_available.get(t)]
    if not good:
        raise ValueError('no native QA-passing annotation timestamp')
    start = good[0]
    count = min(frames, int((timestamps[-1]-start) / 1e9 * 30)+1)
    grid = start + np.rint(np.arange(count) * 1e9/30).astype(np.int64)
    i = np.searchsorted(timestamps, grid).clip(0, len(timestamps)-1)
    previous = (i-1).clip(0)
    i = np.where(abs(timestamps[previous]-grid) < abs(timestamps[i]-grid), previous, i)
    native = timestamps[i]
    residual = native-grid
    aligned = abs(residual) <= 2_000_000
    if len(np.unique(native)) != len(native):
        raise ValueError('resampling would duplicate a native annotation')
    hand_rows = {}
    with (path / 'mano_hand_pose_trajectory.jsonl').open() as f:
        wanted = set(native.tolist())
        for line in f:
            row = json.loads(line)
            if row['timestamp_ns'] in wanted:
                hand_rows[row['timestamp_ns']] = row['hand_poses']
    object_rows = {}
    with (path / 'dynamic_objects.csv').open() as f:
        for row in csv.DictReader(f):
            t = int(row['timestamp[ns]'])
            if t in wanted:
                object_rows[t, row['object_uid']] = row
    uids = metadata['object_uids']
    if not 1 <= len(uids) <= 6:
        raise ValueError('explicit one-to-six-object scene required')
    hand = np.full((count, 2, 11, 3), np.nan, np.float32)
    hand_valid = np.zeros(hand.shape[:-1], bool)
    poses = np.full((count, len(uids), 4, 4), np.nan, np.float32)
    known = np.zeros(poses.shape[:2], bool)
    quality = np.array([aligned[k] and qa.get(t, False) for k, t in enumerate(native)])
    for side, model in enumerate(models):
        selected = [(k, hand_rows.get(t, {}).get(str(1-side))) for k, t in enumerate(native)
                    if quality[k] and hand_available.get(t, False)]
        selected = [(k, r) for k, r in selected if r is not None]
        for offset in range(0, len(selected), 128):
            batch = selected[offset:offset+128]
            orient = Rotation.from_quat([r['wrist_xform']['q_wxyz'][1:]
                                        + r['wrist_xform']['q_wxyz'][:1] for _, r in batch]).as_rotvec()
            with torch.no_grad():
                out = model(betas=torch.tensor([r['betas'] for _, r in batch], dtype=torch.float32),
                            global_orient=torch.tensor(orient, dtype=torch.float32),
                            hand_pose=torch.tensor([r['pose'] for _, r in batch], dtype=torch.float32),
                            transl=torch.tensor([r['wrist_xform']['t_xyz'] for _, r in batch], dtype=torch.float32))
            points = torch.cat((out.joints[:, [0, 13, 1, 4, 10, 7]],
                                out.vertices[:, [744, 320, 443, 554, 671]]), 1).numpy()
            ids = [k for k, _ in batch]
            hand[ids, side] = points
            hand_valid[ids, side] = True
    for k, t in enumerate(native):
        if not quality[k] or not object_available.get(t, False):
            continue
        for slot, uid in enumerate(uids):
            row = object_rows.get((t, uid))
            if row:
                values = [float(row['t_wo_' + axis + '[m]']) for axis in 'xyz']
                values += [float(row['q_wo_' + axis]) for axis in 'xyzw']
                poses[k, slot] = xyzquat_to_matrix(values)
                known[k, slot] = True
    sequence = dict(time=np.arange(count, dtype=np.float64)/30, hand=hand, hand_valid=hand_valid,
                    object_poses=poses, object_pose_valid=known, segment=np.zeros(count, np.int64),
                    native_timestamp_ns=native, grid_timestamp_ns=grid, alignment_residual_ns=residual,
                    native_qa_pass=quality)
    validate_sequence(sequence)
    report = dict(sequence=path.name, participant=metadata['participant_id'], stream=stream,
                  frames=count, max_alignment_residual_ms=float(abs(residual).max()/1e6),
                  alignment_limit_ms=2, resampling='nearest native timestamp; no pose interpolation',
                  qa_passing_frames=int(quality.sum()), native_hands=int(hand_valid.any(-1).sum()),
                  known_object_frames=int(known.sum()), units='m', training_allowed=False)
    return sequence, metadata, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--mano-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=512)
    args = parser.parse_args()
    if not 28 <= args.frames <= 1800:
        parser.error('bounded frames28..1800 required')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'sequences').mkdir(); (args.output/'geometry').mkdir()
    models = mano_models(args.mano_root)
    import torch
    torch.set_num_threads(4)
    manifest = dict(schema=SCHEMA, units='m', fps=30, training_allowed=False, objects=[], sequences=[])
    reports, seen = [], set()
    assets = args.input/'native/assets/assets'
    for family, split in [('aria', 'train'), ('quest', 'dev')]:
        path = next((args.input/'native'/family).iterdir())
        sequence, metadata, report = convert(path, models, args.frames)
        np.savez_compressed(args.output/'sequences'/f'{path.name}.npz', **sequence)
        for uid in metadata['object_uids']:
            if uid not in seen:
                np.savez_compressed(args.output/'geometry'/f'{uid}.npz', **geometry(assets/f'{uid}.glb'))
                manifest['objects'].append(dict(key=uid, path=f'geometry/{uid}.npz'))
                seen.add(uid)
        manifest['sequences'].append(dict(path=f'sequences/{path.name}.npz', format='canonical',
            objects=metadata['object_uids'], target_slot=0, target_selection='first metadata object; engineering only',
            source_id=6, profile='human-mano-native11', split=split,
            group_id='HOT3D/'+metadata['participant_id']))
        reports.append(report)
    manifest['limitations'] = ['Two-participant engineering split confounds headset with participant.',
                              'Target role not yet a manipulation/contact annotation.',
                              'Nearest 30Hz alignment, native clocks retained; no interpolated labels.']
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    (args.output/'conversion-audit.json').write_text(json.dumps(reports, indent=2)+'\n')
    print(json.dumps(reports, indent=2))


if __name__ == '__main__':
    main()
