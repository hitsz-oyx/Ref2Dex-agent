#!/usr/bin/env python3
"""Stream verified annotation arrivals into bounded 30Hz caches and full-corpus indices."""
import argparse
import hashlib
import importlib.util
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import trimesh

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK / 'src'))
from oakink_wm.data import local_objects
spec = importlib.util.spec_from_file_location('geometry_audit', TASK / 'tools/audit/audit_oakink2_sequences.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def write_json(path, value):
    temp = path.with_suffix('.json.part')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def rigid_valid(T):
    finite = np.isfinite(T).all((-1, -2))
    R = np.where(finite[..., None, None], T[..., :3, :3], np.eye(3))
    return finite & (np.abs(R.swapaxes(-1, -2) @ R - np.eye(3)).max((-1, -2)) < 1e-3) & (np.abs(np.linalg.det(R) - 1) < 1e-3) & (np.abs(T[..., 3, :] - [0, 0, 0, 1]).max(-1) < 1e-6)


def canonical(dataset, obj, out):
    path = out / (obj + '.npz')
    if path.exists(): return
    matches = []
    for source in ('object_repair', 'object_raw'):
        matches = sorted((dataset / source / 'align_ds' / obj).glob('*.ply')) + sorted((dataset / source / 'align_ds' / obj).glob('*.obj'))
        if matches: break
    if not matches: raise FileNotFoundError('mesh ' + obj)
    mesh = trimesh.load(matches[0], process=False, force='mesh', skip_materials=True)
    tri = np.asarray(mesh.triangles)
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = np.linalg.norm(cross, axis=-1)
    if not np.isfinite(tri).all() or area.sum() <= 0: raise ValueError('invalid mesh ' + obj)
    rng = np.random.default_rng(int(hashlib.sha256(obj.encode()).hexdigest()[:8], 16))
    faces = rng.choice(len(tri), 512, p=area / area.sum())
    chosen = tri[faces]
    u, v = np.sqrt(rng.random(512)), rng.random(512)
    points = chosen[:, 0] * (1-u[:, None]) + chosen[:, 1] * (u*(1-v))[:, None] + chosen[:, 2] * (u*v)[:, None]
    normals = cross[faces] / area[faces, None]
    radius = np.sqrt(np.mean(np.sum((points - points.mean(0))**2, axis=-1)))
    np.savez_compressed(path, points=points.astype('float32'), normals=normals.astype('float32'), radius=radius, mesh_path=str(matches[0]))


def process_sequence(root, name, layers, device, reuse):
    seq = Path(name).stem
    dest = root / 'processed/sequences' / seq
    if (dest / 'meta.json').exists(): return json.loads((dest / 'meta.json').read_text())
    with (root / 'download' / name).open('rb') as f: anno = pickle.load(f)
    rawids = np.asarray(anno['mocap_frame_id_list'], np.int64)
    assert np.all(np.diff(rawids) > 0)
    selected = np.flatnonzero(rawids % 4 == 0)
    ids = rawids[selected]
    if len(ids) < 28: raise ValueError('too short ' + seq)
    old = reuse / 'audit/sequences' / (seq + '.npz')
    if old.exists():
        with np.load(old) as f:
            assert np.array_equal(f['frame_ids'], rawids)
            rawhand, rawvalid = f['hand_xyz'], f['hand_valid']
    else:
        rawhand, rawvalid, _ = audit.dense_mano(anno, rawids, layers, device)
    hand, valid = rawhand[selected], rawvalid[selected]
    hjump = np.linalg.norm(np.diff(rawhand, axis=0), axis=-1).max(-1)
    hstable = (rawvalid[1:] == rawvalid[:-1]).all(-1)
    hbad = (np.diff(rawids) != 1) | (~hstable) | ((hjump > .05) & rawvalid[1:] & rawvalid[:-1]).any(-1)
    hprefix = np.concatenate(([0], np.cumsum(hbad)))
    hand_edge_good = np.ones(len(ids), bool)
    hand_edge_good[1:] = hprefix[selected[1:]] == hprefix[selected[:-1]]
    objects = list(anno['obj_list'])
    n, m = len(ids), len(objects)
    poses = np.full((n, m, 4, 4), np.nan, np.float32)
    pose_valid = np.zeros((n, m), bool)
    clean = np.zeros((n, m), bool)
    program = np.zeros((n, m), bool)
    near = np.zeros((n, m), bool)
    anchors, jump_count = [], 0
    for j, obj in enumerate(objects):
        canonical(root / 'dataset', obj, root / 'processed/canonical')
        T = np.full((len(rawids), 4, 4), np.nan)
        mapping = anno['obj_transf'].get(obj, {})
        for k, fid in enumerate(rawids):
            if int(fid) in mapping: T[k] = mapping[int(fid)]
        rigid = rigid_valid(T)
        edgevalid = rigid[1:] & rigid[:-1]
        R = T[1:, :3, :3] @ T[:-1, :3, :3].transpose(0, 2, 1)
        angle = np.arccos(np.clip((np.trace(R, axis1=1, axis2=2)-1)/2, -1, 1))
        jump = (np.linalg.norm(np.diff(T[:, :3, 3], axis=0), axis=-1) > .05) | (angle > .2)
        bad = (~edgevalid) | jump | (np.diff(rawids) != 1)
        prefix = np.concatenate(([0], np.cumsum(bad)))
        clean[:, j] = rigid[selected]
        clean[1:, j] &= prefix[selected[1:]] == prefix[selected[:-1]]
        poses[:, j], pose_valid[:, j] = T[selected], rigid[selected]
        pm, _, present = audit.program_mask(root / 'dataset', seq, obj, ids)
        if pm.any(): anchors.append(j)
        program[:, j] = pm
        jump_count += int((jump & edgevalid).sum())
        with np.load(root / 'processed/canonical' / (obj + '.npz')) as f: points = f['points']
        for b in range(0, n, 128):
            rt = torch.as_tensor(poses[b:b+128, j, :3, :3], device=device)
            tt = torch.as_tensor(poses[b:b+128, j, :3, 3], device=device)
            cloud = torch.as_tensor(points, device=device) @ rt.transpose(1, 2) + tt[:, None]
            h = torch.as_tensor(hand[b:b+128].reshape(-1, 22, 3), device=device)
            distance = torch.cdist(h, cloud).amin(-1).reshape(-1, 2, 11)
            hv = torch.as_tensor(valid[b:b+128], device=device)
            distance = distance.masked_fill(~hv[:, :, None], float('inf')).amin((1, 2))
            near[b:b+128, j] = (distance < .05).cpu().numpy() & pose_valid[b:b+128, j]
    rows = []
    # Prefix sums make continuity checks independent of horizon and raw gaps.
    cp = np.vstack((np.zeros((1, m), int), np.cumsum(~clean, axis=0)))
    hp = np.concatenate(([0], np.cumsum(~hand_edge_good)))
    idp = np.concatenate(([0], np.cumsum(np.diff(ids) != 4)))
    for anchor in anchors:
        for t in range(3, n - 24):
            if idp[t+24] != idp[t-3] or hp[t+25] != hp[t-2]: continue
            if not valid[t].any() or not (valid[t-3:t+25] == valid[t]).all(): continue
            if not pose_valid[t, anchor]: continue
            local = local_objects(poses[t], pose_valid[t], anchor)
            # clean[t-3] includes preceding irrelevant edge; check exact28 frames
            # pose validity and only edges connecting the selected28 frames.
            if not pose_valid[t-3:t+25, local].all(): continue
            if np.any(cp[t+25, local] != cp[t-2, local]): continue
            delta = np.linalg.norm(poses[t+1:t+25, anchor, :3, 3] - poses[t, anchor, :3, 3], axis=-1)
            r = poses[t+1:t+25, anchor, :3, :3] @ poses[t, anchor, :3, :3].T
            a = np.arccos(np.clip((np.trace(r, axis1=1, axis2=2)-1)/2, -1, 1))
            moving = bool((delta > .002).any() or (a > .02).any())
            category = 0 if moving else (1 if program[t, anchor] or near[t, anchor] else 2)
            rows.append([anchor, t, category])
    dest.mkdir(parents=True, exist_ok=True)
    for key, value in dict(hand=hand, hand_valid=valid, poses=poses, pose_valid=pose_valid,
                           program=program, near=near, frame_ids=ids).items(): np.save(dest / (key + '.npy'), value)
    rows = np.asarray(rows, dtype=np.int32).reshape(-1, 3)
    np.save(dest / 'rows.npy', rows)
    result = dict(sequence=seq, objects=objects, frames=n, windows=len(rows),
                  strata=np.bincount(rows[:, 2], minlength=3).tolist(), object_jump_steps=jump_count,
                  anchors=anchors, source_sha256=hashlib.sha256((root / 'download' / name).read_bytes()).hexdigest())
    write_json(dest / 'meta.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--reuse-dir', type=Path, required=True)
    p.add_argument('--mano-root', type=Path, required=True)
    p.add_argument('--seconds', type=int, default=10800)
    p.add_argument('--limit', type=int, default=627)
    p.add_argument('--selection-manifest', type=Path)
    args = p.parse_args()
    root = args.run_dir.resolve()
    start = time.monotonic()
    torch.set_num_threads(2)
    layers = [audit.ManoLayer(mano_assets_root=str(args.mano_root), rot_mode='quat', side=side,
                             center_idx=0, use_pca=False, flat_hand_mean=True).cuda() for side in ('right', 'left')]
    out = root / 'processed'
    (out / 'sequences').mkdir(parents=True, exist_ok=True)
    (out / 'canonical').mkdir(exist_ok=True)
    ann = sorted(json.loads((root / 'anno_tree.json').read_text()), key=lambda x: x['path'])
    if args.selection_manifest:
        names = set(json.loads(args.selection_manifest.read_text())['selection'])
        ann = [x for x in ann if x['path'] in names]
    ann = ann[:args.limit]
    pending = {x['path'] for x in ann}
    done = {}
    while pending:
        if time.monotonic() - start > args.seconds: raise TimeoutError('preprocessing deadline')
        try: download = json.loads((root / 'download_manifest.json').read_text())
        except (FileNotFoundError, json.JSONDecodeError): time.sleep(2); continue
        ready = sorted(pending.intersection(download['files']))
        if not ready:
            if download['status'] == 'FAILED': raise RuntimeError(download['error'])
            time.sleep(3)
            continue
        for name in ready:
            result = process_sequence(root, name, layers, 'cuda:0', args.reuse_dir.resolve())
            done[name] = result
            pending.remove(name)
            write_json(out / 'progress.json', dict(completed=len(done), total=len(ann),
                       windows=sum(x['windows'] for x in done.values()), elapsed_seconds=time.monotonic()-start))
            print(json.dumps(dict(completed=len(done), total=len(ann), sequence=result['sequence'],
                                  windows=result['windows'], elapsed_seconds=time.monotonic()-start)), flush=True)
    sequence_names = [Path(x['path']).stem for x in ann]
    stats, split_rows = {}, {'train': [], 'val': [], 'test': []}
    split_sequences = {k: [] for k in split_rows}
    for i, seq in enumerate(sequence_names):
        score = int(hashlib.sha256(('210:' + seq).encode()).hexdigest()[:8], 16) % 100
        split = 'train' if score < 80 else ('val' if score < 90 else 'test')
        split_sequences[split].append(seq)
        rows = np.load(out / 'sequences' / seq / 'rows.npy')
        split_rows[split].append(np.column_stack((np.full(len(rows), i, np.int32), rows)))
    for split, parts in split_rows.items():
        combined = np.concatenate(parts) if parts else np.empty((0, 4), np.int32)
        np.save(out / ('index_' + split + '.npy'), combined)
        stats[split] = dict(sequences=len(split_sequences[split]), windows=len(combined),
                            strata=np.bincount(combined[:, 3], minlength=3).tolist())
    manifest = dict(status='COMPLETED', fps=30, horizon=24, history=4, source_fps=120,
                    source_revision=download['revision'], sequences=sequence_names, split_sequences=split_sequences,
                    statistics=stats, elapsed_seconds=time.monotonic()-start,
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    sampling=[.6, .2, .2], scope='observational future hands; no causal or policy utility claim')
    write_json(out / 'manifest.json', manifest)
    print(json.dumps(manifest['statistics']), flush=True)


if __name__ == '__main__': main()
