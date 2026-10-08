"""Create weak GRAB lift stage sidecars without altering PointWorld geometry."""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
import time

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(TASK/'src'))

import numpy as np
from consequence_evaluator.progress_contracts import RunGuard, SCHEMA, HAND_ORDER, SEMANTICS, sha, write
from consequence_evaluator.grab_progress_labels import (DEFAULTS, propose, training_stage_weights,
    dense_labels, support_planes, plane_in_object_frame, lowest_clearance)


def review_svg(path, clearance, contacts, boundaries, end):
    """Small train-only annotation chart; no video or external viewer dependency."""
    n = len(clearance)
    width, height = 900, 280
    low, high = min(-.02, float(clearance.min())), max(.06, float(clearance.max()))
    xy = lambda t, v: (45+810*t/max(n-1, 1), 230-180*(v-low)/(high-low))
    points = ' '.join('%.2f,%.2f' % xy(i, float(v)) for i, v in enumerate(clearance))
    lines = ['<svg xmlns="http://www.w3.org/2000/svg" width="900" height="280">',
             '<rect width="900" height="280" fill="white"/>',
             '<text x="45" y="22">'+path.stem+' — frame index / clearance (m)</text>',
             '<polyline fill="none" stroke="#246" stroke-width="2" points="'+points+'"/>']
    for value in (0., .01, .03):
        y = xy(0, value)[1]
        lines.append('<path stroke="#bbb" d="M45 %.2f H855"/><text x="2" y="%.2f">%.2f</text>' % (y, y, value))
    for name, tick in zip(('b0', 'b1', 'b2', 'b3', 'end'), [*boundaries, end]):
        x = xy(tick, 0)[0]
        lines.append('<path stroke="#b53" d="M%.2f 35 V240"/><text x="%.2f" y="255">%s:%d</text>' % (x, x, name, tick))
    for i in np.flatnonzero(contacts):
        x = xy(int(i), 0)[0]
        lines.append('<path stroke="#3a5" d="M%.2f 265 v5"/>' % x)
    lines.append('</svg>')
    path.write_text('\n'.join(lines)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--wm-input-manifest', type=Path, default=ROOT/'outputs/cm-pointflow-effect-pretrain/pointworld-main3-20261007/train-action/input_manifest.json')
    p.add_argument('--wm-checkpoint', type=Path, default=ROOT/'outputs/cm-pointflow-effect-pretrain/pointworld-main3-20261007/train-action/best.pt')
    p.add_argument('--raw-root', type=Path, default=ROOT/'data/raw_data/GRAB')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seconds', type=int, default=600)
    a = p.parse_args()
    if not 1 <= a.seconds <= 1800:
        p.error('bounded <=1800s metadata preparation')
    g = RunGuard(a.output, a.seconds, .5)
    out = g.output
    meta = dict(schema=SCHEMA, status='RUNNING', task='consequence-evaluator', run_id=out.name,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
        fps=30, history=4, horizon=24, hand_order=HAND_ORDER, hand_semantics=SEMANTICS,
        action_semantics='measured_hand_trajectory', effect_semantics='current_target_inverse_times_future_target',
        annotation_version='auto-weak-ref4-v1', annotation_review_status='pending',
        training_allowed=False, probe_training_allowed=True, label_config=DEFAULTS,
        cpu_reason='reuse immutable geometry, read contacts/table and compute weak annotation statistics',
        wm_checkpoint=str(a.wm_checkpoint.resolve()), wm_step_expected=46000,
        records=[], rejections=[], frozen_inputs={}, deadline=g.deadline)
    try:
        import trimesh
        from scipy.spatial.transform import Rotation
        g.pin(Path(__file__))
        for name in ('progress_contracts.py', 'grab_progress_labels.py'):
            g.pin(TASK/'src/consequence_evaluator'/name)
        g.pin(a.wm_input_manifest)
        g.pin(a.wm_checkpoint)
        wm = json.loads(a.wm_input_manifest.read_text())
        descriptor = next(x for x in wm['source_manifest']['sources'] if x['name'] == 'grab')
        pack = Path(descriptor['root'])
        g.pin(pack/'processed/manifest.json', descriptor['manifest_sha256'])
        native = json.loads((pack/'processed/manifest.json').read_text())
        if (native['hand_order'] != HAND_ORDER or native['hand_semantics'] != SEMANTICS
                or native['fps'] != 30 or native['status'] != 'COMPLETED'):
            raise ValueError('native geometry contract mismatch')
        sequence_split = {}
        for split, index in descriptor['indices'].items():
            g.pin(index['path'], index['sha256'])
            rows = np.load(index['path'], mmap_mode='r')
            for sid in np.unique(rows[:, 0]):
                name = native['sequences'][int(sid)]
                if name in sequence_split and sequence_split[name] != split:
                    raise ValueError('PointWorld sequence split leak')
                sequence_split[name] = split
        hand_ids = {}
        for side in ('rhand', 'lhand'):
            path = a.raw_root/'tools/smplx_correspondence'/(side+'_smplx_ids.npy')
            g.pin(path)
            hand_ids[side] = np.load(path)
        hand_union = np.union1d(hand_ids['rhand'], hand_ids['lhand'])
        meshes = {}
        candidates = []
        (out/'labels').mkdir()
        (out/'review').mkdir()
        for native_id, record in enumerate(native['records']):
            if record['source'] != 'grab' or 'lift' not in Path(record['source_sequence']).stem.split('_'):
                continue
            g.check()
            path = Path(record['source_path'])
            g.pin(path, record['source_sha256'])
            with np.load(path, allow_pickle=True) as z:
                raw = {k: z[k].item() if z[k].shape == () else z[k] for k in z.files}
            if raw['motion_intent'] != 'lift' or float(raw['framerate']) != 120:
                meta['rejections'].append(dict(sequence=record['sequence'], reason='not_native_120Hz_lift'))
                continue
            seq = pack/'processed/sequences'/record['sequence']
            arrays = {}
            for name in ('source_frame_ids', 'timestamps', 'hand', 'hand_valid', 'poses', 'pose_valid'):
                file = seq/(name+'.npy')
                g.pin(file, record['array_sha256'][file.name])
                arrays[name] = np.load(file, mmap_mode='r')
            ids = np.asarray(arrays['source_frame_ids'])
            if (ids.dtype.kind not in 'iu' or (np.diff(ids) != 4).any()
                    or ids[0] < 0 or ids[-1] >= int(raw['n_frames'])
                    or not np.allclose(arrays['timestamps'], ids/120., atol=1e-6, rtol=0)):
                raise ValueError('raw contact frame alignment mismatch')
            poses = np.asarray(arrays['poses'][:, 0])
            expected_R = Rotation.from_rotvec(raw['object']['params']['global_orient'][ids]).as_matrix().swapaxes(-1,-2)
            if (not np.allclose(poses[:, :3, :3], expected_R, atol=1e-6)
                    or not np.allclose(poses[:, :3, 3], raw['object']['params']['transl'][ids], atol=1e-6)):
                raise ValueError('native/raw object pose mismatch')
            for field, meshkey in (('table','table_mesh'), ('object','object_mesh')):
                meshfile = (a.raw_root/raw[field][meshkey]).resolve()
                if str(meshfile) not in meshes:
                    g.pin(meshfile)
                    meshes[str(meshfile)] = np.asarray(trimesh.load(meshfile, process=False).vertices)
            tp = raw['table']['params']
            planes = support_planes(meshes[str((a.raw_root/raw['table']['table_mesh']).resolve())],
                Rotation.from_rotvec(tp['global_orient'][ids]).as_matrix().swapaxes(-1,-2), tp['transl'][ids])
            context_plane = plane_in_object_frame(planes, poses)
            clearance = lowest_clearance(meshes[str((a.raw_root/raw['object']['object_mesh']).resolve())], context_plane)
            contact = raw['contact']['body'][ids] > 0
            right = contact[:, hand_ids['rhand']].any(1)
            left = contact[:, hand_ids['lhand']].any(1)
            other_ids = np.setdiff1d(np.arange(contact.shape[1]), hand_union)
            other = contact[:, other_ids].any(1)
            boundary, reason = propose(clearance, right, left, other, arrays['hand'][:,0,0])
            if record['sequence'] not in sequence_split:
                reason = 'no_frozen_PointWorld_split_membership'
            if reason:
                meta['rejections'].append(dict(sequence=record['sequence'], reason=reason))
                continue
            candidate = dict(sequence=record['sequence'], native_sequence_id=native_id,
                object=raw['obj_name'], source_sequence=record['source_sequence'], split=sequence_split[record['sequence']],
                source_path=str(path), source_sha256=record['source_sha256'], target_slot=0,
                raw_boundaries=ids[boundary['boundaries']].tolist(), **boundary)
            if not (np.asarray(arrays['pose_valid'])[boundary['boundaries'][0]:boundary['end']]).all():
                meta['rejections'].append(dict(sequence=record['sequence'], reason='invalid_object_pose'))
                continue
            candidate['_signals'] = (context_plane, clearance, right, left, other, ids, arrays['hand_valid'])
            candidates.append(candidate)
            if len(candidates)%20 == 0:
                print(json.dumps(dict(stage='label',accepted=len(candidates),rejected=len(meta['rejections']),elapsed=time.time()-g.started)),flush=True)
        train = [x for x in candidates if x['split'] == 'train']
        if not train:
            raise ValueError('no eligible training lifts')
        alpha = training_stage_weights([x['boundaries'] for x in train])
        indices = {split: [] for split in ('train','val','test')}
        objects = {split: set() for split in indices}
        for sid, candidate in enumerate(candidates):
            plane, clearance, right, left, other, ids, hv = candidate.pop('_signals')
            stage, progress, valid = dense_labels(len(ids), candidate['boundaries'], candidate['end'], alpha)
            np.savez_compressed(out/'labels'/(candidate['sequence']+'.npz'), stage=stage, progress=progress,
                label_valid=valid, source_frame_ids=ids, support_plane_current_frame=plane,
                clearance=clearance, right_contact=right, left_contact=left, other_contact=other)
            candidate['labels_sha256'] = sha(out/'labels'/(candidate['sequence']+'.npz'))
            for tick in range(candidate['boundaries'][0]+3, candidate['end']-24):
                if np.asarray(hv[tick-3:tick+25]).all():
                    indices[candidate['split']].append([sid,tick])
            objects[candidate['split']].add(candidate['object'])
            if candidate['split'] == 'train' and len(list((out/'review').glob('*.svg'))) < 12:
                review_svg(out/'review'/(candidate['sequence']+'.svg'), clearance, right, candidate['boundaries'],candidate['end'])
            meta['records'].append(candidate)
        for split, rows in indices.items():
            np.save(out/('index_'+split+'.npy'), np.asarray(rows,np.int64).reshape(-1,2))
        if any(objects[a]&objects[b] for a,b in [('train','val'),('train','test'),('val','test')]):
            raise ValueError('object-level split leak')
        g.check(full=True)
        meta.update(status='COMPLETED', native_descriptor=descriptor, stage_weights=alpha.tolist(),
            stage_weights_source='train mean per-sequence pre-hold duration proportions',
            wm_checkpoint_sha256=sha(a.wm_checkpoint), frozen_inputs=g.inputs,
            splits={s:dict(sequences=sum(x['split']==s for x in candidates),windows=len(indices[s]),objects=sorted(objects[s])) for s in indices},
            rejection_counts=dict(Counter(x['reason'] for x in meta['rejections'])), elapsed_seconds=time.time()-g.started,
            limitation='automatic unreviewed geometric/contact weak labels; not physical success probabilities')
        write(out/'review/candidates.json',dict(annotation_review_status='pending',config=DEFAULTS,records=meta['records']))
        write(out/'manifest.json',meta)
        print(json.dumps({k:meta[k] for k in ('status','splits','stage_weights','rejection_counts','elapsed_seconds')}),flush=True)
    except BaseException as error:
        meta.update(status='FAILED',error=repr(error),elapsed_seconds=time.time()-g.started,frozen_inputs=g.inputs)
        write(out/'manifest.json',meta)
        raise


if __name__ == '__main__':
    main()
