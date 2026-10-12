#!/usr/bin/env python3
"""Admit native official-TRAIN hands only; never admit offline scene teachers."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess

import h5py
import numpy as np

JOINTS = ['Hand', 'ThumbKnuckle', 'IndexFingerKnuckle', 'MiddleFingerKnuckle',
          'RingFingerKnuckle', 'LittleFingerKnuckle', 'ThumbTip', 'IndexFingerTip',
          'MiddleFingerTip', 'RingFingerTip', 'LittleFingerTip']


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--download-manifest', type=Path, action='append', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--confidence', type=float, default=.5)
    p.add_argument('--split-seed', type=int, default=283)
    a = p.parse_args()
    if not 0 < a.confidence <= 1 or a.output.exists():
        raise ValueError('new output and confidence in (0,1] required')
    records, rejected = [], []
    for source in a.download_manifest:
        manifest = json.loads(source.read_text())
        if manifest.get('status') != 'COMPLETED' or manifest.get('split') != 'official_train':
            raise ValueError('only completed official TRAIN acquisition admitted')
        if manifest['url'] not in {f'https://ml-site.cdn-apple.com/datasets/egodex/part{i}.zip' for i in range(1, 6)}:
            raise ValueError('unknown archive provenance')
        for item in manifest['files']:
            if not item['member'].endswith('.hdf5'):
                continue
            member = Path(item['member'])
            if member.is_absolute() or '..' in member.parts:
                raise ValueError('unsafe source member')
            raw = source.parent/'raw'/member
            if raw.stat().st_size != item['bytes'] or sha(raw) != item['sha256']:
                raise ValueError('native HDF5 acquisition identity changed')
            with h5py.File(raw, 'r') as f:
                if 'confidences' not in f:
                    rejected.append(dict(member=str(member), raw_sha256=item['sha256'],
                                         reason='no native confidences; not imputed'))
                    continue
                names = [[side+j for j in JOINTS] for side in ('right', 'left')]
                matrices = np.stack([np.stack([f['transforms'][n][:] for n in side], 1) for side in names], 1)
                confidence = np.stack([np.stack([f['confidences'][n][:] for n in side], 1) for side in names], 1)
                camera = f['transforms/camera'][:]
                session = str(f.attrs['session_name'])
            n = len(camera)
            video = raw.with_suffix('.mp4')
            clock = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                '-show_entries', 'stream=avg_frame_rate,nb_frames', '-of', 'json', str(video)], text=True))['streams'][0]
            numerator, denominator = map(int, clock['avg_frame_rate'].split('/'))
            if abs(numerator/denominator-30) > 1e-4 or int(clock['nb_frames']) != n:
                raise ValueError('native annotation/video 30Hz clock mismatch')
            if matrices.shape != (n, 2, 11, 4, 4) or confidence.shape != (n, 2, 11):
                raise ValueError('native 2x11 contract mismatch')
            if not np.isfinite(camera).all() or not np.allclose(camera[:, 3], [0, 0, 0, 1], atol=1e-4):
                raise ValueError('invalid camera transform')
            rotations = camera[:, :3, :3]
            if not np.allclose(rotations.swapaxes(-1, -2)@rotations, np.eye(3), atol=1e-3):
                raise ValueError('nonrigid native camera')
            if not np.isfinite(confidence).all() or np.any((confidence < 0) | (confidence > 1)):
                raise ValueError('invalid native confidences')
            valid = np.isfinite(matrices).all((-1, -2)) & (confidence >= a.confidence)
            hand = matrices[..., :3, 3].astype('float32')
            hand[~valid] = 0
            # Index qualification uses label availability, never future motion.
            # Occluded tips stay masked. Six persistent measured joints in one
            # hand suffice; requiring all eleven would discard nearly all data.
            persistent = np.lib.stride_tricks.sliding_window_view(valid, 28, axis=0).all(-1)
            full = (persistent.sum(-1) >= 6).any(-1)
            ticks = np.flatnonzero(full).astype('int32') + 3
            native_current = (hand[ticks]-camera[ticks, None, None, :3, 3]) @ camera[ticks, None, :3, :3]
            current_valid = valid[ticks]
            outside = ((np.abs(native_current) > 1.99).any(-1) & current_valid).any((-1, -2))
            rejected_workspace_windows = int(outside.sum())
            ticks = ticks[~outside]
            record = dict(member=str(member), raw_hdf5=str(raw.resolve()), raw_sha256=item['sha256'],
                          acquisition_manifest=str(source.resolve()), acquisition_sha256=sha(source),
                          session=session, frames=n, windows=len(ticks), observed_fraction=float(valid.mean()),
                          rejected_workspace_windows=rejected_workspace_windows,
                          camera_clock=clock, hand=hand, valid=valid, camera=camera.astype('float32'), ticks=ticks)
            if not len(ticks):
                rejected.append({k:v for k,v in record.items() if k not in ('hand', 'valid', 'camera', 'ticks')})
            else:
                records.append(record)
    if len({r['member'] for r in records}) != len(records):
        raise ValueError('duplicate raw records')
    sessions = sorted({r['session'] for r in records}, key=lambda s: hashlib.sha256(f'{a.split_seed}:{s}'.encode()).hexdigest())
    if len(sessions) < 2:
        raise ValueError('need separate train/dev recording sessions')
    val_sessions = set(sessions[:max(1, math.ceil(.2*len(sessions)))])
    a.output.mkdir(parents=True)
    rows = dict(train=[], val=[])
    for i, record in enumerate(records):
        split = 'val' if record['session'] in val_sessions else 'train'
        dest = a.output/f'record{i:03d}.npz'
        np.savez_compressed(dest, hand=record.pop('hand'), valid=record.pop('valid'), camera=record.pop('camera'))
        ticks = record.pop('ticks')
        record.update(split=split, path=dest.name, sha256=sha(dest))
        rows[split].extend((i, int(tick)) for tick in ticks)
    for split in rows:
        if not rows[split]:
            raise ValueError('empty admitted split')
        np.save(a.output/f'index_{split}.npy', np.asarray(rows[split], dtype='int32'))
    result = dict(schema='goal-point-wam.egodex-native-actor.v1', status='COMPLETED',
                  training_allowed=True, supervision='native_hand_trajectory_only',
                  object_goal_available=False, world_supervision_available=False, scene_teacher_admitted=False,
                  fps=30, history=4, horizon=24, confidence_threshold=a.confidence,
                  missing_confidence_policy='reject record; never impute tracked confidence',
                  window_admission='at least six confidence-qualified persistent joints in one hand across H4+K24; other points masked',
                  confidence_threshold_source='declared conservative recipe, not official recommendation',
                  joint_suffixes=JOINTS, hand_order=['right', 'left'], split_seed=a.split_seed,
                  split_unit='session_name', val_sessions=sorted(val_sessions), records=records,
                  rejected_records=rejected, train_windows=len(rows['train']), val_windows=len(rows['val']),
                  index_hashes={k:sha(a.output/f'index_{k}.npy') for k in rows},
                  input_frame='one CURRENT camera frame for all history and future points',
                  engineering_admission='new actor-only admission; original acquisition/teacher flags unchanged',
                  code_sha256=sha(Path(__file__)))
    (a.output/'manifest.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('records', 'rejected_records')}))


if __name__ == '__main__':
    main()
