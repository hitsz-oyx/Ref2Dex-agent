"""Decode official TACO parameters in a bounded CPU FK engineering pilot."""
import argparse
import io
import json
import os
from pathlib import Path
import pickle
import sys

import numpy as np
import torch
from convert_hot3d_pilot import geometry

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'src'))
from pose_mesh_data.windows import validate_sequence


class CPUReader(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda value: torch.load(io.BytesIO(value), map_location='cpu', weights_only=False)
        return super().find_class(module, name)


def load(path):
    with path.open('rb') as f:
        return CPUReader(f).load()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--hot3d-corpus', type=Path, required=True)
    p.add_argument('--mano-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    for name, value in [('bool', bool), ('int', int), ('float', float), ('complex', complex),
                        ('object', object), ('str', str), ('unicode', str)]:
        if name not in np.__dict__:
            setattr(np, name, value)
    import manopth
    # TACO's editable checkout exposes its inner package, while its reader
    # imports both mano and manopth below the outer directory. Fix this process's
    # package search only; keep the external checkout read-only.
    outer = Path(manopth.__file__).resolve().parents[1]
    if (outer/'mano').is_dir() and str(outer) not in manopth.__path__:
        manopth.__path__.append(str(outer))
    from manopth.manopth.manolayer import ManoLayer
    torch.set_num_threads(4)
    counts, keys_per_side, arrays = [], [], []
    for side in ('right', 'left'):
        poses = load(a.input/(side+'_hand.pkl'))
        shape = load(a.input/(side+'_hand_shape.pkl'))['hand_shape']
        keys = sorted(poses)
        frame_ids = np.array([int(k) for k in keys])
        if not np.array_equal(np.diff(frame_ids), np.ones(len(keys)-1)):
            raise ValueError('noncontiguous TACO frame IDs; do not bridge gaps')
        theta = torch.stack([poses[k]['hand_pose'].cpu().float() for k in keys])
        trans = torch.stack([poses[k]['hand_trans'].cpu().float() for k in keys])
        model = ManoLayer(mano_root=str(a.mano_root), use_pca=False, ncomps=45, side=side,
                          center_idx=0, flat_hand_mean=True)
        with torch.no_grad():
            _, joints, _ = model(theta, shape.cpu().float().reshape(1,10).repeat(len(keys),1))
        # Official manopth order is wrist then five groups of four joints.
        hand = (joints/1000 + trans[:,None])[:,[0,1,5,9,13,17,4,8,12,16,20]].numpy()
        if not np.isfinite(hand).all():
            raise ValueError('nonfinite native TACO FK')
        arrays.append(hand); counts.append(len(keys)); keys_per_side.append(keys)
    if keys_per_side[0] != keys_per_side[1]:
        raise ValueError('right/left native frame identities mismatch')
    obj = np.stack([np.load(a.input/name, allow_pickle=False) for name in ('tool_164.npy','target_057.npy')],1)
    if obj.shape != (counts[0],2,4,4):
        raise ValueError('hand/object native frame count mismatch')
    hand = np.stack(arrays,1)
    sequence = dict(time=np.arange(counts[0])/30, hand=hand, hand_valid=np.ones(hand.shape[:-1],bool),
                    object_poses=obj, object_pose_valid=np.ones(obj.shape[:2],bool),
                    segment=np.zeros(counts[0],np.int64), native_frame_id=np.array([int(k) for k in keys_per_side[0]]))
    validate_sequence(sequence)
    np.savez_compressed(a.output/'taco-20230927_032.npz', **sequence)
    meta = json.loads((a.hot3d_corpus/'manifest.json').read_text())
    for collection in ('objects','sequences'):
        for row in meta[collection]:
            row['path'] = os.path.relpath((a.hot3d_corpus/row['path']).resolve(),a.output.resolve())
    for uid in ('164','057'):
        key = 'TACO/'+uid
        np.savez_compressed(a.output/('taco-'+uid+'.npz'), **geometry(a.input/(uid+'_cm.obj'), .01))
        meta['objects'].append(dict(key=key,path='taco-'+uid+'.npz'))
    meta['sequences'].append(dict(path='taco-20230927_032.npz',format='canonical',objects=['TACO/164','TACO/057'],
        target_slot=0,target_selection='annotated tool; not inferred from future motion',source_id=5,
        profile='human-mano-manopth-native11',split='train',group_id='TACO/(dust, roller, pan)/20230927_032'))
    meta['limitations'] += ['TACO30Hz is an explicit assumption pending native clock verification.',
                            'One TACO sequence has no independent TACO dev; not a validation corpus.',
                            'TACO/HOT3D fingertip mesh indices follow their respective official readers.']
    (a.output/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
    report = dict(status='ENGINEERING_ONLY',frames=counts[0],objects=2,hands=2,units='m',
        source_fps=30,source_fps_status='assumed; official viewer output30fps is not native timestamp proof',
        object_mesh_scale=.01,hand_translation='official centered MANO joints/1000 + hand_trans',
        flat_hand_mean=True,
        cpu_reason='small deterministic geometry FK, no learned model computation',training_allowed=False)
    (a.output/'taco-conversion-audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
