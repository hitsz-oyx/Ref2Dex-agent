"""Physical native category/clock adaptation, source sampling and pretrained weights."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

TASK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK/'src'))
from oakink_wm.multisource import MixedWindows, mixed_indices, sha, validate_pretrained


def make_source(root, name):
    processed = root/'processed'; processed.mkdir(parents=True)
    (processed/'canonical').mkdir(); (processed/'sequences').mkdir()
    points = np.zeros((512, 3), dtype='float32'); points[:, 0] = np.linspace(-.02, .02, 512)
    np.savez(processed/'canonical/cloud.npz', points=points, normals=np.tile([0, 0, 1], (512, 1)),
             radius=.02, center=np.zeros(3))
    sequences, records = [], []
    for split in ('train', 'val', 'test'):
        rows = []
        categories = (0, 1, 2) if name == 'oakink2' else (1, 2)
        for category in categories:
            moving = category == (0 if name == 'oakink2' else 1)
            seq = name+'_'+split+'_'+str(category); sid = len(sequences); sequences.append(seq)
            records.append(dict(source=name, sequence=seq, split=split))
            folder = processed/'sequences'/seq; folder.mkdir()
            poses = np.tile(np.eye(4), (32, 1, 1, 1))
            if moving: poses[:, 0, 0, 3] = np.arange(32)*.001
            hand = np.zeros((32, 2, 11, 3)); hand[..., 0] = poses[:, 0, 0, 3, None, None]+.01
            arrays = dict(hand=hand.astype('float32'), hand_valid=np.ones((32, 2), bool),
                          poses=poses.astype('float32'), pose_valid=np.ones((32, 1), bool),
                          program=np.zeros((32, 1), bool), near=np.ones((32, 1), bool),
                          frame_ids=4*np.arange(32) if name == 'oakink2' else 100+np.arange(32),
                          timestamps=np.arange(32)/30, centers=np.zeros((1, 3)))
            for key, value in arrays.items(): np.save(folder/(key+'.npy'), value)
            (folder/'meta.json').write_text(json.dumps(dict(objects=['cloud'])))
            rows.append([sid, 0, 3, category])
        np.save(processed/('index_'+split+'.npy'), np.array(rows, dtype=np.int64))
    metadata = dict(schema='oakink.wm30.geometry-centers.v2' if name == 'oakink2' else 'ref2dex.native-wm30.v1',
                    status='COMPLETED', fps=30, history=4, horizon=24, units='m', training_allowed=True,
                    hand_order=['right', 'left'], sequences=sequences, records=records,
                    split_protocol='GRAB author object holdouts; ARCTIC protocol_p1')
    (processed/'manifest.json').write_text(json.dumps(metadata))
    return root


def test_actual_tensor_clock_category_identity_and_source_balanced_draws(tmp_path):
    spec = importlib.util.spec_from_file_location('mixed_prepare', TASK/'tools/run/prepare_mixed_manifest.py')
    prepare = importlib.util.module_from_spec(spec); spec.loader.exec_module(prepare)
    prepare.ROOT = tmp_path
    roots = [make_source(tmp_path/name, name) for name in ('oakink2', 'grab', 'arctic', 'contactpose')]
    stats = tmp_path/'stats.json'
    stats.write_text(json.dumps(dict(split='train', input_manifest_sha256=sha(roots[0]/'processed/manifest.json'))))
    output = tmp_path/'outputs/cm-pointflow-effect-pretrain/mixed'
    prepare.prepare(roots, output, stats)
    data = MixedWindows(output, 'train')
    identifiers = []
    for source_id, source in enumerate(data.sources):
        sample = data[int(data.offsets[source_id]+source.groups[0][0])]
        assert sample['category'] == 0 and sample['action'].shape == (24, 22, 9)
        np.testing.assert_allclose(sample['effect'][0, -1, :3, 3], [.024, 0, 0], atol=1e-7)
        identifiers.append(tuple(sample['sample_id']))
        static = data[int(data.offsets[source_id]+source.groups[1][0])]
        np.testing.assert_allclose(static['effect'][0, :, :3, 3], 0, atol=1e-7)
        if source_id:
            assert len(source.groups[2]) == 0
            # Native physical frame identities stay unchanged on disk.
            np.testing.assert_array_equal(np.load(source.root/'processed/sequences'/source.sequences[0]/'frame_ids.npy'),
                                          100+np.arange(32))
    assert len(set(identifiers)) == 4
    draws = mixed_indices(data, 10000, 226)
    source_ids = np.searchsorted(data.offsets, draws, side='right')-1
    np.testing.assert_allclose(np.bincount(source_ids)/len(draws), [.5, .2, .2, .1], atol=.02)
    val = MixedWindows(output, 'val')
    panel = mixed_indices(val, 256, 212, equal_sources=True)
    np.testing.assert_array_equal(np.bincount(np.searchsorted(val.offsets, panel, side='right')-1), [64]*4)
    np.testing.assert_array_equal(draws, mixed_indices(data, 10000, 226))
    # Runtime index mutation must be rejected rather than silently fitting another split.
    index = Path(data.meta['sources'][1]['indices']['train']['path'])
    index.write_bytes(index.read_bytes()+b'drift')
    with pytest.raises(ValueError, match='index drift'): MixedWindows(output, 'train')


def test_mixed_model_only_initialization_rejects_scale_and_model_drift():
    source = torch.nn.Linear(2, 1); target = torch.nn.Linear(2, 1)
    identity = dict(dataset_hash='mixed', normalization_source_manifest_sha256='parent-data',
                    stats_sha256='stats', arm='action', vendor_sources={'vendor': 'same'})
    previous = dict(identity, implementation_sources={'model.py': 'same'})
    state = dict(step=10000, dataset_hash='parent-data', identity=previous,
                 config=dict(horizon=24, microbatch=64), model=source.state_dict())
    result = validate_pretrained(state, target, dict(horizon=24, microbatch=64), identity, {'model.py': 'same'})
    assert result['optimizer_reset'] and result['schedule_reset'] and result['weights_only']
    for key, value in source.state_dict().items(): torch.testing.assert_close(value, target.state_dict()[key], rtol=0, atol=0)
    with pytest.raises(ValueError, match='provenance mismatch'):
        validate_pretrained(state, target, state['config'], dict(identity, stats_sha256='new'), {'model.py': 'same'})
    with pytest.raises(ValueError, match='provenance mismatch'):
        validate_pretrained(state, target, state['config'], identity, {'model.py': 'changed'})
    with pytest.raises(ValueError, match='semantics'):
        validate_pretrained(state, target, dict(state['config'], horizon=8), identity, {'model.py': 'same'})
