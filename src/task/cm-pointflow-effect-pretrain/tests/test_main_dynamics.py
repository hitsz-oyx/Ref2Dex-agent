"""Three-source supervision, strict subset warm starts and separate loss scales."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

TASK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK/'src'))
from oakink_wm.loss_normalization import physical_loss, validate_loss_statistics
from oakink_wm.model import geodesic, rigid_points
from oakink_wm.multisource import MixedWindows, MAIN_SOURCE_NAMES, mixed_indices, sha, validate_pretrained
from test_multisource import make_source


def prepare_corpora(tmp_path):
    spec = importlib.util.spec_from_file_location('main_prepare', TASK/'tools/run/prepare_mixed_manifest.py')
    prepare = importlib.util.module_from_spec(spec); spec.loader.exec_module(prepare)
    prepare.ROOT = tmp_path
    roots = [make_source(tmp_path/name, name) for name in ('oakink2', 'grab', 'arctic', 'contactpose')]
    stats = tmp_path/'stats.json'
    stats.write_text(json.dumps(dict(split='train', input_manifest_sha256=sha(roots[0]/'processed/manifest.json'))))
    mixed = tmp_path/'outputs/cm-pointflow-effect-pretrain/four'
    main = tmp_path/'outputs/cm-pointflow-effect-pretrain/three'
    old = prepare.prepare(roots, mixed, stats)
    new = prepare.prepare(roots[:3], main, stats, main_dynamics=True)
    return old, new, main


def test_main_excludes_transport_and_preserves_frozen_source_windows(tmp_path):
    old, new, main = prepare_corpora(tmp_path)
    assert tuple(d['name'] for d in new['sources']) == MAIN_SOURCE_NAMES
    assert new['excluded_auxiliary_sources'] == ['contactpose']
    for before, after in zip(old['sources'], new['sources']):
        assert before['manifest_sha256'] == after['manifest_sha256']
        assert all(before['indices'][s]['sha256'] == after['indices'][s]['sha256'] for s in ('train', 'val', 'test'))
    data = MixedWindows(main, 'train')
    draws = mixed_indices(data, 30000, 226)
    source_ids = np.searchsorted(data.offsets, draws, side='right')-1
    np.testing.assert_allclose(np.bincount(source_ids)/len(draws), [5/9, 2/9, 2/9], atol=.015)
    ids = mixed_indices(MixedWindows(main, 'val'), 192, 212, equal_sources=True)
    np.testing.assert_array_equal(np.bincount(np.searchsorted(data.offsets, ids, side='right')-1), [64]*3)
    with pytest.raises(ValueError, match='equal source panels'):
        mixed_indices(data, 256, 212, equal_sources=True)


def test_four_to_three_warm_start_only_allows_unchanged_source_subset(tmp_path):
    old, new, main = prepare_corpora(tmp_path)
    source, target = torch.nn.Linear(2, 1), torch.nn.Linear(2, 1)
    identity = dict(dataset_hash=sha(main/'processed/manifest.json'), source_manifest=new,
                    normalization_source_manifest_sha256=new['normalization_source_manifest_sha256'],
                    stats_sha256='stats', arm='action', vendor_sources={'vendor': 'same'})
    previous = dict(identity, mixed_data=True, dataset_hash='oldmixed', source_manifest=old,
                    implementation_sources={'model.py': 'same'})
    state = dict(step=14250, dataset_hash='oldmixed', identity=previous,
                 config=dict(horizon=24, validation_samples=256), model=source.state_dict())
    result = validate_pretrained(state, target, dict(horizon=24, validation_samples=192), identity, {'model.py': 'same'})
    assert result['parent_step'] == 14250 and result['optimizer_reset'] and result['schedule_reset']
    for key, value in source.state_dict().items(): torch.testing.assert_close(value, target.state_dict()[key], rtol=0, atol=0)
    for kind in ('manifest', 'indices', 'model'):
        changed = copy.deepcopy(identity)
        expected = {'model.py': 'same'}
        if kind == 'manifest': changed['source_manifest']['sources'][1]['manifest_sha256'] = 'other'
        elif kind == 'indices': changed['source_manifest']['sources'][1]['indices']['train']['sha256'] = 'other'
        else: expected['model.py'] = 'changed'
        with pytest.raises(ValueError, match='provenance mismatch'):
            validate_pretrained(state, target, state['config'], changed, expected)


def test_loss_only_stats_reject_foreign_corpus_validation_and_transport_sources():
    stats = dict(schema='pointworld-wm24.loss-normalization.v1', role='loss_only', split='train',
                 input_manifest_sha256='main', sources=list(MAIN_SOURCE_NAMES),
                 flow_std=np.full((24,3), .02).tolist(), translation_std=np.full((24,3), .01).tolist(),
                 rotation_scale=np.full(24,.1).tolist())
    assert set(validate_loss_statistics(stats, 'main', MAIN_SOURCE_NAMES)) == {'flow_std','translation_std','rotation_scale'}
    for change in (dict(split='val'),dict(input_manifest_sha256='old'),dict(role='forward'),
                   dict(sources=list(MAIN_SOURCE_NAMES)+['contactpose']),dict(flow_std=np.zeros((24,3)).tolist())):
        with pytest.raises(ValueError): validate_loss_statistics(dict(stats, **change), 'main', MAIN_SOURCE_NAMES)


def test_separate_scales_preserve_model_buffers_and_match_original_huber():
    tree = ast.parse((TASK/'src/oakink_wm/pointworld_temporal.py').read_text())
    method = next(n for c in tree.body if isinstance(c, ast.ClassDef) and c.name == 'TemporalPointWorldWM'
                  for n in c.body if isinstance(n, ast.FunctionDef) and n.name == 'loss')
    module = ast.Module(body=[method], type_ignores=[])
    namespace = dict(torch=torch, F=F, rigid_points=rigid_points, geodesic=geodesic)
    exec(compile(ast.fix_missing_locations(module), '<unchanged temporal loss>', 'exec'), namespace)
    class Model:
        motion_weighting = 'cumulative_effect'
        flow_mean = torch.rand(24,3)
        flow_std = torch.full((24,3), .02)
        translation_mean = torch.rand(24,3)
        translation_std = torch.full((24,3), .01)
        rotation_scale = torch.full((24,), .1)
        def motion_weights(self, batch, actual):
            return torch.ones_like(actual[...,0])*batch['object_valid'][:,:,None,None]
    model = Model()
    frozen = {k:getattr(model,k).clone() for k in ('flow_std','flow_mean','translation_mean','translation_std','rotation_scale')}
    gt = torch.eye(4).repeat(1,2,24,1,1); gt[:,:,:,:3,3] = .003
    batch = dict(effect=gt, points=torch.rand(1,2,4,3), object_valid=torch.tensor([[True,False]]))
    translation = torch.full((1,2,24,3), .009, requires_grad=True)
    pred = dict(rotation=torch.eye(3).repeat(1,2,24,1,1), translation=translation)
    scales = {k:getattr(model,k) for k in ('flow_std','translation_std','rotation_scale')}
    actual, _ = physical_loss(model,pred,batch,scales)
    expected, _ = namespace['loss'](model,pred,batch)
    torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
    actual.backward()
    assert torch.count_nonzero(translation.grad[:,0]) > 0 and torch.count_nonzero(translation.grad[:,1]) == 0
    larger, _ = physical_loss(model,pred,batch,{k:v*2 for k,v in scales.items()})
    assert larger < actual
    for key,value in frozen.items(): torch.testing.assert_close(getattr(model,key),value,rtol=0,atol=0)


def test_loss_stats_drift_and_resume_mismatch_are_rejected(tmp_path):
    spec = importlib.util.spec_from_file_location('main_ddp_trainer', TASK/'tools/run/train_oakink2_pointworld_ddp.py')
    trainer = importlib.util.module_from_spec(spec); spec.loader.exec_module(trainer)
    config_path, forward_stats, loss_stats = [tmp_path/name for name in ('config.json','forward.json','loss.json')]
    for path in (config_path,forward_stats,loss_stats): path.write_text('{}')
    identity = dict(dataset_hash='data', arm='action', stats_sha256=sha(forward_stats),
                    input_config_sha256=sha(config_path), loss_stats_path=str(loss_stats),
                    loss_stats_sha256=sha(loss_stats), implementation_sources={},vendor_sources={})
    assert not trainer.sources_drifted(identity,config_path,forward_stats)
    loss_stats.write_text('{"changed":true}')
    assert trainer.sources_drifted(identity,config_path,forward_stats)
    config = dict(seed=225)
    state = dict(config=config,dataset_hash='data',identity=dict(identity,loss_stats_sha256='foreign'),
                 checkpoint_kind='pointworld-temporal.ddp.v1',world_size=2)
    model = torch.nn.Linear(2,1); optimizer = torch.optim.AdamW(model.parameters())
    with pytest.raises(ValueError,match='implementation mismatch'):
        trainer.load_checkpoint(state,model,optimizer,config,identity,0,2)


def test_actual_statistics_entry_reads_only_main_train_windows(tmp_path, monkeypatch):
    _, new, main = prepare_corpora(tmp_path)
    # Corrupt validation motion magnitudes: they must never affect fitted scales.
    for source in new['sources']:
        root = Path(source['root'])
        meta = json.loads((root/'processed/manifest.json').read_text())
        for record in meta['records']:
            if record['split'] != 'val': continue
            path = root/'processed/sequences'/record['sequence']/'poses.npy'
            poses = np.load(path); poses[...,0,3] = np.arange(len(poses))[:,None]*100
            np.save(path,poses)
    spec = importlib.util.spec_from_file_location('main_stats_entry', TASK/'tools/run/prepare_main_loss_stats.py')
    entry = importlib.util.module_from_spec(spec); spec.loader.exec_module(entry)
    entry.ROOT = tmp_path
    output = main/'loss_stats.json'
    monkeypatch.setattr(sys,'argv',['stats','--data',str(main),'--output',str(output),
                                   '--samples','24','--seconds','60','--device','cpu'])
    entry.main()
    fitted = json.loads(output.read_text())
    assert fitted['split'] == 'train' and fitted['sources'] == list(MAIN_SOURCE_NAMES)
    assert sum(fitted['source_window_counts']) == 24
    assert fitted['input_manifest_sha256'] == sha(main/'processed/manifest.json')
    assert np.abs(fitted['translation_mean']).max() <= .02401
    validate_loss_statistics(fitted,sha(main/'processed/manifest.json'),MAIN_SOURCE_NAMES)
    with pytest.raises(FileExistsError): entry.main()
