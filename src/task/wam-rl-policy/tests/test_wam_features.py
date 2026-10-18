import json
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
from torch import nn
TASK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK.parent / 'shadow-finetune/src'))
from wam_rl_policy.wam_features import FeatureController


def test_v4_feature_source_builds_512_geometry_and_summary_only_mode(tmp_path, monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'ptv3-shared-wam-v4/src'))
    from ptv3_shared_wam_v4 import model as v4

    class Encoder(nn.Module):
        def __init__(self, stats, config):
            super().__init__()
            assert stats == {'own': 'v4'}
            self.weight = nn.Parameter(torch.zeros(2))

        def encode(self, batch):
            raise AssertionError('not called during construction')

    monkeypatch.setattr(v4, 'SharedWAMV4', Encoder)
    monkeypatch.setattr(nn.Module, 'cuda', lambda self: self)
    state = {'weight': torch.tensor([2., 3.])}
    checkpoint = tmp_path / 'v4.pt'
    torch.save(dict(schema='ref2dex.ptv3-shared-wam.v4', stats={'own': 'v4'}, config={},
                    model=state, step=12), checkpoint)
    surface = np.stack((np.linspace(-.1, .1, 512), np.zeros(512),
                        np.zeros(512)), -1).astype(np.float32)
    normals = np.tile([0., 1., 0.], (512, 1)).astype(np.float32)
    query = surface[np.linspace(0, 511, 32, dtype=np.int64)]
    geometry = tmp_path / 'geometry.npz'
    np.savez(geometry, surface=surface, normals=normals, query=query)
    (tmp_path / 'manifest.json').write_text(json.dumps({'records': [
        {'object_code': 'one', 'scale': .08, 'geometry': str(geometry)}]}))
    controller = FeatureController(checkpoint, [{'object_code': 'one', 'scale': .08}], tmp_path)
    assert controller.mode == 'v4-summary-encode-only'
    assert len(controller.geometry) == 1
    assert controller.geometry[0][0].shape == (512, 3)
    assert controller.geometry[0][2].shape == (32, 3)
    assert controller.geometry[0][3].shape == (128, 3)


@pytest.mark.parametrize('schema',['ref2dex.ptv3-shared-wam.v2','ref2dex.ptv3-shared-wam.v3','ref2dex.shadow-native-policy.v1'])
def test_feature_source_is_separate_strict_frozen_and_uses_own_stats(tmp_path,monkeypatch,schema):
    class Encoder(nn.Module):
        def __init__(self,stats,config):
            super().__init__();assert stats=={'own':schema};self.weight=nn.Parameter(torch.zeros(2))
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'ptv3-shared-wam-v3/src'))
    from ptv3_shared_wam_v2 import model as v2
    from ptv3_shared_wam_v3 import model as v3
    monkeypatch.setattr(v2,'SharedWAMV2',Encoder);monkeypatch.setattr(v3,'SharedWAMV3',Encoder)
    monkeypatch.setattr(nn.Module,'cuda',lambda self:self)
    state={'weight':torch.tensor([2.,3.])}
    if schema=='ref2dex.shadow-native-policy.v1':state={'forecaster.weight':state['weight'],'actor.unused':torch.ones(1)}
    checkpoint=tmp_path/'source.pt';torch.save(dict(schema=schema,stats={'own':schema},config={},model=state,step=80000),checkpoint)
    geometry=tmp_path/'geometry.npz';np.savez(geometry,surface=np.zeros((2,3)),normals=np.ones((2,3)))
    (tmp_path/'manifest.json').write_text(json.dumps({'records':[{'object_code':'one','geometry':str(geometry)}]}))
    controller=FeatureController(checkpoint,[{'object_code':'one'}],tmp_path)
    assert controller.features_only and controller.expose_wam_features and not controller.model.training
    assert torch.equal(controller.model.forecaster.weight,torch.tensor([2.,3.]))
    assert not controller.model.forecaster.weight.requires_grad and controller.source_step==80000
    assert not hasattr(controller.model,'actor')
