import json
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
from torch import nn
from wam_rl_policy.wam_features import FeatureController


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
