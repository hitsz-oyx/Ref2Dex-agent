import importlib.util
from pathlib import Path

import pytest
import torch

path = Path(__file__).resolve().parents[1] / 'tools/run/prepare_pointworld_backbone_initialization.py'
spec = importlib.util.spec_from_file_location('backbone_initialization', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_only_spatial_module_changes_and_nonfinite_or_partial_import_aborts():
    model = torch.nn.Module()
    model.backbone = torch.nn.Module()
    model.backbone.core = torch.nn.Linear(3, 2)
    model.head = torch.nn.Linear(2, 1)
    model.register_buffer('scene_mean', torch.arange(18.).float())
    outside = module.state_hash(model.state_dict(), exclude_backbone=True)
    weights = {module.PREFIX+k: torch.full_like(v, .25)
               for k, v in model.backbone.core.state_dict().items()}
    weights['scene_feature_encoder.dino.fake'] = torch.ones(5)
    module.import_backbone(model.backbone.core, weights)
    assert module.state_hash(model.state_dict(), exclude_backbone=True) == outside
    assert torch.equal(model.backbone.core.weight, torch.full((2, 3), .25))
    before = module.state_hash(model.state_dict())
    partial = dict(weights)
    del partial[module.PREFIX+'bias']
    with pytest.raises(ValueError, match='keys differ'):
        module.import_backbone(model.backbone.core, partial)
    bad = dict(weights, **{module.PREFIX+'bias': torch.full((2,), float('nan'))})
    with pytest.raises(ValueError, match='nonfinite'):
        module.import_backbone(model.backbone.core, bad)
    assert module.state_hash(model.state_dict()) == before


def test_shape_or_extra_backbone_keys_fail_before_mutation():
    core = torch.nn.Linear(3, 2)
    source = {module.PREFIX+k: v.clone() for k, v in core.state_dict().items()}
    before = module.state_hash(core.state_dict())
    bad = dict(source, **{module.PREFIX+'weight': torch.zeros(3, 3)})
    with pytest.raises(ValueError, match='shape/dtype'):
        module.import_backbone(core, bad)
    bad = dict(source, **{module.PREFIX+'unrecognized': torch.zeros(1)})
    with pytest.raises(ValueError, match='keys differ'):
        module.import_backbone(core, bad)
    assert module.state_hash(core.state_dict()) == before
