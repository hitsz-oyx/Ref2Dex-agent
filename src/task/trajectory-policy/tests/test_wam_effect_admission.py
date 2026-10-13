import json

import pytest
import torch

from trajectory_policy import wam_effect_admission as admission


def test_effect_patch_rejects_changed_interface_and_frozen_state(tmp_path, monkeypatch):
    monkeypatch.setattr(admission, 'ROOT', tmp_path)
    directory = tmp_path/'outputs/goal-point-wam/run/train'
    directory.mkdir(parents=True)
    warm = tmp_path/'outputs/trajectory-policy/warm'
    chunk = tmp_path/'outputs/trajectory-policy/chunk'
    path = directory/'checkpoint_step50500.pt'
    packet = dict(schema='ref2dex.effect-policy-patch.v1', step=50500, C0_pass=True,
                  manifest=dict(warm_policy=str(warm), qualified_chunk=str(chunk/'chunk.pt'),
                                base_checkpoint=str(tmp_path/'base.pt')),
                  recipe=dict(train_modules=['trajectory']), trajectory={'weight': torch.ones(2)},
                  curve=[dict(model_frozen_max_difference=0., R_max_difference=0.)])
    (directory/'result.json').write_text(json.dumps(dict(status='COMPLETED', step=50500)))
    torch.save(packet, path)
    identity, values = admission.effect_admission(path, warm, chunk)
    assert identity['global_step'] == 50500 and torch.equal(values['weight'], torch.ones(2))
    with pytest.raises(ValueError, match='identity differs'):
        admission.effect_admission(path, tmp_path/'another_warm', chunk)
    with pytest.raises(ValueError, match='identity differs'):
        admission.effect_admission(path, warm, tmp_path/'another_chunk')
    packet['curve'][0]['R_max_difference'] = .01
    torch.save(packet, path)
    with pytest.raises(ValueError, match='identity differs'):
        admission.effect_admission(path, warm, chunk)
