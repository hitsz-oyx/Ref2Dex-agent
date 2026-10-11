"""Information boundaries and masked joint-supervision semantics."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from oakink_wm.pointwam_data import pack_clip, sha256
from oakink_wm.pointwam_learning import (JointForecaster, batch, current_inputs,
    fit_contract, joint_loss, load_qualified)


def clips():
    rng = np.random.default_rng(4)
    scene = rng.normal(size=(11, 8, 3)).astype(np.float32) * .1
    hand = rng.normal(size=(11, 2, 10, 3)).astype(np.float32) * .1
    hand[:, 1] = np.nan
    valid = np.ones((11, 2, 10), bool); valid[:, 1] = False
    pack = pack_clip(scene, np.ones((11, 8), bool), hand, valid,
                     np.zeros((8, 3), np.uint8), np.arange(11)*3, "fold paper")
    return [dict(pack=pack, partition="fit_candidate", sha256="fit"),
            dict(pack=copy.deepcopy(pack), partition="dev_candidate", sha256="dev")]


def test_future_perturbation_cannot_change_model_inputs():
    rows = clips(); contract = fit_contract(rows)
    pack = rows[0]["pack"]; ids = np.arange(8)
    original = current_inputs(pack, ids, contract)
    changed = copy.deepcopy(pack)
    for kind in ("scene", "hand"):
        changed[kind + "_xyz_m"][1:] = 999
        changed[kind + "_valid"][1:] = False
        changed[kind + "_displacement_m"][:] = -999
        changed[kind + "_target_valid"][:] = False
    for key, value in original.items():
        np.testing.assert_array_equal(value, current_inputs(changed, ids, contract)[key])


def test_dev_targets_and_text_cannot_change_fit_contract():
    rows = clips(); contract = fit_contract(rows)
    rows[1]["pack"]["language"] = np.asarray("unseen invented vocabulary")
    for kind in ("scene", "hand"):
        rows[1]["pack"][kind + "_displacement_m"][:] = 1e9
    assert fit_contract(rows) == contract
    assert "unseen" not in contract["vocabulary"]


def test_missing_hand_gradient_is_zero_and_elements_are_equally_weighted():
    # First example has one valid point, second has two: equal example weights.
    scene = torch.zeros((2, 1, 2, 3), requires_grad=True)
    hand = torch.tensor([[[[1., 1., 1.], [999., 999., 999.]]],
                         [[[2., 2., 2.], [2., 2., 2.]]]], requires_grad=True)
    target = dict(scene_target=torch.zeros_like(scene), scene_weight=torch.ones((2, 1, 2)),
                  hand_target=torch.zeros_like(hand), hand_mask=torch.tensor([[[True, False]], [[True, True]]]))
    loss, parts = joint_loss(dict(scene=scene, hand=hand), target)
    assert parts["hand"].item() == 2.5
    loss.backward()
    assert torch.equal(hand.grad[0, 0, 1], torch.zeros(3))


def test_model_scene_permutation_and_invalid_hand_padding_invariance():
    rows = clips(); contract = fit_contract(rows)
    x, _ = batch(rows[:1], [np.arange(8)], contract, "cpu")
    torch.manual_seed(9)
    model = JointForecaster(len(contract["vocabulary"]) + 1, hidden=16)
    # Enable a nonconstant output, otherwise zero initialization makes this vacuous.
    for head in (model.scene_head, model.hand_head):
        torch.nn.init.normal_(head[-1].weight, std=.01)
    expected = model(**x)
    swapped = {key: value.clone() for key, value in x.items()}
    order = torch.arange(7, -1, -1)
    for key in ("scene_xyz", "scene_rgb", "scene_valid"):
        swapped[key] = swapped[key][:, order]
    swapped["hand_xyz"][:, 10:] = 100
    actual = model(**swapped)
    torch.testing.assert_close(actual["scene"], expected["scene"][:, :, order])
    torch.testing.assert_close(actual["hand"][:, :, :10], expected["hand"][:, :, :10])
    assert torch.isfinite(actual["hand"]).all()


def test_admission_rejects_test_pack_even_with_valid_hash(tmp_path):
    pack = clips()[0]["pack"]
    pack.update(source_split=np.asarray("official_test"), source_episode_id=np.asarray("test-record"))
    path = tmp_path / "pack.npz"; np.savez(path, **pack)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(dict(status="COMPLETED", gate_pass=True,
        clips=[dict(qualified=True, pack=str(path), pack_sha256=sha256(path),
                    episode_id="test-record", partition="fit_candidate")])))
    with pytest.raises(ValueError, match="TEST/unknown"):
        load_qualified(manifest)
