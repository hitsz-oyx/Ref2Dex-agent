import numpy as np
import torch

from pointwam_repro.contract import ARKIT_TO_ZUP, audit_pack
from pointwam_repro.forecaster import ReducedPatchForecaster
from pointwam_repro.loss import motion_weights, trajectory_loss
from pointwam_repro.tokenize import pack_patches, patch_slots


def _pack(points: int = 256) -> dict:
    scene = np.zeros((31, points, 3), np.float32)
    scene[:, :, 0] = np.linspace(0, 0.2, 31)[:, None]
    hand = np.zeros((31, 2, 10, 3), np.float32)
    hand[:, 0, :, 0] = np.linspace(0, 0.08, 31)[:, None]
    valid_scene = np.ones(scene.shape[:-1], bool)
    valid_hand = np.ones(hand.shape[:-1], bool)
    frames = np.arange(0, 33, 3)
    return dict(
        scene_xyz_m=scene,
        scene_valid=valid_scene,
        scene_rgb=np.zeros((points, 3), np.uint8),
        hand_xyz_m=hand,
        hand_valid=valid_hand,
        scene_displacement_m=scene[1:] - scene[0],
        hand_displacement_m=hand[1:] - hand[0],
        scene_target_valid=np.ones((30, points), bool),
        hand_target_valid=np.ones((30, 2, 10), bool),
        source_frame_ids=frames,
        timestamps_s=np.linspace(frames[0] / 30, frames[-1] / 30, 31),
        arkit_to_zup=ARKIT_TO_ZUP.copy(),
        training_allowed=np.asarray(False),
        schema=np.asarray("ref2dex.pointwam-data-smoke.v1"),
    )


def test_scene_sample_ignores_future_and_keeps_current_targets() -> None:
    from pointwam_repro.packs import example_from_pack, select_scene_ids

    valid = np.zeros(6, dtype=bool)
    valid[[0, 2, 5]] = True
    assert list(select_scene_ids(valid, 2)) == [0, 5]
    pack = _pack(points=256)
    pack["scene_valid"][0, 10:] = False
    example = example_from_pack(pack, 4)
    assert example["scene_xyz"].shape == (31, 4, 3)
    gap = np.abs((example["scene_xyz"][1:] - example["scene_xyz"][0]) - example["scene_displacement"])
    assert float(gap.max()) == 0.0


def test_contract_accepts_relative_displacement() -> None:
    report = audit_pack(_pack())
    assert report["passed"], report["errors"]
    assert report["scene_points"] == 256


def test_contract_rejects_enabled_training_flag() -> None:
    pack = _pack()
    pack["training_allowed"] = np.asarray(True)
    report = audit_pack(pack)
    assert not report["passed"]
    assert "training_allowed" in report["errors"]


def test_patch_grid_groups_one_centimeter_and_keeps_negative_slots() -> None:
    xyz = torch.tensor([[0.0, 0.0, 0.0], [0.004, 0.0, 0.0], [0.06, 0.0, 0.0], [-0.001, 0.0, 0.0]])
    patch, slot = patch_slots(xyz)
    assert torch.equal(patch[0], patch[1])
    assert not torch.equal(patch[0], patch[2])
    assert int(slot[3]) == 4
    features = torch.tensor([[0.0], [2.0], [5.0], [7.0]])
    packed = pack_patches(features, xyz)
    same = packed["tokens"][int(packed["point_patch"][0]), int(slot[0])]
    assert torch.allclose(same, torch.tensor([1.0]))


def test_motion_weight_prefers_moving_points_and_sums_to_one() -> None:
    positions = torch.zeros(1, 2, 2, 3)
    positions[0, 1, 1, 0] = 0.02
    valid = torch.ones(1, 1, 2, dtype=torch.bool)
    weights = motion_weights(positions, valid)
    assert torch.allclose(weights.sum(), torch.tensor(1.0))
    assert weights[0, 0, 1] > weights[0, 0, 0]


def test_invalid_nan_targets_stay_out_of_the_loss() -> None:
    prediction = torch.zeros(1, 1, 2, 3)
    target = torch.zeros(1, 1, 2, 3)
    target[0, 0, 0, 0] = 3
    target[0, 0, 1] = float("nan")
    valid = torch.tensor([[[True, False]]])
    positions = torch.zeros(1, 2, 2, 3)
    positions[0, 1, 1] = float("nan")
    loss = trajectory_loss(prediction, target, valid, positions)
    assert torch.isfinite(loss)
    assert torch.allclose(loss, torch.tensor(3.0))


def test_trajectory_loss_is_per_coordinate() -> None:
    prediction = torch.zeros(1, 1, 1, 3)
    target = torch.zeros(1, 1, 1, 3)
    target[0, 0, 0, 0] = 3
    valid = torch.ones(1, 1, 1, dtype=torch.bool)
    loss = trajectory_loss(prediction, target, valid)
    assert torch.allclose(loss, torch.tensor(3.0))


def test_zero_output_head_matches_static_prediction() -> None:
    model = ReducedPatchForecaster()
    output = model.forward_clip(
        torch.zeros(4, 3), torch.zeros(4, 3), torch.ones(4, dtype=torch.bool),
        torch.zeros(2, 10, 3), torch.ones(2, 10, dtype=torch.bool))
    assert torch.count_nonzero(output["scene_displacement"]) == 0
    assert torch.count_nonzero(output["hand_displacement"]) == 0


def test_reduced_forecaster_matches_horizon_and_backpropagates() -> None:
    torch.manual_seed(0)
    model = ReducedPatchForecaster()
    scene = torch.randn(8, 3)
    rgb = torch.randint(0, 255, (8, 3))
    hand = torch.randn(2, 10, 3)
    output = model.forward_clip(
        scene, rgb, torch.ones(8, dtype=torch.bool), hand, torch.ones(2, 10, dtype=torch.bool))
    assert output["scene_displacement"].shape == (30, 8, 3)
    assert output["hand_displacement"].shape == (30, 2, 10, 3)
    output["scene_displacement"].sum().backward()
    assert model.scene_feature[0].weight.grad is not None
    assert torch.isfinite(model.scene_feature[0].weight.grad).all()
