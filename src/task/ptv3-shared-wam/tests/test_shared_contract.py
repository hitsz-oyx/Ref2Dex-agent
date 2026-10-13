"""Semantic tests: leakage, separate hand identity and actual gradient paths."""
import copy

import numpy as np
import pytest
import torch
from torch import nn

from ptv3_shared_wam.data import endpoint_delta, point_labels
from ptv3_shared_wam.contracts import GOAL_ENCODING
from ptv3_shared_wam.execution import ActionBuffer
from ptv3_shared_wam.model import SharedWAM
from ptv3_shared_wam.objectives import forecast_loss
from ptv3_shared_wam.robot import RobotPair, native_trajectory, robot_loss
from ptv3_shared_wam.checkpoint import load_checkpoint, save_checkpoint

torch.set_num_threads(1)


class DenseScene(nn.Module):
    """CPU interface fixture; actual sparse PTv3 is checked by the GPU smoke."""
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(3, 128)

    def forward(self, batch):
        xyz = batch["xyz"]
        feature = self.backbone(xyz)
        points = batch["scene_points"]
        return dict(tokens=feature, xyz=xyz, mask=batch["point_valid"],
                    point_features=self.backbone(points))


def stats():
    return dict(scene_mean=[0.] * 18, scene_std=[1.] * 18)


def batch():
    torch.manual_seed(19)
    b = dict(xyz=torch.randn(2, 8, 3) * .05, features=torch.zeros(2, 8, 18),
             point_valid=torch.ones(2, 8, dtype=torch.bool), scene_object=torch.zeros(2, 8, dtype=torch.long),
             object_valid=torch.ones(2, 2, dtype=torch.bool), scene_points=torch.randn(2, 2, 4, 3) * .05,
             start_hand=torch.randn(2, 2, 11, 3) * .05, start_hand_history=torch.randn(2, 2, 11, 9),
             start_hand_valid=torch.ones(2, 2, 11, dtype=torch.bool),
             start_object_points=torch.randn(2, 4, 3) * .05, goal_object_points=torch.randn(2, 4, 3) * .05,
             object_goal_valid=torch.ones(2, 4, dtype=torch.bool),
             trajectory_target=torch.randn(2, 24, 2, 11, 3) * .02,
             trajectory_valid=torch.ones(2, 24, 2, 11, dtype=torch.bool),
             scene_target=torch.randn(2, 2, 24, 4, 3) * .02,
             scene_valid=torch.ones(2, 2, 24, 4, dtype=torch.bool))
    b["goal_object_delta"] = endpoint_delta(b["start_object_points"], b["goal_object_points"], b["object_goal_valid"])
    return b


def model():
    return SharedWAM(stats(), dict(interaction_layers=2), DenseScene()).eval()


def test_goal_changes_hand_but_cannot_change_scene_or_shared_representation():
    net, b = model(), batch()
    before = net(b)
    z = net.encode(b)[0]
    altered = dict(b, goal_object_delta=b["goal_object_delta"] + .2)
    after = net(altered)
    assert torch.equal(before["scene"], after["scene"])
    assert torch.equal(z, net.encode(altered)[0])
    assert not torch.equal(before["trajectory"], after["trajectory"])
    goal = b["goal_object_delta"].clone().requires_grad_()
    prediction = net(dict(b, goal_object_delta=goal))
    assert torch.autograd.grad(prediction["scene"].sum(), goal, allow_unused=True)[0] is None
    assert torch.autograd.grad(prediction["trajectory"].sum(), goal)[0].abs().sum() > 0


def test_future_labels_and_native_state_do_not_enter_forecasting():
    net, b = model(), batch()
    before = net(b)
    changed = dict(b, scene_target=torch.full_like(b["scene_target"], 100.),
                   trajectory_target=torch.full_like(b["trajectory_target"], -100.),
                   native_state=torch.randn(2, 87), native_tau=torch.randn(2, 24, 33),
                   goal_object_points=torch.full_like(b["goal_object_points"], 100.),
                   action=torch.randn(2, 24, 22, 9), effect=torch.randn(2, 2, 24, 4, 4))
    after = net(changed)
    assert all(torch.equal(before[k], after[k]) for k in before)


def test_both_losses_reach_shared_scene_and_hand_encoders_without_V_to_U():
    net, b = model(), batch()
    p = net(b)
    _, terms = forecast_loss(p, b, {})
    for loss in terms.values():
        for parameter in (net.scene.backbone.weight, net.hand.mlp[0].weight,
                          net.interaction.layers[0].self_attn.in_proj_weight):
            gradient = torch.autograd.grad(loss, parameter, retain_graph=True)[0]
            assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    parameters = list(net.hand_future.parameters()) + list(net.goal.parameters())
    assert all(g is None for g in torch.autograd.grad(terms["scene"], parameters,
                                                    retain_graph=True, allow_unused=True))


def test_missing_hands_goals_and_padded_objects_cannot_leak_or_create_nan():
    net, b = model(), batch()
    b["start_hand_valid"][:, 1] = False
    b["trajectory_valid"][:, :, 1] = False
    b["object_valid"][:, 1] = False
    b["scene_valid"][:, 1] = False
    b["object_goal_valid"][:] = False
    before = net(b)
    dirty = copy.deepcopy(b)
    dirty["start_hand"][:, 1] = float("nan")
    dirty["start_hand_history"][:, 1] = float("nan")
    dirty["scene_points"][:, 1] = float("nan")
    dirty["goal_object_points"][:] = float("nan")
    dirty["goal_object_delta"][:] = float("nan")
    dirty["start_object_points"][:] = float("nan")
    after = net(dirty)
    assert all(torch.equal(before[k], after[k]) and torch.isfinite(after[k]).all() for k in before)
    assert not after["trajectory"][:, :, 1].any() and not after["scene"][:, 1].any()
    forecast_loss(after, dirty, {})[0].backward()
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)


def test_scene_label_rotation_and_translation_use_the_current_anchor():
    points = np.array([[[1., 0., 0.], [0., 1., 0.]]], np.float32)
    effect = np.tile(np.eye(4, dtype=np.float32), (1, 24, 1, 1))
    effect[..., :3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    effect[..., :3, 3] = [2, 3, 4]
    labels = point_labels(dict(points=points, effect=effect), 2)
    np.testing.assert_allclose(labels["scene_target"][0, 0], [[1, 4, 4], [1, 2, 4]])
    np.testing.assert_array_equal(labels["scene_points"], points)


def test_scene_count_cannot_dilute_hand_loss_and_static_points_keep_weight():
    b = batch()
    p = dict(trajectory=torch.zeros_like(b["trajectory_target"]), scene=torch.zeros_like(b["scene_target"]))
    _, before = forecast_loss(p, b, {})
    more = dict(b, scene_target=b["scene_target"].repeat(1, 3, 1, 1, 1),
                scene_valid=b["scene_valid"].repeat(1, 3, 1, 1), object_valid=b["object_valid"].repeat(1, 3))
    _, after = forecast_loss(dict(p, scene=p["scene"].repeat(1, 3, 1, 1, 1)), more, {})
    torch.testing.assert_close(before["hand"], after["hand"])
    torch.testing.assert_close(before["scene"], after["scene"])
    b["scene_target"].zero_()
    _, static = forecast_loss(dict(p, scene=torch.ones_like(p["scene"]) * .05), b, {})
    torch.testing.assert_close(static["scene"], torch.tensor(1.))


def test_robot_predicted_action_updates_pair_but_GT_anchor_updates_R_only():
    net, b = RobotPair(model()).eval(), batch()
    b.update(native_state=torch.randn(2, 87), native_tau=torch.randn(2, 24, 33) * .02,
             native_action=torch.randn(2, 8, 12) * .1, has_action=torch.ones(2, dtype=torch.bool))
    prediction = net(b)
    assert prediction["action"].shape == (2, 8, 12)
    _, terms = robot_loss(prediction, b, {}, net.retargeter)
    actor = list(net.forecaster.parameters())
    for name in ("action", "anchor"):
        grads = torch.autograd.grad(terms[name], actor, retain_graph=True, allow_unused=True)
        norm = sum(float(g.abs().sum()) for g in grads if g is not None)
        assert norm > 0 if name == "action" else norm == 0
        rg = torch.autograd.grad(terms[name], list(net.retargeter.parameters()), retain_graph=True)
        assert sum(float(g.abs().sum()) for g in rg) > 0


def test_native_semantic_reordering_and_missing_right_hand_fail_closed():
    tau = torch.arange(11.)[None, None, None, :, None].expand(2, 24, 2, 11, 3)
    valid = torch.ones(2, 2, 11, dtype=torch.bool)
    native = native_trajectory(tau, valid).reshape(2, 24, 11, 3)
    torch.testing.assert_close(native[0, 0, :, 0], torch.tensor([0., 1., 6., 2., 7., 3., 8., 4., 9., 5., 10.]))
    valid[0, 0, 4] = False
    with pytest.raises(ValueError, match="right-hand"):
        native_trajectory(tau, valid)


@pytest.mark.parametrize("steps", [1, 4, 8])
def test_action_buffer_consumes_real_chunk_indices_without_requery(steps):
    buffer = ActionBuffer(steps)
    chunk = np.arange(96, dtype=np.float32).reshape(8, 12)
    for tick in range(steps * 2):
        if buffer.needs_query:
            buffer.refill(chunk + tick * 100, tick)
        command, receipt = buffer.pop()
        assert receipt == dict(query_tick=tick - tick % steps, chunk_index=tick % steps)
        np.testing.assert_array_equal(command, chunk[tick % steps] + receipt["query_tick"] * 100)
    assert buffer.needs_query


def test_parameter_groups_cover_every_parameter_once_and_keep_small_scene_lr():
    net = RobotPair(model())
    groups = net.parameter_groups(dict(scene_lr=1e-5, new_lr=1e-4))
    ids = [id(p) for group in groups for p in group["params"]]
    assert len(ids) == len(set(ids)) == len(list(net.parameters()))
    assert groups[0]["lr"] == 1e-5 and groups[-1]["lr"] == 1e-4


def test_mixed_batch_cannot_silently_discard_robot_action_labels():
    net, b = RobotPair(model()).eval(), batch()
    b["has_action"] = torch.tensor([True, False])
    with pytest.raises(ValueError, match="separate"):
        net(b)
    b["has_action"][:] = False
    assert set(net(b)) == {"scene", "trajectory"}


def test_checkpoint_is_complete_distinct_and_immutable(tmp_path):
    net = model()
    path = tmp_path / "step1.pt"
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4)
    config = dict(goal_encoding=GOAL_ENCODING)
    save_checkpoint(path, net, opt, config, stats(), 1, "pretrain", dict(engineering_only=True))
    packet = load_checkpoint(path)
    restored = model()
    restored.load_state_dict(packet["model"], strict=True)
    b = batch()
    assert all(torch.equal(net(b)[key], restored(b)[key]) for key in ("scene", "trajectory"))
    with pytest.raises(FileExistsError):
        save_checkpoint(path, net, opt, config, stats(), 1, "pretrain", {})
    wrong = tmp_path / "legacy.pt"
    torch.save(dict(schema="ref2dex.learned-native-retargeter.v1"), wrong)
    with pytest.raises(ValueError, match="full shared"):
        load_checkpoint(wrong)
    legacy = tmp_path / "absolute-goal.pt"
    old_packet = dict(packet, config={})
    torch.save(old_packet, legacy)
    with pytest.raises(ValueError, match="legacy absolute-goal"):
        load_checkpoint(legacy)


def test_goal_encoder_reads_current_and_delta_exactly_once_and_target_is_trainable():
    net, b = model(), batch()
    captured = []
    handle = net.goal.point[0].register_forward_pre_hook(lambda module, args: captured.append(args[0].detach()))
    prediction = net(b)
    handle.remove()
    torch.testing.assert_close(captured[0][..., :3], b["start_object_points"] / .05)
    torch.testing.assert_close(captured[0][..., 3:], b["goal_object_delta"] / .05)
    # Delta is not subtracted a second time, nor replaced with the raw endpoint.
    assert not torch.equal(captured[0][..., 3:], b["goal_object_points"] / .05)
    gradient = torch.autograd.grad(prediction["trajectory"].square().sum(), net.goal.target)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


def test_goal_delta_represents_translation_rotation_and_missing_pairs():
    start = torch.tensor([[[1., 0., 0.], [0., 1., 0.], [float("nan")] * 3]])
    rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    end = start @ rotation.T + torch.tensor([2., 3., 4.])
    valid = torch.tensor([[True, True, False]])
    delta = endpoint_delta(start, end, valid)
    torch.testing.assert_close(delta, torch.tensor([[[1., 4., 4.], [1., 2., 4.], [0., 0., 0.]]]))
    offset = torch.tensor([10., -20., 30.])
    torch.testing.assert_close(delta, endpoint_delta(start + offset, end + offset, valid))
    stationary = endpoint_delta(start, start, valid)
    assert torch.isfinite(stationary).all() and not stationary.any()
    with pytest.raises(ValueError, match="finite valid"):
        endpoint_delta(start, end, torch.ones_like(valid))


def test_paired_goal_permutation_preserves_hand_forecast():
    net, b = model(), batch()
    permutation = torch.tensor([2, 0, 3, 1])
    changed = dict(b, start_object_points=b["start_object_points"][:, permutation],
                   goal_object_delta=b["goal_object_delta"][:, permutation],
                   object_goal_valid=b["object_goal_valid"][:, permutation])
    torch.testing.assert_close(net(b)["trajectory"], net(changed)["trajectory"], atol=1e-6, rtol=1e-5)
