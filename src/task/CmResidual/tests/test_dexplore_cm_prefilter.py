"""Conservative Cmv2 interaction gating and zero-reward skip contract."""
from types import SimpleNamespace

import pytest
import torch

from src.task.CmResidual.dexplore_cm_reward import (
    CmRewardConfig, predict_cm_reward, swept_object_sphere_candidates,
)


def _point_distance_to_segment(object_point, hand_point, flow):
    relative = hand_point - object_point
    alpha = (-(relative * flow).sum() / flow.square().sum().clamp_min(1e-12)).clamp(0, 1)
    return (relative + alpha * flow).norm()


def test_sphere_prefilter_never_skips_actual_swept_contact():
    generator = torch.Generator().manual_seed(42)
    objects = torch.randn(20, 11, 3, generator=generator) * 0.16
    hands = torch.randn(20, 17, 3, generator=generator) * 0.4
    flows = torch.randn(20, 17, 3, generator=generator) * 0.35
    # Also cover a long segment that passes through the object with distant endpoints.
    hands[0, 0] = torch.tensor([-2.0, 0.0, 0.0])
    flows[0, 0] = torch.tensor([4.0, 0.0, 0.0])
    objects[0, 0] = 0
    hands[-1, :, 0] = 4.0
    hands[-1, :, 1:] = 0.0
    flows[-1] = 0.0
    candidate = swept_object_sphere_candidates(objects, hands, flows)
    assert candidate[0]
    for batch in range(objects.shape[0]):
        actual_contact = any(
            _point_distance_to_segment(point, hand, flow) < 0.02
            for point in objects[batch] for hand, flow in zip(hands[batch], flows[batch]))
        if actual_contact:
            assert candidate[batch], f"false-negative broad phase at sample {batch}"
    assert (~candidate).any()


def test_sphere_prefilter_rejects_nonfinite():
    with pytest.raises(FloatingPointError, match="finite"):
        swept_object_sphere_candidates(torch.zeros(1, 1, 3),
                                       torch.tensor([[[float("nan"), 0, 0]]]), torch.zeros(1, 1, 3))


def test_predict_cm_reward_only_infers_candidates_and_zero_fills_skips():
    count = 2
    object_pose = torch.eye(4).expand(count, 4, 4).clone()
    object_points = torch.zeros(count, 1024, 3)
    object_points[:, :512, 0] = 0.01
    object_points[:, 512:, 0] = -0.01
    hand_points = torch.zeros(count, 1538, 3)
    hand_points[0, :, 0] = 0.01
    hand_points[1, :, 0] = 3.0

    class Bridge:
        def current(self, current_native, object_root_state):
            return SimpleNamespace(object_points=object_points, object_pose=object_pose,
                                   object_normals=torch.zeros_like(object_points))

        def nominal_hand_sweep(self, current_native, actions, lower, upper):
            return hand_points, torch.zeros_like(hand_points), torch.zeros(count, 1, 1538, 3)

    class Adapter:
        calls = 0

        def predict_effect_only(self, object_points, *args, **kwargs):
            self.calls += 1
            assert object_points.shape[0] == 1
            return {"delta_xi_root": torch.tensor([[0.005, 0, 0, 0, 0, 0.]]),
                    "token_mask": torch.ones(1, 16, dtype=torch.bool),
                    "token_mass": torch.ones(1, 16)}

    adapter = Adapter()
    goals = object_pose.clone()
    goals[:, 0, 3] = 0.01
    output = predict_cm_reward(adapter=adapter, bridge=Bridge(), action=torch.zeros(count, 18),
                               current_native=torch.zeros(count, 18),
                               object_root_state=torch.zeros(count, 13), goal_pose=goals,
                               lower=torch.zeros(18), upper=torch.ones(18))
    assert adapter.calls == 1
    assert output["prefilter_candidate"].tolist() == [True, False]
    assert output["valid"].tolist() == [True, False]
    assert output["reward"][0] > 0
    assert output["reward"][1] == output["improvement"][1] == 0
    assert torch.equal(output["predicted_delta_xi"][1], torch.zeros(6))

    # All-far batches never call the effect model and preserve batch shape.
    hand_points[:, :, 0] = 3.0
    skipped = predict_cm_reward(adapter=adapter, bridge=Bridge(), action=torch.zeros(count, 18),
                                current_native=torch.zeros(count, 18),
                                object_root_state=torch.zeros(count, 13), goal_pose=goals,
                                lower=torch.zeros(18), upper=torch.ones(18))
    assert adapter.calls == 1
    assert not skipped["prefilter_candidate"].any()
    assert not skipped["valid"].any()
    assert torch.equal(skipped["reward"], torch.zeros(count))
