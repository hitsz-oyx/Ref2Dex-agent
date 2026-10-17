import json
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn

TASK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK / "src"))
from ptv3_shared_wam_v4.interaction import C_DIM, C_NAMES, interaction_target
from ptv3_shared_wam_v4.model import SharedWAMV4
from ptv3_shared_wam_v4.objectives import forecast_loss
from ptv3_shared_wam_v4.paths import SCHEMA, STATS_SCHEMA


class Geometry(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.project = nn.Linear(width, width)

    def forward(self, batch, hand, valid):
        tokens = self.project(hand.mean(1, keepdim=True).expand(-1, 4, -1))
        return dict(tokens=tokens, xyz=torch.zeros(len(hand), 4, 3, device=hand.device),
                    mask=torch.ones(len(hand), 4, dtype=torch.bool, device=hand.device),
                    hand=self.project(hand))


def config():
    return dict(schema=SCHEMA, interaction_encoding="object-centric-contact-approach.v1",
        interaction_dim=8, goal_encoding="current-point-horizon-delta.v1",
        task_conditioning="shared-scene-hand-goal.v1", horizon=24, point_width=8, shared_width=16,
        shared_heads=4, shared_layers=2, head_width=8, head_heads=2,
        position_scale_m=.05, velocity_scale_mps=.5, acceleration_scale_mps2=5.,
        checkpoint_interaction=False, interaction_scales=[1., .05, .05, .05, .05, .5, .5, .5])


def stats():
    return dict(schema=STATS_SCHEMA, split="train", source_ids=[0, 1, 2, 4, 5, 6, 8, 9, 10],
        fps=30, history=4, horizon=24, units="m",
        hand=dict(mean=np.zeros((24, 3)).tolist(), std=np.ones((24, 3)).tolist()),
        scene=dict(mean=np.zeros((24, 3)).tolist(), std=np.ones((24, 3)).tolist()))


def batch():
    torch.manual_seed(4)
    return dict(start_hand=torch.randn(2, 2, 11, 3) * .02,
        start_hand_history=torch.randn(2, 2, 11, 9) * .01,
        start_hand_valid=torch.ones(2, 2, 11, dtype=torch.bool),
        xyz=torch.rand(2, 8, 3), features=torch.zeros(2, 8, 18),
        point_valid=torch.ones(2, 8, dtype=torch.bool),
        scene_points=torch.randn(2, 2, 4, 3) * .02,
        object_valid=torch.tensor([[True, True], [True, False]]),
        start_object_points=torch.rand(2, 3, 3) * .02,
        goal_object_delta=torch.rand(2, 3, 3) * .05,
        object_goal_valid=torch.ones(2, 3, dtype=torch.bool),
        trajectory_target=torch.randn(2, 24, 2, 11, 3) * .01,
        trajectory_valid=torch.ones(2, 24, 2, 11, dtype=torch.bool),
        scene_target=torch.randn(2, 2, 24, 4, 3) * .002,
        scene_valid=torch.ones(2, 2, 24, 4, dtype=torch.bool))


def model():
    c = config()
    return SharedWAMV4(stats(), c, Geometry(c["point_width"])).eval()


def test_c_contract_and_target_are_masked():
    b = batch()
    target, valid = interaction_target(b)
    assert target.shape == (2, 2, 24, C_DIM)
    assert C_NAMES[0] == "contact" and valid.shape == (2, 2, 24)
    assert not valid[1, 1].any() and not target[1, 1].any()
    assert torch.isfinite(target).all()


def test_v4_emits_tau_scene_and_interaction_and_goal_reaches_c():
    m = model(); b = batch()
    first = m(b)
    assert first["trajectory"].shape == (2, 24, 2, 11, 3)
    assert first["scene"].shape == (2, 2, 24, 4, 3)
    assert first["interaction"].shape == (2, 2, 24, C_DIM)
    assert not first["interaction"][1, 1].any()
    b["goal_object_delta"] = b["goal_object_delta"] + .2
    second = m(b)
    assert not torch.allclose(first["trajectory"], second["trajectory"])
    assert not torch.allclose(first["interaction"], second["interaction"])


def test_interaction_loss_backpropagates_to_new_head():
    c = config(); m = SharedWAMV4(stats(), c, Geometry(c["point_width"]))
    b = batch(); p = m(b)
    total, terms = forecast_loss(p, b, {**c, "hand_weight": 1., "scene_weight": 1.,
        "motion_tau_m": .002, "motion_floor": .1}, m.normalizer)
    assert torch.isfinite(total) and torch.isfinite(terms["interaction"])
    grad = torch.autograd.grad(terms["interaction"], list(m.interaction_future.parameters()),
                               allow_unused=True)
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grad)
