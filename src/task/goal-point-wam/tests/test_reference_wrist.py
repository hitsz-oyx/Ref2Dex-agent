import numpy as np
import torch

from goal_point_wam.reference_wrist import wrist_context, ReferenceWristAdapter, wrist_actor
from goal_point_wam.control_context import RobotControlAdapter, conditioned_actor
from goal_point_wam.model import TrajectoryDecoder


def test_current_measured_anchor_and_rigid_transform():
    obj = np.eye(4, dtype=np.float32)
    obj[:3, 3] = [1, 2, 3]
    pos = np.array([[2, 2, 3], [1, 4, 3], [1, 2, 6]], np.float32)
    rot = np.tile(np.eye(3, dtype=np.float32), (3, 1, 1))
    value = wrist_context(obj, pos, rot)
    np.testing.assert_array_equal(value.reshape(3, 9)[:, :3], np.diag([1, 2, 3]))
    common = np.eye(4, dtype=np.float32)
    common[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    common[:3, 3] = [4, -2, 7]
    changed = wrist_context(common @ obj, pos @ common[:3, :3].T + common[:3, 3], common[:3, :3] @ rot)
    np.testing.assert_allclose(value, changed)


def test_zero_gate_base_equivalence_human_mask_and_connected_gradient():
    torch.set_num_threads(1)
    torch.manual_seed(295)
    class Model:
        trajectory = TrajectoryDecoder(layers=1).eval()
        control_adapter = RobotControlAdapter()
    model = Model()
    adapter = ReferenceWristAdapter()
    with torch.no_grad():
        model.trajectory.head[-1].weight.normal_(0, .01)
        model.control_adapter.alpha_p.fill_(.2)
    context = (dict(scene_tokens=torch.randn(2, 4, 128), scene_mask=torch.ones(2, 4, dtype=torch.bool)),
               torch.randn(2, 3, 128), torch.ones(2, 3, dtype=torch.bool), torch.randn(2, 2, 11, 128))
    batch = dict(proprio=torch.randn(2, 48), robot_goal=torch.randn(2, 27),
                 robot_mask=torch.tensor([True, False]), wrist_goal=torch.randn(2, 27))
    old = conditioned_actor(model, context, model.control_adapter, batch)
    assert torch.equal(old, wrist_actor(model, adapter, context, batch))
    batch['wrist_goal'][1] = float('nan')
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-3)
    adapter(batch['wrist_goal'], batch['robot_mask']).sum().backward()
    assert adapter.alpha.grad.abs() > 0
    optimizer.step(); optimizer.zero_grad()
    adapter(batch['wrist_goal'], batch['robot_mask']).sum().backward()
    assert sum(float(p.grad.abs().sum()) for p in adapter.net.parameters()) > 0
    new = wrist_actor(model, adapter, context, batch)
    assert torch.isfinite(new).all() and torch.equal(old[1], new[1])
    assert not torch.equal(old[0], new[0])
