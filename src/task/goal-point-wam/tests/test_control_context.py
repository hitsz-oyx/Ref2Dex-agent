import numpy as np
import torch

from goal_point_wam.control_context import RobotControlAdapter, conditioned_actor
from goal_point_wam.model import TrajectoryDecoder
from trajectory_policy.robot_context import control_context
from trajectory_policy.retargeter_data import ACTIVE


def test_relative_goal_columns_and_previous_active_native_command():
    obj = np.eye(4,dtype=np.float32)
    obj[:3,:3] = [[0,-1,0],[1,0,0],[0,0,1]]
    obj[:3,3] = [1,2,3]
    ref = np.repeat(obj[None],3,axis=0)
    ref[:,:3,3] += [0,1,0]
    previous = np.arange(18,dtype=np.float32)
    prop,goal = control_context(np.zeros(18),np.ones(18),previous,obj,ref)
    np.testing.assert_array_equal(prop[36:],previous[list(ACTIVE)])
    np.testing.assert_allclose(goal.reshape(3,9),np.tile([1,0,0,1,0,0,0,1,0],(3,1)))
    # A common rigid transform must not change the relative goal input.
    common = np.eye(4,dtype=np.float32)
    common[:3,:3] = [[0,0,1],[0,1,0],[-1,0,0]]
    common[:3,3] = [-2,4,7]
    _,transformed = control_context(np.zeros(18),np.ones(18),previous,common@obj,common@ref)
    np.testing.assert_allclose(goal,transformed)


def test_zero_gates_preserve_actual_trajectory_decoder_and_human_mask():
    torch.set_num_threads(1);torch.manual_seed(295)
    adapter = RobotControlAdapter()
    decoder = TrajectoryDecoder(layers=1).eval()
    with torch.no_grad():
        decoder.head[-1].weight.normal_(0,.01)
    class Model:
        trajectory = decoder
    context = (dict(scene_tokens=torch.randn(2,4,128),scene_mask=torch.ones(2,4,dtype=torch.bool)),
        torch.randn(2,3,128),torch.ones(2,3,dtype=torch.bool),torch.randn(2,2,11,128))
    batch = dict(proprio=torch.randn(2,48),robot_goal=torch.randn(2,27),robot_mask=torch.tensor([True,False]))
    old = decoder(context[0]['scene_tokens'],context[0]['scene_mask'],context[1],context[2],context[3])
    assert torch.equal(old,conditioned_actor(Model(),context,adapter,batch))
    with torch.no_grad():
        adapter.alpha_p.fill_(.1);adapter.alpha_g.fill_(.2)
    batch['proprio'][1] = float('nan');batch['robot_goal'][1] = float('nan')
    new = conditioned_actor(Model(),context,adapter,batch)
    assert torch.equal(old[1],new[1]) and torch.isfinite(new).all()
    assert not torch.equal(old[0],new[0])


def test_gate_then_mlp_gradients_and_masked_input_gradients():
    torch.manual_seed(295)
    adapter = RobotControlAdapter()
    prop = torch.randn(2,48,requires_grad=True)
    goal = torch.randn(2,27,requires_grad=True)
    mask = torch.tensor([True,False])
    opt = torch.optim.AdamW(adapter.parameters(),lr=1e-4)
    adapter(prop,goal,mask).sum().backward()
    assert adapter.alpha_p.grad.abs()>0 and adapter.alpha_g.grad.abs()>0
    assert all(p.grad.count_nonzero()==0 for p in adapter.prop.parameters())
    opt.step();opt.zero_grad();prop.grad=None;goal.grad=None
    adapter(prop,goal,mask).sum().backward()
    assert sum(float(p.grad.abs().sum()) for p in adapter.prop.parameters())>0
    assert sum(float(p.grad.abs().sum()) for p in adapter.robot_goal.parameters())>0
    assert prop.grad[1].count_nonzero()==0 and goal.grad[1].count_nonzero()==0
