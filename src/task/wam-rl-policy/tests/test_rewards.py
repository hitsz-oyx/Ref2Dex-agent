import torch
from wam_rl_policy.rewards import position_hold_bonus


def test_hold_bonus_requires_nearness_lift_and_low_speed():
    objects=torch.tensor([[0.,0.,.9],[0.,0.,.9],[0.,0.,.9],[0.,0.,.9]])
    goals=objects.clone();goals[1,0]=.1
    velocity=torch.zeros(4,3);velocity[2,0]=.2
    initial_z=torch.tensor([.7,.7,.7,.85])
    assert torch.equal(position_hold_bonus(objects,goals,velocity,initial_z),torch.tensor([1.,0.,0.,0.]))
