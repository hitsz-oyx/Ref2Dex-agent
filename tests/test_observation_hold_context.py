from types import SimpleNamespace
import torch
from src.task.CmResidual.observation_hold_policy import hold_context,canonical_action


def test_only_current_physical_state_and_next_available_plan_enter_policy():
    refs=torch.zeros(1,1,10,155);refs[0,0,3,119:137]=.2
    task=SimpleNamespace(num_envs=1,data_id=torch.tensor([0]),ref_index=torch.tensor([0]),progress_buf=torch.tensor([2]),hoi_refs=refs,
        _dof_pos=torch.zeros(1,18),_dof_vel=torch.zeros(1,18),_target_states=torch.zeros(1,13),
        _contact_forces=torch.zeros(1,5,3),_contact_body_ids=torch.arange(5),_tar_contact_forces=torch.zeros(1,3))
    context=hold_context(task,torch.tensor([5]))
    refs[0,0,4:]=100
    assert torch.equal(context,hold_context(task,torch.tensor([5])))
    task._target_states[0,2]=1.
    changed=hold_context(task,torch.tensor([5]))
    assert changed[0,38]==1 and context[0,38]==0
    assert context.shape==(1,70)


def test_null_commands_zero_without_changing_effective_commands_or_input():
    action=torch.ones(2,18)*.2;canonical=canonical_action(action)
    assert torch.equal(canonical[:,:6],action[:,:6])
    assert not canonical[:,[7,9,11,13,16,17]].any()
    assert (action==.2).all()
