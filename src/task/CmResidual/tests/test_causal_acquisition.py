import torch
from src.task.CmResidual.causal_acquisition import eligible_steps,first_geometry_trigger,axis_pulse


def test_trigger_uses_current_geometry_and_leaves_missing_explicit():
    state=torch.zeros(20,3,55);state[...,49:51]=1
    done=torch.zeros(20,3,dtype=torch.bool)
    gap=torch.ones(20,3);gap[12,0]=.01;gap[11,1]=.02
    triggers=first_geometry_trigger(eligible_steps(state,done),gap)
    assert triggers.tolist()==[12,11,-1]


def test_terminal_five_step_window_and_warmup_are_excluded():
    state=torch.zeros(20,1,55);state[...,49:51]=1
    done=torch.zeros(20,1,dtype=torch.bool);done[15]=True
    result=eligible_steps(state,done)[:,0]
    assert result.nonzero().flatten().tolist()==[10]


def test_axis_changes_only_scheduled_environment_without_mutation():
    source=torch.zeros(3,18);schedule=torch.tensor([12,-1,11])
    result=axis_pulse(source,schedule,12,0,-.01)
    assert result[0,0]==-.01 and torch.count_nonzero(result)==1
    assert torch.count_nonzero(source)==0
