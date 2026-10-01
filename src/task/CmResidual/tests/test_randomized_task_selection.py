import pytest
import torch
from src.task.CmResidual.randomized_task_selection import candidate_validity,choose_actions


def test_clipped_candidates_are_masked_and_actor_is_preserved():
    action=torch.zeros(4,18);action[:,0]=1
    valid=candidate_validity(action);assert not valid[:,1].any() and valid[:,0].all()
    scores=torch.zeros(4,7);scores[:,1]=100;scores[:,5]=3
    selected=choose_actions(torch.arange(4),scores,torch.tensor([0.,99.,0.,0.,0.,2.,0.]),valid,torch.full((4,),2))
    assert selected.tolist()==[0,2,5,5]
    assert torch.equal(action[:,0],torch.ones(4))


def test_zero_candidate_wins_ties_without_unrequested_correction():
    selected=choose_actions(torch.full((3,),3),torch.zeros(3,7),torch.zeros(7),torch.ones(3,7,dtype=torch.bool),torch.zeros(3,dtype=torch.long))
    assert selected.tolist()==[0,0,0]


def test_invalid_random_choice_is_rejected():
    valid=torch.ones(1,7,dtype=torch.bool);valid[:,1]=False
    with pytest.raises(ValueError):choose_actions(torch.ones(1,dtype=torch.long),torch.zeros(1,7),torch.zeros(7),valid,torch.ones(1,dtype=torch.long))


@pytest.mark.parametrize('held_steps,lost_steps',[(44,6),(45,5),(45,6)])
def test_task_audit_uses_45_step_success_and_six_step_drop(held_steps,lost_steps):
    from scripts.analyze_randomized_task_selection import audit_episode
    from src.task.CmResidual.physical_value_contract import HoldTracker
    n=held_steps+lost_steps;height=torch.full((n,1),.04);contact=torch.ones(n,1,2)
    contact[held_steps:]=0;tracker=HoldTracker(1,'cpu');tracker.reset(torch.tensor([0]),torch.tensor([0.]))
    for h,c in zip(height,contact):tracker.step(h,c.bool().all(-1))
    stable=bool(tracker.stable[0]);drop=bool(tracker.drop_after_success[0])
    assert stable==(held_steps>=45)
    assert drop==(held_steps>=45 and lost_steps>=6)
    done=torch.zeros(n,1,dtype=torch.bool);done[-1]=True
    audit_episode(dict(height=height,contact=contact,active=torch.ones(n,1,dtype=torch.bool),done=done,initial_height=torch.zeros(1)),
        dict(environment=0,steps=n,stable_success=stable,drop_after_success=drop,retained_success=stable and not drop,
            max_hold_seconds=float(tracker.max_run[0]),mean_lift_meters=.04))
