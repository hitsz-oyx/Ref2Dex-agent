import numpy as np
import torch
from wam_rl_policy_v3.bc_mlp import CurrentMLP,NormalizedCommandMLP
from wam_rl_policy_v3.shadow_task_probe import expand_official_state,TaskGoalScheduler


def test_bc_evaluation_rejects_any_ppo_checkpoint_before_runtime_initialization():
    import pytest
    from types import SimpleNamespace
    from wam_rl_policy_v3.shadow_task_probe import run_bc
    args=SimpleNamespace(bc_evaluation=True,evaluate=True,rl_checkpoint='ppo.pt',ppo_updates=0)
    with pytest.raises(ValueError,match='BC-only evaluation'):
        run_bc(None,None,None,None,None,args)


def test_trained_actor_and_critic_are_preserved_by_named_task_column_migration():
    torch.manual_seed(246)
    actor=CurrentMLP(565);critic=torch.nn.Linear(565,7)
    with torch.no_grad():actor.net[-1].weight.normal_(0,.01)
    state={'actor.'+k:v for k,v in actor.state_dict().items()}
    state.update({'critic.0.weight':critic.weight,'critic.0.bias':critic.bias,'log_std':torch.zeros(24)})
    expanded=expand_official_state(state)
    model=NormalizedCommandMLP(575,300,24)
    model.load_state_dict({k[6:]:v for k,v in expanded.items() if k.startswith('actor.')})
    native=torch.randn(16,565);task=torch.randn(16,10)
    x=torch.cat((native[:,:300],task,native[:,300:]),-1)
    torch.testing.assert_close(model(x),actor(native),atol=1e-6,rtol=1e-5)
    out=x@expanded['critic.0.weight'].T+expanded['critic.0.bias']
    torch.testing.assert_close(out,critic(native),atol=1e-6,rtol=1e-5)
    assert not expanded['actor.net.0.weight'][:,300:310].any()
    assert not expanded['critic.0.weight'][:,300:310].any()


def test_shadow_task_goal_comes_from_the_same_decision_as_task10():
    points=np.random.default_rng(246).normal(size=(32,3))*.01
    state=dict(pose=np.eye(4)[None],goal=np.array([[0,0,.2]]),velocity=np.zeros((1,6)),contact_forces=np.zeros((1,5,3)))
    scheduler=TaskGoalScheduler(state,0,(points,None,points))
    decision=scheduler.update(state['pose'][0],0.,hand_points=np.ones((11,3)))
    np.testing.assert_array_equal(decision['task'][-3:],[1,0,0])
    np.testing.assert_array_equal(decision['goal_delta'],0)
    assert scheduler.max_goal_error<1e-7
