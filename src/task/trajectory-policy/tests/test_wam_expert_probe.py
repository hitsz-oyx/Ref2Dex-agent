"""Expose actual expert means without bypassing player preprocessing/clipping."""
import importlib.util
from pathlib import Path

import pytest
import torch

path=Path(__file__).resolve().parents[1]/'tools/run/wam_expert_probe.py'
spec=importlib.util.spec_from_file_location('wam_expert_probe_test',path)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Player:
    is_rnn=False
    has_batch_dimension=True
    clip_actions=True
    actions_low=torch.full((18,),-1.)
    actions_high=torch.ones(18)

    def __init__(self):
        self.calls=[]

    def get_action(self, observation, deterministic):
        obs=observation['obs']
        assert deterministic and obs.shape==(64,20)
        assert torch.equal(obs,obs[:1].repeat(64,1))
        self.calls.append(obs.clone())
        mean=obs[:,:18]*2
        return mean.clamp(-1,1) if self.clip_actions else mean


def test_expert_records_preclamp_same_row_mean_and_restores_player():
    player=Player()
    obs=torch.stack((torch.full((20,),.8),torch.full((20,),-.1)))
    raw,action,clipped=module.expert_on_rows(player,{'obs':obs},verify_official=True)
    assert player.clip_actions
    assert len(player.calls)==4
    assert torch.equal(raw,obs[:,:18]*2)
    assert torch.equal(action,raw.clamp(-1,1))
    assert clipped.tolist()==[True,False]
    assert torch.equal(player.calls[0][0],obs[0])
    assert torch.equal(player.calls[1][0],obs[1])


def test_expert_restores_flag_after_player_failure():
    player=Player()
    def failure(*args):
        raise RuntimeError('inference failure')
    player.get_action=failure
    with pytest.raises(RuntimeError):
        module.expert_on_rows(player,{'obs':torch.zeros(4,20)})
    assert player.clip_actions


def test_prefix_equality_checks_bool_and_actual_float_state():
    current={'q':torch.zeros(4,18),'table_footprint':torch.tensor([True,True,False,False])}
    assert max(module.paired_prefix_error(current).values())==0
    current['q'][3,0]=1e-7
    assert module.paired_prefix_error(current)['q']>0


def test_prefix_physical_bounds_allow_tiny_solver_differences_but_reject_drift():
    q=torch.zeros(4,18)
    hand=torch.zeros(4,11,3)
    obj=torch.eye(4).repeat(4,1,1)
    current={'q':q,'hand':hand,'obj':obj}
    q[3,0]=4.768e-7
    hand[3,:,0]=7.75e-7
    obj[3,0,3]=2.17e-5
    metrics=module.prefix_physical_metrics(current)
    assert all(metrics[key]<=bound for key,bound in module.PREFIX_PHYSICAL_BOUNDS.items())
    q[3,0]=.054
    assert module.prefix_physical_metrics(current)['q_xyz_norm_m']>module.PREFIX_PHYSICAL_BOUNDS['q_xyz_norm_m']
