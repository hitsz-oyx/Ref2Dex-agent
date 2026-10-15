from copy import deepcopy
from pathlib import Path
import sys
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from ptv3_shared_wam_v3.lr_transition import validate_lr_resume, apply_learning_rates
from ptv3_shared_wam_v3.checkpoint import restore_optimizer


def packet():
    c=dict(geometry_lr=1e-4,shared_lr=1e-4,seed=170,updates=80000,warmup_updates=500)
    return dict(config=c,stats={'recipe':'strict','normalizer':[1,2]},step=37000,
                provenance={'world_size':4},rank_rng=[{} for _ in range(4)])


def test_lr_transition_rejects_data_model_seed_budget_and_upward_lr_changes():
    p=packet();c={**p['config'],'geometry_lr':3e-5,'shared_lr':3e-5}
    assert validate_lr_resume(p,c,p['stats'],4)['parent_step']==37000
    for key,value in [('seed',171),('updates',90000),('new_architecture',True),('geometry_lr',2e-4),('geometry_lr',0),('shared_lr',float('nan'))]:
        with pytest.raises(ValueError):validate_lr_resume(p,{**c,key:value},p['stats'],4)
    with pytest.raises(ValueError):validate_lr_resume(p,c,{'recipe':'expanded'},4)
    with pytest.raises(ValueError):validate_lr_resume(p,c,p['stats'],2)
    with pytest.raises(ValueError):validate_lr_resume(p,p['config'],p['stats'],4)


def test_optimizer_moments_and_step_survive_lr_drop_and_next_update_matches_control():
    p=torch.nn.Parameter(torch.tensor([.5,-.3]));opt=torch.optim.AdamW([dict(params=[p],name='joint_geometry')],lr=1e-4)
    for _ in range(3):opt.zero_grad();p.square().sum().backward();opt.step()
    state=deepcopy(opt.state_dict());q=torch.nn.Parameter(p.detach().clone())
    restored=torch.optim.AdamW([dict(params=[q],name='joint_geometry')],lr=3e-5)
    restore_optimizer(restored,state)
    c=dict(geometry_lr=3e-5,shared_lr=3e-5,warmup_updates=500)
    assert apply_learning_rates(restored,c,37001)=={'joint_geometry':3e-5}
    for key in ('step','exp_avg','exp_avg_sq'):assert torch.equal(opt.state[p][key],restored.state[q][key])
    apply_learning_rates(opt,c,37001)
    for x,o in ((p,opt),(q,restored)):o.zero_grad();x.square().sum().backward();o.step()
    assert torch.equal(p,q)
    assert restored.state[q]['step'].item()==4
