import json
from pathlib import Path
import sys
import numpy as np
import pytest
import torch
TASK=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(TASK/'src'))
from ptv3_shared_wam_v3.normalization import HorizonNormalizer
from ptv3_shared_wam_v3.paths import SOURCE_IDS, STATS_SCHEMA
from ptv3_shared_wam_v3.data import NineWindows, Steps
from ptv3_shared_wam_v3.checkpoint import save_checkpoint,load_checkpoint,restore_optimizer
from ptv3_shared_wam_v3.objectives import forecast_loss,loss_denominators
from ptv3_shared_wam_v2.objectives import forecast_loss as original_loss


def stats():
    return dict(schema=STATS_SCHEMA,split='train',source_ids=list(SOURCE_IDS),fps=30,history=4,horizon=24,units='m',
        source_probabilities={str(s):1/9 for s in SOURCE_IDS},
        **{name:dict(mean=np.full((24,3),.01).tolist(),std=np.full((24,3),.1).tolist()) for name in ('hand','scene')})


def test_normalization_roundtrip_and_reject_dev():
    n=HorizonNormalizer(stats())
    for name,shape in [('hand',(2,24,2,11,3)),('scene',(2,3,24,32,3))]:
        value=torch.randn(*shape)
        assert torch.allclose(value,n.denormalize(name,n.normalize(name,value)),atol=1e-6)
    for change in [dict(split='dev'),dict(source_ids=list(SOURCE_IDS[:-1])),dict(fps=15)]:
        with pytest.raises(ValueError):HorizonNormalizer({**stats(),**change})


def test_draw_stateless_resume_and_rank_separation():
    d=NineWindows.__new__(NineWindows);d.split='train'
    d.recipe={'source_probabilities':{str(s):1/9 for s in SOURCE_IDS}}
    d.draw_source=lambda sid,count,rng: int(sid)*100000+rng.integers(0,100000,count)
    whole=list(Steps(d,128,170,0,1,8))
    resumed=list(Steps(d,128,170,0,5,8))
    assert whole[4:]==resumed
    assert whole!=list(Steps(d,128,170,1,1,8))
    assert set(np.asarray(whole).ravel()//100000)==set(SOURCE_IDS)
    d.split='dev'
    with pytest.raises(ValueError):d.draw(1,170)


def test_full_checkpoint_optimizer_and_rng(tmp_path):
    torch.manual_seed(17)
    m=torch.nn.Linear(3,2);opt=torch.optim.AdamW(m.parameters(),lr=.01)
    m(torch.randn(2,3)).square().sum().backward();opt.step()
    rng=torch.get_rng_state();path=tmp_path/'step1.pt'
    save_checkpoint(path,m,opt,{},stats(),1,{'world_size':1},[{'torch':rng}])
    state=load_checkpoint(path);other=torch.nn.Linear(3,2);other.load_state_dict(state['model'])
    opt2=torch.optim.AdamW(other.parameters(),lr=.01);restore_optimizer(opt2,state['optimizer'])
    assert torch.equal(state['rank_rng'][0]['torch'],rng)
    for a,b in zip(m.parameters(),other.parameters()):assert torch.equal(a,b)
    assert all(v['step'].item()==1 for v in opt2.state.values())
    with pytest.raises(FileExistsError):save_checkpoint(path,m,opt,{},stats(),1,{},[])


def test_invalid_targets_ignored_and_distributed_denominators():
    n=HorizonNormalizer(stats())
    batch=dict(trajectory_valid=torch.ones(2,24,2,11,dtype=torch.bool),start_hand_valid=torch.ones(2,2,11,dtype=torch.bool),
        scene_valid=torch.ones(2,2,24,32,dtype=torch.bool),object_valid=torch.ones(2,2,dtype=torch.bool),
        trajectory_target=torch.randn(2,24,2,11,3),scene_target=torch.randn(2,2,24,32,3))
    batch['start_hand_valid'][0,1]=False;batch['object_valid'][0,1]=False
    pred={k+'_normalized':torch.zeros_like(batch[k+'_target']) for k in ('trajectory','scene')}
    c=dict(hand_weight=1,scene_weight=1,motion_tau_m=.002,motion_floor=.1)
    before=forecast_loss(pred,batch,c,n)[0]
    batch['trajectory_target'][0,:,1]=float('nan');batch['scene_target'][0,1]=float('nan')
    assert torch.equal(before,forecast_loss(pred,batch,c,n)[0])
    assert torch.equal(before,original_loss(pred,batch,c,n)[0])
    counts=loss_denominators(batch,c)
    assert counts[0]==24*33
    assert counts[1]>0 and torch.isfinite(counts).all()


def test_ddp_unequal_masks_matches_combined_batch(monkeypatch):
    from ptv3_shared_wam_v3 import objectives
    n=HorizonNormalizer(stats())
    c=dict(hand_weight=1,scene_weight=1,motion_tau_m=.002,motion_floor=.1)
    batch=dict(trajectory_valid=torch.ones(2,24,2,11,dtype=torch.bool),start_hand_valid=torch.ones(2,2,11,dtype=torch.bool),
        scene_valid=torch.ones(2,2,24,32,dtype=torch.bool),object_valid=torch.ones(2,2,dtype=torch.bool),
        trajectory_target=torch.randn(2,24,2,11,3)*.03,scene_target=torch.randn(2,2,24,32,3)*.003)
    batch['start_hand_valid'][0,1]=False;batch['object_valid'][0,1]=False
    prediction={k+'_normalized':torch.randn_like(batch[k+'_target'],requires_grad=True) for k in ('trajectory','scene')}
    expected=forecast_loss(prediction,batch,c,n)[0]
    counts=loss_denominators(batch,c)
    monkeypatch.setattr(objectives.dist,'is_initialized',lambda:True)
    monkeypatch.setattr(objectives.dist,'get_world_size',lambda:2)
    monkeypatch.setattr(objectives.dist,'all_reduce',lambda value:value.copy_(counts))
    terms=[]
    for rank in (0,1):
        terms.append(forecast_loss({k:v[rank:rank+1] for k,v in prediction.items()},
            {k:v[rank:rank+1] for k,v in batch.items()},c,n,distributed=True)[0])
    actual=sum(terms)/2
    assert torch.allclose(actual,expected,atol=1e-6)
    expected_grad=torch.autograd.grad(expected,list(prediction.values()),retain_graph=True)
    actual_grad=torch.autograd.grad(actual,list(prediction.values()))
    for a,b in zip(actual_grad,expected_grad):assert torch.allclose(a,b,atol=1e-7)
