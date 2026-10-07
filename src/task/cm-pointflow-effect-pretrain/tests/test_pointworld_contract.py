"""PointWorld point identity, normalized loss and real CUDA input isolation."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

TASK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK/'src'))
from oakink_wm.pointworld import PointWorldWM, capped_collate
from oakink_wm.data import Windows


def test_point_cap_retains_each_object_and_full_supervision():
    M = 10
    sample = dict(xyz=np.zeros((M*512+22,3), 'float32'),
                  features=np.zeros((M*512+22,18), 'float32'),
                  scene_object=np.r_[np.repeat(np.arange(M),512),np.full(22,-1)],
                  action=np.zeros((24,22,9),'float32'), action_valid=np.ones((24,22),bool),
                  sample_id=np.zeros(3,np.int64), hand_presence=np.ones(2,bool),
                  points=np.zeros((M,512,3),'float32'), radius=np.ones(M,'float32'),
                  object_features=np.zeros((M,15),'float32'),
                  effect=np.tile(np.eye(4,dtype='float32'),(M,24,1,1)),category=0)
    b = capped_collate([sample])
    assert b['point_valid'].sum() <= 4096
    assert b['points'].shape == (1,M,512,3)
    for m in range(M): assert (b['scene_object']==m).sum() > 0
    assert (b['scene_object']==-1).sum() == 22


def real_inputs():
    data, stats = os.environ.get('POINTWORLD_DATA'), os.environ.get('POINTWORLD_STATS')
    if not data or not stats or not torch.cuda.is_available():
        pytest.skip('Set POINTWORLD_DATA/STATS for real GPU semantic contract')
    dataset = Windows(data,'train')
    samples = [dataset[int(dataset.groups[0][i])] for i in (0,20)]
    b = {k:v.cuda() for k,v in capped_collate(samples).items()}
    torch.manual_seed(9)
    return PointWorldWM(json.loads(Path(stats).read_text())).cuda(), b


def test_real_gpu_masks_labels_rigidity_action_gradient_and_checkpoint():
    model,b = real_inputs()
    model.eval()
    class NoLabels(dict):
        def __getitem__(self,key):
            if key=='effect': raise AssertionError('future effect label leaked')
            return super().__getitem__(key)
    with torch.no_grad():
        original = model(b)
        repeated = model(NoLabels(b))
        torch.testing.assert_close(original['translation'],repeated['translation'],rtol=0,atol=1e-6)
        masked = dict(b)
        masked['action_valid'] = b['action_valid'].clone()
        masked['action_valid'][:,:,11:] = False
        before = model(masked)
        altered = dict(masked)
        altered['action'] = b['action'].clone()
        altered['action'][:,:,11:] += 1000
        after = model(altered)
        torch.testing.assert_close(before['translation'],after['translation'],rtol=0,atol=1e-6)
        altered['action'] = b['action'].clone()
        altered['action'][:,:,:11,:3] += .1
        changed = model(altered)
        assert not torch.allclose(before['translation'],changed['translation'],rtol=0,atol=1e-8)
        history = model(b,'history')
        altered = dict(b,action=b['action']+1000)
        torch.testing.assert_close(history['translation'],model(altered,'history')['translation'],rtol=0,atol=1e-6)
        R = original['rotation']
        torch.testing.assert_close(R.transpose(-1,-2)@R,torch.eye(3,device='cuda').expand_as(R),rtol=0,atol=2e-5)
        perfect = dict(rotation=b['effect'][...,:3,:3],translation=b['effect'][...,:3,3])
        loss,_ = model.loss(perfect,b)
        assert loss.item() < 1e-10
    model.train()
    loss,_ = model.loss(model(b),b)
    loss.backward()
    g = model.action_proj[0].weight.grad
    assert torch.isfinite(g).all() and g.abs().sum()>0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    model.eval()
    with torch.no_grad():
        expected=model(b)
        state={k:v.clone() for k,v in model.state_dict().items()}
        model.head[-1].weight.add_(1)
        model.load_state_dict(state)
        restored=model(b)
        torch.testing.assert_close(expected['translation'],restored['translation'],rtol=0,atol=1e-6)
        for k,v in model.state_dict().items(): assert torch.equal(v,state[k])
