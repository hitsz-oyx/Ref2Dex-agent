import copy
from unittest.mock import patch
import pytest
import torch
from torch import nn
from ptv3_shared_wam_v2.shared_actor import SharedQueryActor,INTERFACES
from ptv3_shared_wam_v2.robot_model import SharedWAMV2Robot,robot_loss
from ptv3_shared_wam_v2.model import SharedWAMV2
from ptv3_shared_wam.action_targets import GEOMETRY_LIFTED
from test_contracts import config,stats,batch,Geometry


def test_matched_heads_and_masked_payloads():
    heads=[]
    for arm in INTERFACES:
        torch.manual_seed(4);heads.append(SharedQueryActor(arm,48,16))
    assert len({sum(p.numel() for p in h.parameters()) for h in heads})==1
    for h in heads[1:]:
        for key,value in heads[0].state_dict().items():torch.testing.assert_close(value,h.state_dict()[key])
    h=heads[1];nn.init.normal_(h.head.weight,std=.02)
    state=torch.randn(2,87);tokens=torch.randn(2,7,48);mask=torch.ones(2,7,dtype=torch.bool);mask[:,-1]=False
    first=h(state,shared=tokens,shared_valid=mask)
    tokens[:,-1]=float('nan')
    torch.testing.assert_close(first,h(state,shared=tokens,shared_valid=mask))
    tokens[:,0]+=.5
    assert not torch.allclose(first,h(state,shared=tokens,shared_valid=mask))


@pytest.mark.parametrize('arm',INTERFACES)
def test_action_reaches_shared_once_without_gt_anchor(arm):
    c=config();c.update(shared_width=48,shared_heads=4,action_interface=arm,anchor_weight=0.)
    f=SharedWAMV2(stats(),c,Geometry(c['point_width'])).eval()
    with patch('ptv3_shared_wam_v2.robot_model.SharedWAMV2',return_value=f):
        model=SharedWAMV2Robot(stats(),c,GEOMETRY_LIFTED).eval()
    # As in the first post-update iteration, the readout must be nonzero for action gradients.
    nn.init.normal_(model.retargeter.head.weight,std=.01)
    b=batch();b.update(has_action=torch.ones(2,dtype=torch.bool),native_state=torch.randn(2,87),
                       native_tau=torch.full((2,24,33),float('nan')),action_target=torch.randn(2,8,15))
    with patch.object(f,'encode',wraps=f.encode) as encode:
        p=model(b)
        assert encode.call_count==1 and 'action_anchor' not in p
    loss,terms=robot_loss(p,b,c,model)
    assert torch.isfinite(loss) and terms['anchor']==0
    groups=[f.geometry,f.interaction,f.goal,f.hand_encoder]
    for group in groups:
        gradients=torch.autograd.grad(terms['action'],list(group.parameters()),retain_graph=True,allow_unused=True)
        assert sum(float(g.abs().sum()) for g in gradients if g is not None)>0
    b['native_tau'].zero_()
    torch.testing.assert_close(p['action'],model(b)['action'])
    if arm=='shared':
        # Direct shared actor does not pass action supervision through the forecast readout.
        grads=torch.autograd.grad(terms['action'],list(f.hand_future.parameters()),allow_unused=True)
        assert all(g is None for g in grads)


def test_latent_cannot_silently_use_gt_anchor():
    c=config();c.update(shared_width=48,action_interface='shared',anchor_weight=1.)
    with patch('ptv3_shared_wam_v2.robot_model.SharedWAMV2',return_value=nn.Identity()):
        with pytest.raises(ValueError,match='anchor_weight=0'):SharedWAMV2Robot(stats(),c,GEOMETRY_LIFTED)
