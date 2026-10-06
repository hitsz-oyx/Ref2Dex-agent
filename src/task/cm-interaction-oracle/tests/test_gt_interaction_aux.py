from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from gt_interaction_aux import interaction_target, rollout_targets, InteractionDecoder


def test_eighth_endpoint_terminal_tail_and_action_alignment():
    before=torch.zeros(12,3,15); before[...,6]=1
    after=before.clone(); after[...,0]=torch.arange(1,13)[:,None]*.003
    actions=torch.arange(12)[:,None,None].expand(12,3,18).float()/12
    dones=torch.zeros(12,3,dtype=torch.bool); dones[7,1]=True
    target,chunk,mask=rollout_targets(before,after,actions,dones)
    assert torch.allclose(target[0,:,0],torch.full((3,),.8))
    assert torch.equal(chunk[0,0,:,0],actions[:8,0,0])
    assert mask[:5,0].all() and not mask[5:].any()
    assert not mask[:5,1].any() # terminal within each complete window
    flipped=after.clone(); flipped[...,3:7]*=-1
    assert torch.equal(interaction_target(before,after),interaction_target(before,flipped))


def test_stopgradient_cuts_only_encoder_and_shuffled_chunk_changes_decoder():
    torch.manual_seed(42)
    decoder=InteractionDecoder(4)
    z=torch.randn(6,4,requires_grad=True)
    a=torch.randn(6,8,18)
    prediction=decoder(z,a)
    assert torch.autograd.grad(prediction.square().mean(),z,retain_graph=True)[0].norm()>0
    loss=decoder(z,a,detach=True).square().mean()
    assert torch.autograd.grad(loss,z,retain_graph=True,allow_unused=True)[0] is None
    loss.backward()
    assert decoder.net[0].weight.grad.norm()>0
    assert not torch.equal(prediction,decoder(z,a.roll(1,0)))
