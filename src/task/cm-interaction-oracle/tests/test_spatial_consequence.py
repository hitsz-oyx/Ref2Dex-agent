"""Spatial integration contracts with synthetic CPU geometry."""
import sys
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(ROOT),str(ROOT/'src/task/cm-interaction-oracle/src')]
from spatial_consequence import SpatialConsequence, action_slots


def packet():
    torch.manual_seed(41)
    n=3
    points=torch.randn(1,8,3)*.003
    geometry=dict(obj_points=points,obj_normals=torch.nn.functional.normalize(points,dim=-1),
        points=points.expand(n,-1,-1).clone()+.001,
        normals=torch.nn.functional.normalize(points,dim=-1).expand(n,-1,-1).clone(),
        nominal_flow=torch.zeros(n,15,8,3),joint=torch.zeros(n,15,18))
    geometry['nominal_flow'][:,0,:,0]=.002
    geometry['nominal_flow'][:,1,:,0]=.004
    geometry['joint'][:,1,6]=.32
    return geometry,torch.ones(n,dtype=torch.long)


def test_state_and_continuous_joint_share_nominal_source_geometry():
    g,arms=packet();oh=torch.nn.functional.one_hot(arms,15)[:,1:].float()
    a,state_arms=action_slots('State',g,arms,oh)
    j,joint_arms=action_slots('Joint',g,arms,oh)
    assert not a.any() and not state_arms.any() and torch.equal(state_arms,joint_arms)
    assert (j[:,6]==1).all() and j.shape==(3,32)
    flow,flow_arms=action_slots('Flow',g,arms,oh)
    assert not flow.any() and torch.equal(flow_arms,arms)
    assert g['nominal_flow'][:,0].abs().sum()>0


def test_actual_spatial_action_changes_prediction_and_receives_gradient():
    g,arms=packet();model=SpatialConsequence(10,26)
    h=torch.zeros(3,10);a=torch.zeros(3,32);p=torch.zeros(3,26);ids=torch.arange(3)
    baseline=model(h,a,p,g,torch.zeros_like(arms),ids)
    factual=model(h,a,p,g,arms,ids)
    assert factual.shape==(3,26) and torch.isfinite(factual).all()
    assert (factual-baseline).abs().max()>1e-7
    factual.square().mean().backward()
    assert model.spatial.local_interaction.edge[0].weight.grad.abs().sum()>0


def test_empty_interaction_is_finite_and_hand_order_invariant():
    g,arms=packet();model=SpatialConsequence(10,26).eval()
    h=torch.zeros(3,10);a=torch.zeros(3,32);p=torch.zeros(3,26);ids=torch.arange(3)
    first=model(h,a,p,g,arms,ids)
    perm=torch.randperm(8)
    shuffled=dict(g,points=g['points'][:,perm],normals=g['normals'][:,perm],nominal_flow=g['nominal_flow'][:,:,perm])
    assert torch.allclose(first,model(h,a,p,shuffled,arms,ids),atol=1e-6)
    far=dict(g,points=g['points']+1)
    assert torch.isfinite(model(h,a,p,far,arms,ids)).all()
