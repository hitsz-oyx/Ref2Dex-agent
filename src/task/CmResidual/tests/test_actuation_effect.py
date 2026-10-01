import torch

from src.task.CmResidual.actuation_effect import (
    geometric_transport,local_vectors,motion_features,position_residual_target,state_features,
)


def test_current_frame_rotates_world_vector_in_the_correct_direction():
    rotation=torch.tensor([[[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]]])
    world=torch.tensor([[1.,0.,0.]])
    assert torch.allclose(local_vectors(world,rotation),torch.tensor([[0.,-1.,0.]]))


def test_passive_translation_has_zero_residual():
    state=torch.zeros(2,55);state[:,36:39]=torch.tensor([1.,2.,3.]);state[:,43:46]=torch.tensor([.3,.6,.9])
    nxt=state.clone();nxt[:,36:39]+=state[:,43:46]/30
    rotation=torch.eye(3).expand(2,3,3)
    assert torch.allclose(position_residual_target(state,nxt,rotation),torch.zeros(2,3),atol=1e-7)


def test_motion_features_are_translation_invariant():
    current=torch.randn(2,6,3);future=current+torch.randn(2,6,3)*.01
    rotation=torch.eye(3).expand(2,3,3);offset=torch.tensor([[[1.,2.,3.]]])
    assert torch.allclose(motion_features(current,future,rotation),
                          motion_features(current+offset,future+offset,rotation),atol=4e-7)


def test_transport_is_passive_without_near_centers():
    residual=geometric_transport(torch.ones(2,18),torch.ones(2,6),torch.ones(2,3))
    assert torch.equal(residual,torch.zeros_like(residual))


def test_state_features_have_declared_geometry_slots():
    state=torch.zeros(2,55);tips=torch.zeros(2,6,3);rotation=torch.eye(3).expand(2,3,3)
    features=state_features(state,tips,rotation)
    assert features.shape==(2,63)
    assert torch.equal(features[:,39:42],torch.tensor([[0.,0.,-1.],[0.,0.,-1.]]))
