"""Current-object-frame actuation/effect features and explicit passive controls."""
from __future__ import annotations
import torch

DT=1/30


def local_vectors(vectors,rotation):
    """Row world vectors multiplied by local-to-world R give local vectors."""
    return torch.einsum('b...j,bjk->b...k',vectors,rotation)


def state_features(state,tips_world,rotation):
    velocities=state[:,18:36].clone()
    velocities[:,:3]=local_vectors(velocities[:,:3],rotation)
    object_velocity=local_vectors(state[:,43:49].reshape(-1,2,3),rotation).flatten(1)
    relative=local_vectors(tips_world-state[:,None,36:39],rotation).flatten(1)
    gravity_axis=-rotation[:,2,:]
    return torch.cat((state[:,3:18],velocities,object_velocity,gravity_axis,
                      state[:,38:39],relative,state[:,49:51]),-1)


def motion_features(current_world,next_world,rotation):
    # Only future HAND positions enter the explicitly named oracle variant.
    # The coordinate frame always uses the current object orientation.
    return local_vectors(next_world-current_world,rotation).flatten(1)


def position_residual_target(state,next_state,rotation):
    residual=next_state[:,36:39]-state[:,36:39]-state[:,43:46]*DT
    return local_vectors(residual,rotation)


def geometric_transport(flow,center_distances,object_local_velocity):
    """Approximate no-slip transport; passive fallback if centers aren't near."""
    weights=torch.exp(-(center_distances/.02).square())*(center_distances<=.02)
    total=weights.sum(-1,keepdim=True)
    weighted=(flow.reshape(-1,6,3)*weights[...,None]).sum(1)/total.clamp_min(1e-8)
    residual=weighted-object_local_velocity*DT
    return torch.where(total>0,residual,torch.zeros_like(residual))
