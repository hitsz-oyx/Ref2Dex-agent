"""New-plan contact-presence proxy; historical0.1N contracts stay immutable."""
import torch


def weight_normalized_contacts(hand_forces,object_forces,mass,gravity_magnitude=9.81):
    weight=torch.as_tensor(mass,device=object_forces.device,dtype=object_forces.dtype)*gravity_magnitude
    if weight.ndim==0:weight=weight.expand(len(object_forces))
    if weight.shape!=(len(object_forces),) or not torch.isfinite(weight).all() or (weight<=0).any():raise ValueError('positive per-object weight required')
    if not torch.isfinite(hand_forces).all() or not torch.isfinite(object_forces).all():raise ValueError('finite raw forces required')
    normalized_hand=hand_forces.norm(dim=-1).amax(-1)/weight
    normalized_object=object_forces.norm(dim=-1)/weight
    # Presence of >one tenth of the static support load; remains a net-force
    # proxy, not identified hand-object pairs. Geometry is a separate contract.
    ratio=torch.stack((normalized_hand,normalized_object),-1)
    return ratio>.1,ratio
