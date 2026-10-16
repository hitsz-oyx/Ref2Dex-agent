"""Optional low-speed position-hold shaping; not a contact-success definition."""
import torch


def position_hold_bonus(object_pos,goal_pos,linear_velocity,initial_z):
    near=torch.linalg.vector_norm(object_pos-goal_pos,dim=-1)<=.05
    lifted=object_pos[:,2]-initial_z>=.1
    slow=torch.linalg.vector_norm(linear_velocity,dim=-1)<=.1
    return (near&lifted&slow).float()
