"""Native anchored wrist with continuously observed expert finger feedback."""
from .executable_contact_options import hold_action


def wrist_anchored_action(expert_action,anchor,position,offset,scale):
    action=expert_action.clone()
    action[:,:6]=hold_action(anchor,position,offset,scale)[:,:6]
    return action
