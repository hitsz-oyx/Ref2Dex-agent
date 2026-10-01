"""Keep lift translation and feedback fingers, stabilize wrist rotation only."""
from .executable_contact_options import hold_action


def orientation_anchored_action(expert_action,anchor,position,offset,scale):
    action=expert_action.clone()
    action[:,3:6]=hold_action(anchor,position,offset,scale)[:,3:6]
    return action
