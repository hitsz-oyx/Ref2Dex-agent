"""Owner-admitted deletion of goal CONDITIONING, never supervision or task intent."""
import numpy as np
import torch

UNKNOWN_GOAL_POLICY = 'v6.no-goal.no-event.unknown-invalid.frame30.v1'
GOAL_KEYS = ('goal_object_delta', 'goal_object_points', 'object_goal_valid')


def validate_goal_input_policy(policy):
    if policy not in (None, UNKNOWN_GOAL_POLICY):
        raise ValueError('unsupported goal input policy: '+str(policy))
    return policy


def apply_goal_input_policy(batch, policy):
    """Same exact zero / false constructor for reader and runtime boundaries."""
    validate_goal_input_policy(policy)
    if policy is None:
        return batch
    result = dict(batch)
    for key in GOAL_KEYS:
        value = batch[key]
        result[key] = torch.zeros_like(value) if torch.is_tensor(value) else np.zeros_like(value)
    return result


def assert_unknown_inputs(batch):
    for key in GOAL_KEYS:
        value = batch[key]
        if torch.count_nonzero(value).item():
            raise AssertionError('unknown goal requires canonical zero / false: '+key)
    if batch['object_goal_valid'].dtype != torch.bool:
        raise AssertionError('unknown goal validity must be bool')
    if any(torch.count_nonzero(batch[k]).item() for k in ('event_type', 'event_delta_steps', 'event_valid')):
        raise AssertionError('no-goal/no-event requires unknown event inputs')
