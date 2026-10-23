"""Explicit consumer for fresh reset-certified data; legacy6k stays unchanged."""
import numpy as np


def reset_bound_sample(sequence, geometry, contact, tick, slot, source_id,
                       sample_id, instruction, expected_reset):
    from wam_rl_policy_v3.goal_channel import validate_instruction
    from ptv3_shared_wam_v5.data import sample_window
    validate_instruction(instruction)
    if instruction['reset'] != expected_reset or instruction['object_slot'] != slot:
        raise ValueError('goal instruction belongs to another reset/object')
    sample = sample_window(sequence, geometry, contact, tick, slot, source_id, sample_id)
    # Future observations remain prediction labels; task goal derives solely
    # from the pre-reset instruction and current observed object coordinate frame.
    pose = np.asarray(sequence['object_poses'][tick,slot],np.float64)
    relative = np.linalg.inv(pose) @ np.asarray(instruction['task_goal_pose'],np.float64)
    query = np.asarray(geometry['query'],np.float64)
    if not np.array_equal(query,np.asarray(instruction['sources']['query']['payload'],np.float64)):
        raise ValueError('goal query registry differs from reset-bound query IDs')
    delta = (query @ relative[:3,:3].T + relative[:3,3] - query).astype(np.float32)
    sample.update(goal_object_delta=delta,goal_object_points=query.astype(np.float32)+delta,
                  object_goal_valid=np.ones(32,bool))
    return sample
