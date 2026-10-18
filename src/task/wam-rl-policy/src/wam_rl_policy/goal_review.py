"""Geometry for inspecting scheduled goals and real WAM predictions."""
import numpy as np

from .goal_scheduler import interpolate_pose, rigid_goal_delta, transform_points

VARIANTS = ('较短前视', '当前调度', '较长前视')


def review_goals(scheduler, actual, decision):
    """Compare half/double path lead without bypassing the pending milestone.

    The selected goal is copied exactly. Recovery connectors retain their real
    meaning: shortening stays on the connector; longer requests are capped by
    the original reference destination and the same geometric budgets.
    """
    selected = decision['goal_pose'].copy()
    if decision['mode'] in ('COMPLETE', 'TIMEOUT'):
        return np.stack([selected.copy() for _ in VARIANTS])
    local = decision['projected_index']
    target_arc = float(np.interp(decision['target_reference_index'],
                                 np.arange(len(scheduler.arc)), scheduler.arc))
    if not decision['target_on_reference']:
        destination, _ = scheduler._at_arc(target_arc, 0, len(scheduler.poses)-1)
        return np.stack([interpolate_pose(actual, selected, .5), selected,
                         scheduler._bounded(actual, destination)])
    last = int(scheduler.milestones[min(scheduler.pointer+1, len(scheduler.milestones)-1)])
    lead = max(0., target_arc-float(scheduler.arc[local]))
    goals = []
    for factor in (.5, 1., 2.):
        if factor == 1.:
            goals.append(selected)
            continue
        desired = min(float(scheduler.arc[last]), float(scheduler.arc[local])+factor*lead)
        goal, _ = scheduler._at_arc(desired, local, last)
        if not scheduler._within_budget(actual, goal):
            low, high = float(scheduler.arc[local]), desired
            for _ in range(28):
                middle = (low+high)/2
                candidate, _ = scheduler._at_arc(middle, local, last)
                if scheduler._within_budget(actual, candidate):
                    low = middle
                else:
                    high = middle
            goal, _ = scheduler._at_arc(low, local, last)
        goals.append(goal)
    return np.stack(goals)


def prediction_in_world(pose, hand, flow, trajectory, scene):
    """Anchor displacements -> world positions, explicitly including t=0."""
    trajectory, scene = np.asarray(trajectory), np.asarray(scene)
    if trajectory.shape != (24, 2, 11, 3) or scene.shape != (1, 24, 32, 3):
        raise ValueError('one physical-unit v4 hand/scene prediction required')
    rotation = np.asarray(pose)[:3, :3]
    predicted_hand = np.asarray(hand)[None] + trajectory[:, 0] @ rotation.T
    current_flow = transform_points(pose, flow)
    predicted_scene = current_flow[None] + scene[0] @ rotation.T
    if not np.isfinite(predicted_hand).all() or not np.isfinite(predicted_scene).all():
        raise ValueError('finite world prediction required')
    return (np.concatenate([np.asarray(hand)[None], predicted_hand]),
            np.concatenate([current_flow[None], predicted_scene]))


def goals_as_delta(actual, goals, flow):
    return np.stack([rigid_goal_delta(actual, goal, flow) for goal in goals])
