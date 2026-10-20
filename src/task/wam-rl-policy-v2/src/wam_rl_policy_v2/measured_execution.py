"""Shared causal boundaries for measured offline goals and online execution."""
import numpy as np

from .teacher_quality import table_support

GOAL_CONTRACT = 'ref2dex.measured-task-controller-goal.v1'
TAU_LAYOUT_SCHEMA = 'ref2dex.measured-scheduled-tau-layout.v2'


def online_support(snapshot, index, pose, points, table_bounds):
    """Use this execution's object force/table pose, with the audit's CAD bounds."""
    if 'object_contact_force' not in snapshot or 'table_pose' not in snapshot:
        raise ValueError('measured support requires live object contact force and table pose')
    if not bool(np.asarray(snapshot['contact_force_valid'])[index]):
        return False
    force = np.asarray(snapshot['object_contact_force'], dtype=np.float64)[index]
    table = np.asarray(snapshot['table_pose'], dtype=np.float64)[index]
    bounds = np.asarray(table_bounds, dtype=np.float64)
    if (force.shape != (3,) or table.shape != (7,) or bounds.shape != (2, 3)
            or not np.isfinite(np.r_[force, table, bounds.ravel()]).all()):
        raise ValueError('finite measured support fields required')
    support, _, _ = table_support(np.asarray(pose)[None], points, table, bounds, force[None])
    return bool(support[0])


def environment_geometry(runtime, index, object_code):
    """Runtime geometry follows environment order, including repeated objects."""
    if runtime.env_objects[index]['object_code'] != object_code:
        raise ValueError('runtime geometry does not match the current environment object')
    return runtime.geometry[index]


def run_single_episode_batch(connection, action, snapshot, controls, *, observe, on_step,
                             is_terminal, send, receive, expand_action):
    """End a bounded batch at its first task terminal; never step across reset.

    The last observation is still delivered to the collector so its terminal
    transition and the other environments' actual bootstrap are retained.
    A future multi-episode collector must explicitly reset before sampling anew.
    """
    completed = 0
    while completed < controls and not is_terminal():
        source_episodes = list(snapshot['episodes'])
        source_ticks = np.asarray(snapshot['tick']).copy()
        send(connection, dict(op='step', action=expand_action(action).tolist(), episodes=source_episodes))
        reply = receive(connection)
        if 'state' not in reply:
            raise ValueError('native step response lacks state')
        # Decode before checking ticks; packed transport keeps episode IDs plain.
        from .inspire_state_codec import unpack_state
        next_state = unpack_state(reply['state'])
        if (list(next_state['episodes']) != source_episodes
                or not np.array_equal(np.asarray(next_state['tick']), source_ticks + 1)):
            raise ValueError('native step crossed an episode or nonconsecutive clock')
        action, snapshot = observe(next_state)
        completed += 1
        on_step(completed)
    return completed
