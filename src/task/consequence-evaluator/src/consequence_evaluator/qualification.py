"""Recompute the parent readiness gate from native first-episode physics."""
import numpy as np

from .collection import pose_matrix
from .supervision import expert_anchor, physical_trace


def qualify_transitions(payload, episodes, count=64):
    """Ignore later reset episodes and reject incomplete or shifted clocks.

    Native transition action exports are diagnostic only: the vendor may mutate
    them during PD conversion. They are never promoted to evaluator examples.
    """
    names = ('object_state', 'next_object_state', 'hand_contact',
             'object_contact', 'done', 'progress', 'data_id', 'action')
    arrays = {}
    for name in names:
        value = payload[name]
        if hasattr(value, 'detach'):
            value = value.detach().cpu().numpy()
        arrays[name] = np.asarray(value)
    rows = len(arrays['done'])
    if count < 1 or rows % count or any(len(v) != rows for v in arrays.values()):
        raise ValueError('native step-major transition shape mismatch')
    for name in ('hand_contact', 'object_contact', 'done'):
        if arrays[name].dtype != np.bool_ or arrays[name].shape != (rows, 1):
            raise ValueError('native physical flags must be boolean [rows,1]')
    if (arrays['object_state'].shape != (rows, 13)
            or arrays['next_object_state'].shape != (rows, 13)
            or arrays['action'].shape != (rows, 18)
            or arrays['progress'].shape != (rows, 1)
            or arrays['data_id'].shape != (rows, 1)):
        raise ValueError('native physical transition fields have wrong shapes')
    if not all(np.isfinite(v).all() for v in arrays.values()):
        raise ValueError('nonfinite native qualification trace')
    records = {r['env_id']: r for r in episodes}
    if len(episodes) != count or set(records) != set(range(count)):
        raise ValueError('exactly one recorded first episode per environment required')
    arrays = {k: v.reshape(-1, count, *v.shape[1:]) for k, v in arrays.items()}
    results = []
    for env in range(count):
        row = records[env]
        done = arrays['done'][:, env, 0]
        ends = np.flatnonzero(done)
        if not len(ends):
            raise ValueError('incomplete first episode')
        length = int(ends[0]) + 1
        progress = arrays['progress'][:length, env, 0]
        motion = arrays['data_id'][:length, env, 0]
        before = arrays['object_state'][:length, env]
        after = arrays['next_object_state'][:length, env]
        first_displacement = float(np.linalg.norm(after[0,:3]-before[0,:3]))
        if first_displacement > .1:
            raise ValueError('unphysical first-step reset displacement exceeds10cm')
        if (row['start_frame'] != 0 or row['steps'] != length
                or not np.array_equal(progress, np.arange(length))
                or not (motion == row['motion_id']).all()
                or not np.allclose(before[1:], after[:-1], atol=1e-5, rtol=0)):
            raise ValueError('full frame0 continuous first-episode alignment required')
        poses = pose_matrix(np.concatenate((before[:1], after)))
        contact = (arrays['hand_contact'][:length, env, 0]
                   & arrays['object_contact'][:length, env, 0])
        diagnostics = dict(contact=np.r_[False, contact],
                           contact_valid=np.r_[False, np.ones(length, dtype=bool)],
                           initial_height=float(poses[0, 2, 3]))
        packet = dict(action=arrays['action'][:length, env], object_pose=poses)
        # Preserve the original operational gate. These traces lack measured
        # geometry and are explicitly forbidden as evaluator training data.
        trace = physical_trace(packet, diagnostics, require_geometry=False)
        quality, _, _, completion = expert_anchor(
            trace, dict(assigned_phase='clean', perturbation_tick=-1))
        results.append(dict(env_id=env, motion_id=row['motion_id'], steps=length,
                            first_step_displacement_m=first_displacement,
                            qualified=quality == 'expert_success',
                            completion_tick=completion,
                            dropped_after_hold=bool(completion is not None
                                and trace['drop'][completion+1:].any()),
                            maximum_held_frames=int(trace['held_run'].max()),
                            maximum_lift_m=float(trace['height'].max()),
                            contact_fraction=float(contact.mean())))
    successes = sum(r['qualified'] for r in results)
    return dict(status='PROMISING' if count == 64 and successes >= 8 else 'UNCLEAR',
                data_readiness_pass=count == 64 and successes >= 8,
                episodes=count, qualified_episodes=successes, required_successes=8,
                criterion='>=3cm elevated native contact proxy for45consecutive frames; no later drop',
                drop_criterion='<2cm height or>=6consecutive lost-contact frames after qualification',
                training_allowed=False, per_episode=results,
                limitation='operational readiness gate; native net-force proxy; not formal validation or evaluator data')
