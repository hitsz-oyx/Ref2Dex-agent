"""Reset distributions and measured new-grasp metrics for residual PPO."""
import numpy as np

UNIFORM_PHASES = (0, 32, 48, 62)
CONTACT_PHASES = (0, 48, 50, 52)


def curriculum_phases(name, envs):
    choices = dict(uniform=UNIFORM_PHASES, contact=CONTACT_PHASES)
    if name not in choices or not 1 <= envs <= 16:
        raise ValueError('known curriculum and 1:16 environments required')
    phases = choices[name]
    return tuple(phases[index % len(phases)] for index in range(envs))


def diagnostic_phases(phases):
    return tuple(sorted(set(UNIFORM_PHASES) | set(phases)))


def query_budget(updates, envs, phases, eval_every=0, rollout=96, diagnostic=96, evaluation=542):
    if updates < 1 or envs < 1 or eval_every < 0:
        raise ValueError('positive update/env budget and nonnegative eval interval required')
    groups = (len(diagnostic_phases(phases)) + envs - 1) // envs
    evaluations = 1 + ((updates - 1) // eval_every if eval_every else 0)
    return groups * diagnostic + updates * (rollout + 1) + evaluations * evaluation


def contact_hold_summary(packed, starts, task_start_height):
    """A hold inherited at reset is never counted as a newly acquired grasp."""
    lift = packed['obj'][:, :, 2, 3] - np.asarray(task_start_height)[None]
    supported = packed['table_footprint'] & (np.abs(packed['support_gap']) <= .02)
    held = (packed['surface_gap'] <= .01) & ~supported & (lift >= .03)
    rows = []
    for env, phase in enumerate(starts):
        longest = streak = 0
        for value in held[1:, env]:
            streak = streak + 1 if value else 0
            longest = max(longest, streak)
        contact = np.flatnonzero(packed['surface_gap'][:, env] <= .03)
        holding = np.flatnonzero(held[:, env])
        initial = bool(held[0, env])
        rows.append(dict(env=env, phase=int(phase), initially_held=initial,
                         initially_near=bool(packed['surface_gap'][0, env] <= .03),
                         maximum_held_frames=longest, held_fraction=float(held[1:, env].mean()),
                         new_hold45=bool(not initial and longest >= 45),
                         first_contact_tick=int(contact[0]) if len(contact) else None,
                         first_held_tick=int(holding[0]) if len(holding) else None,
                         max_lift_m=float(lift[:, env].max()), terminal_lift_m=float(lift[-1, env])))
    return rows
