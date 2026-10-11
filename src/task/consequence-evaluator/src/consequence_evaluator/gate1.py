"""Frozen early-contact GT-value Probe mechanics, independent of the simulator."""
import numpy as np

from .contracts import K
from .supervision import consecutive

QUERY_TICKS = (48, 56, 64, 72)
SEEDS = (282, 283, 284, 285)
CANDIDATES = ('baseline', 'finger_positive', 'finger_negative')
INDEPENDENT_FINGERS = (6, 8, 10, 12, 15)


def candidate_plan(candidate):
    if candidate not in range(len(CANDIDATES)):
        raise ValueError('unknown frozen candidate')
    plan = np.zeros((K, 18), np.float32)
    if candidate:
        # A decision-known constant residual for the24step scoring horizon.
        # Replanning explicitly discards the unexecuted suffix after8steps.
        plan[:, INDEPENDENT_FINGERS] = .2 if candidate == 1 else -.2
    return plan


def choose_candidate(values, epsilon=.01):
    values = np.asarray(values, np.float64)
    if values.shape != (3,) or not np.isfinite(values).all() or epsilon != .01:
        raise ValueError('three finite GT values and frozen epsilon required')
    best = int(np.argmax(values))
    # Preserve baseline unless a candidate beats it by the frozen deadzone.
    # Otherwise ties use the declared candidate order, never episode outcomes.
    return best if values[best] > values[0] + epsilon else 0


def episode_outcome(packet):
    """Recoverable stable grasp and controlled final placing, independent of Y.

    Sampled proximity/table-plane checks are weak physical proxies. Intermediate
    loss is counted but a later45frame stable hold resets the placing obligation.
    No phase embedding, reward or P value enters this outcome.
    """
    poses = np.asarray(packet['object_pose']); count = len(poses)
    gap = np.asarray(packet['surface_gap']); support = np.asarray(packet['support_gap'])
    footprint = np.asarray(packet['table_footprint']); velocity = np.asarray(packet['object_velocity'])
    if (count < 46 or poses.shape != (count, 4, 4) or gap.shape != (count,)
            or support.shape != (count,) or footprint.shape != (count,)
            or velocity.shape != (count, 6) or not np.isfinite(poses).all()
            or not np.isfinite(gap).all() or not np.isfinite(support).all()
            or not np.isfinite(velocity).all() or np.any(gap < 0)):
        raise ValueError('finite measured full episode required')
    valid = np.arange(count) > 0  # reset contact/proximity is not a held sample.
    height = poses[:, 2, 3] - poses[0, 2, 3]
    near = valid & (gap <= .01)
    supported = valid & footprint.astype(bool) & (np.abs(support) <= .02)
    held = near & ~supported & (height >= .03)
    held_run = consecutive(held); stable = np.flatnonzero(held_run >= 45)
    settled = supported & (np.linalg.norm(velocity[:, :3], axis=1) <= .05)
    settled &= np.linalg.norm(velocity[:, 3:], axis=1) <= .3
    unsupported_loss = valid & ~near & ~supported
    loss_run = consecutive(unsupported_loss)
    loss_events = (loss_run == 6)
    first_stable = int(stable[0]) if len(stable) else count
    intermediate_losses = int(loss_events[first_stable + 1:].sum())
    # Controlled placement after the latest stable grasp permits a recovered
    # intermediate failure, while rejecting a last unheld fall onto the table.
    last_stable = int(stable[-1]) if len(stable) else None
    unsafe = (loss_run >= 6) | (unsupported_loss & (velocity[:, 2] < -.25))
    controlled = last_stable is not None and not unsafe[last_stable + 1:].any()
    success = bool(controlled and consecutive(settled)[-1] >= 15)
    return dict(success=success, maximum_held_frames=int(held_run.max()),
        first_stable_tick=None if first_stable == count else first_stable,
        last_stable_tick=last_stable, terminal_settled_frames=int(consecutive(settled)[-1]),
        intermediate_loss_events=intermediate_losses, controlled_final_place=bool(controlled),
        terminal_height_m=float(height[-1]), metric='recoverable_hold45_controlled_place_settle15')


def paired_counts(pairs):
    if not pairs:
        raise ValueError('paired outcomes required')
    before = np.array([p['baseline']['success'] for p in pairs], bool)
    after = np.array([p['rolling']['success'] for p in pairs], bool)
    return dict(episodes=len(pairs), baseline_success=int(before.sum()),
        rolling_success=int(after.sum()), rescued=int((~before & after).sum()),
        harmed=int((before & ~after).sum()))


def legacy_batched_actor_action(player, observation):
    """Keep archived GPU inference on its verified64row path, with one sim env.

    The archived Torch2.0.1 runtime produced incorrect single-row GEMV results
    on this machine. Independent nonrecurrent copies preserve the controller
    mean; all fresh workers use the same64row model call/RNG consumption.
    """
    raw = observation['obs']
    if raw.ndim != 2 or raw.shape[0] != 1 or player.is_rnn:
        raise ValueError('single-environment nonrecurrent actor required')
    batch = dict(observation, obs=raw.repeat(64, 1))
    actions = player.get_action(batch, True)
    if actions.shape != (64, 18):
        raise ValueError('archived actor batch shape changed')
    return actions[:1].clone()


def legacy_group_actor_action(player, observation, copies=64):
    """Run a synchronous group through fixed-size deterministic actor blocks."""
    raw = observation['obs']
    if raw.ndim != 2 or raw.shape[0] < 2 or player.is_rnn:
        raise ValueError('synchronous actor group requires a nonrecurrent batch')
    if not isinstance(copies, int) or copies < 1:
        raise ValueError('actor copies must be a positive integer')
    envs = raw.shape[0]
    repeated = raw[:, None, :].expand(envs, copies, raw.shape[-1]).reshape(envs * copies, -1)
    actions = player.get_action(dict(observation, obs=repeated), True)
    if actions.shape != (envs * copies, 18):
        raise ValueError('archived actor group batch shape changed')
    return actions.reshape(envs, copies, 18)[:, 0].clone()
