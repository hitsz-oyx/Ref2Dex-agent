"""Closed-loop execution of a decoded hand trajectory.

The high-level policy owns a future hand trajectory.  This module owns the
interface between that trajectory and the native position-drive command:
select a short receding-horizon goal, add the nominal wrist velocity
feedforward, apply a bounded residual, and recompute the command from the
current measured joint state on every control step.

It deliberately contains no policy network and no simulator dependency.  A
native runner can therefore use the same contract for a GT plan, a generated
plan, or a learned residual policy.
"""
import torch

from consequence_evaluator.reference_tracking import (
    ACTIVE,
    apply_coupling,
    native_action,
    wrist_feedforward,
)


SCHEMA = "ref2dex.goal-conditioned-executor.v1"
HORIZON = 24
EXECUTE_STEPS = 8
DOF = 18
RESIDUAL_DIM = len(ACTIVE)


def _require_tensor(name, value):
    if not isinstance(value, torch.Tensor):
        raise TypeError(name + " must be a torch.Tensor")
    if not torch.is_floating_point(value):
        raise TypeError(name + " must be floating point")
    if not torch.isfinite(value).all():
        raise ValueError(name + " must be finite")


def validate_plan(reference_q, reference_velocity, horizon=HORIZON):
    """Validate a ``[batch, horizon + 1, 18]`` target plan.

    Index zero is the live-state anchor.  Control consumes indices 1 through
    ``horizon``; keeping the anchor in the contract prevents callers from
    accidentally treating the first future target as the current state.
    """
    _require_tensor("reference_q", reference_q)
    _require_tensor("reference_velocity", reference_velocity)
    if reference_q.ndim != 3 or reference_velocity.shape != reference_q.shape:
        raise ValueError("reference plan shape mismatch")
    if reference_q.shape[1:] != (horizon + 1, DOF):
        raise ValueError("reference plan must have horizon+1 x 18 entries")
    return reference_q, reference_velocity


def receding_goal(reference_q, reference_velocity, slot, execute_steps=EXECUTE_STEPS):
    """Return the next local goal consumed at ``slot`` of a short chunk."""
    validate_plan(reference_q, reference_velocity)
    if not isinstance(execute_steps, int) or execute_steps < 1 or execute_steps > HORIZON:
        raise ValueError("execute_steps must be in 1..24")
    if not isinstance(slot, int) or not 0 <= slot < execute_steps:
        raise ValueError("receding slot must be an integer in the executed prefix")
    return reference_q[:, slot + 1], reference_velocity[:, slot + 1]


def _match(value, reference):
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
    else:
        value = value.to(device=reference.device, dtype=reference.dtype)
    if not torch.isfinite(value).all():
        raise ValueError("executor constants must be finite")
    return value


class GoalConditionedExecutor:
    """Stateless native command adapter with measured-state feedback.

    ``residual_latent`` is an unconstrained policy output.  ``tanh`` and the
    declared ``residual_limits`` turn it into a bounded target displacement.
    The caller must pass the current measured ``q`` for every invocation.
    """

    def __init__(self, offset, scale, damping_over_stiffness,
                 residual_limits=(.06, .06, .06, .6, .6, .6,
                                  .5, .5, .5, .5, .35, .35),
                 execute_steps=EXECUTE_STEPS):
        if not isinstance(execute_steps, int) or not 1 <= execute_steps <= HORIZON:
            raise ValueError("execute_steps must be in 1..24")
        self.offset = offset
        self.scale = scale
        self.damping_over_stiffness = damping_over_stiffness
        self.residual_limits = residual_limits
        self.execute_steps = execute_steps

    @staticmethod
    def _check_step(current_q, reference_q, reference_velocity, residual_latent):
        for name, value in (("current_q", current_q), ("reference_q", reference_q),
                            ("reference_velocity", reference_velocity),
                            ("residual_latent", residual_latent)):
            _require_tensor(name, value)
        if current_q.ndim != 2 or current_q.shape[-1] != DOF:
            raise ValueError("current q must have shape [batch, 18]")
        if reference_q.shape != current_q.shape or reference_velocity.shape != current_q.shape:
            raise ValueError("single-step goal/current shape mismatch")
        if residual_latent.shape != (len(current_q), RESIDUAL_DIM):
            raise ValueError("residual must have one value per active coordinate")

    def target(self, reference_q, reference_velocity, residual_latent):
        """Build a bounded target from a local reference and live-step velocity."""
        _require_tensor("reference_q", reference_q)
        _require_tensor("reference_velocity", reference_velocity)
        _require_tensor("residual_latent", residual_latent)
        if reference_q.ndim != 2 or reference_q.shape[-1] != DOF:
            raise ValueError("reference q must have shape [batch, 18]")
        if reference_velocity.shape != reference_q.shape:
            raise ValueError("reference velocity shape mismatch")
        if residual_latent.shape != (len(reference_q), RESIDUAL_DIM):
            raise ValueError("residual shape mismatch")
        ratio = _match(self.damping_over_stiffness, reference_q)
        limits = _match(self.residual_limits, reference_q)
        if ratio.shape != (6,) or limits.shape != (RESIDUAL_DIM,):
            raise ValueError("native feedback constants have invalid shape")
        target = wrist_feedforward(reference_q, reference_velocity, ratio)
        target = target.clone()
        target[:, list(ACTIVE)] += torch.tanh(residual_latent) * limits
        return apply_coupling(target)

    def command_from_target(self, current_q, target):
        """Convert a target into requested/applied native action tensors."""
        _require_tensor("current_q", current_q)
        _require_tensor("target", target)
        if current_q.shape != target.shape or current_q.ndim != 2 or current_q.shape[-1] != DOF:
            raise ValueError("current q/target must both have shape [batch, 18]")
        offset = _match(self.offset, current_q)
        scale = _match(self.scale, current_q)
        if offset.shape != (DOF,) or scale.shape != (DOF,) or (scale <= 0).any():
            raise ValueError("native offset/scale must be positive 18-D vectors")
        intended = native_action(target, current_q, offset, scale)
        command = intended.clamp(-1, 1)
        return dict(target=target, intended=intended, command=command,
                    clipped=(intended - command).abs().amax(-1) > 1e-6)

    def step(self, current_q, reference_q, reference_velocity, residual_latent):
        """Recompute a closed-loop native command from the current measured q."""
        self._check_step(current_q, reference_q, reference_velocity, residual_latent)
        target = self.target(reference_q, reference_velocity, residual_latent)
        return self.command_from_target(current_q, target)

    def receding_step(self, current_q, reference_q, reference_velocity,
                      residual_latent, slot):
        """Select one target from a 24-step plan and execute it immediately."""
        next_q, next_velocity = receding_goal(reference_q, reference_velocity, slot, self.execute_steps)
        return self.step(current_q, next_q, next_velocity, residual_latent)
