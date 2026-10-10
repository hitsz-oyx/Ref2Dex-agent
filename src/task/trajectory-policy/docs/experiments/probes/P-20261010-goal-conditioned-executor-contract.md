---
schema: ref2dex.probe.v2
probe_id: P-20261010-goal-conditioned-executor-contract
experiment_id: P-20261010-goal-conditioned-executor-contract
date: 2026-10-10
task: trajectory-policy
branch: main
git_commit: ab7d270
claim_id: C3
hypothesis_family: HF-trajectory-policy-execution-interface
probe_index_in_family: 1
seed_pool: engineering
seeds: []
decision_changed_if_positive: use the shared goal-conditioned executor for the next GT-vs-generated-tau physics comparison
decision_changed_if_negative: keep the existing direct execution path and diagnose the contract mismatch before physics
status: PROMISING
run_id: goal-conditioned-executor-contract-20261010-r1
---

# Can the trajectory-policy executor expose a single closed-loop goal contract?

## Motivation / Decision Note

The ref1_2 route changes the execution boundary from direct `tau -> native
action` regression to `tau -> q_des/velocity_des -> feedback + bounded
residual -> native action`. Existing trajectory-policy code already decodes a
24-step `q`/velocity plan and uses the same reference-tracker equations, but
the conversion is duplicated in each native runner. A duplicated conversion
can let GT, generated, and PPO paths silently use different semantics.

This is a bounded engineering Decision probe before any new training. It asks
whether one stateless executor can encode the intended contract and reproduce
the existing requested command exactly. Positive evidence permits the next
decision experiment to compare GT and generated plans through one executor;
negative evidence blocks physics and requires fixing the contract first.

## Fixed contract

`reference_q` and `reference_velocity` are `[batch, 25, 18]`; index zero is
the live-state anchor and only indices 1--8 are consumed before a replan.
Every control step supplies measured current `q`. Wrist target velocity is
added through the existing damping/stiffness ratio. An unconstrained
12-coordinate residual is transformed by `tanh` and fixed limits, coupled to
the native 18-DOF target, converted to the native action, then clamped only at
the final dispatch boundary. No future robot q, object pose, phase, clock,
tactile value, or teacher command enters this adapter.

The module is pure PyTorch and has no simulator or policy dependency. Existing
PPO/history native runners are migrated to call it; the latent policy and
trajectory decoder remain unchanged. The tests compare its output against the
previous reference-tracking equations and verify that changing live q changes
the command while the target plan stays fixed.

## Stop and evidence

Run the focused CPU contract tests plus the existing trajectory-policy decoder
tests. Any shape/finite/active-coordinate mismatch, non-bitwise-equivalent
requested command, or mutation of the user-owned untracked files stops the
probe. This is engineering evidence only; it does not establish a grasp or a
generated-tau result. A later physics Probe must use the shared module and
report GT and generated plans through the same path.

## Results

The shared implementation is `trajectory_policy.goal_executor` and the four
trajectory-policy native entry points now call it. The focused task suite
passes `18 passed`; all five changed Python entry points and the executor pass
`py_compile`, and `git diff --check` is clean.

The existing complete PPO execution wave
`outputs/trajectory-policy/trajectory-ppo-execution-20261010-r1/` was replayed
one control at a time through the new executor. Across 8,672 rows, the
requested native command differs from the recorded command by at most
`2.3841858e-7`; the pre-clamp requested action differs by at most
`4.7683716e-7`. The replayed clipping count is 36, matching the saved wave's
native execution semantics. This verifies an interface migration, not a new
grasp or policy result.

The result is `PROMISING` for the bounded engineering question. The next
physics Probe should route both GT and generated tau plans through this module
before any new training or residual-policy sweep. The output manifest and
replay metrics are in
`outputs/trajectory-policy/goal-conditioned-executor-contract-20261010-r1/`.
