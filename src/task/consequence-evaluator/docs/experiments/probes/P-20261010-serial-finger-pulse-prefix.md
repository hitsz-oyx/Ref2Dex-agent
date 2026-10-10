---
schema: ref2dex.probe.v2
probe_id: P-20261010-serial-finger-pulse-prefix
experiment_id: P-20261010-serial-finger-pulse-prefix
date: 2026-10-10
task: consequence-evaluator
branch: main
git_commit: b9a7a95
claim_id: C3
hypothesis_family: HF-hand-action-retarget
probe_index_in_family: 7
seed_pool: probe
seeds: [421]
decision_changed_if_positive: retain a narrow serial contact-stage paired-capture contract for future matched-state data collection
decision_changed_if_negative: close the serial paired-capture route and keep decoder, H-to-hand, PointWorld, selector, and native integration frozen
status: PROMISING
run_id: serial-finger-pulse-prefix-20261010-r1
---

# Can a deterministic serial prefix capture a commanded-finger contact perturbation?

## Motivation and decision note

The structured 96/64/64 packets contain commanded-finger variation but almost
no matched contact/hold states. The existing native runner has no fork or
restore API, so ordinary multi-environment rows cannot be treated as paired
counterfactuals. A cheap same-seed serial replay was bitwise identical in two
baseline runs; that makes one bounded diagnostic worth trying before closing
the data route.

The current decision is whether to preserve a *data-contract* for a later
contact-stage collection, not whether a finger pulse improves grasping. The
Probe therefore uses one serial world, one frozen actor/backend, and one
single-tick pulse at the observed contact-onset interval (tick 50). It does
not fit a decoder, score native success, or claim causal preload evidence.

## Frozen protocol

Run four fresh single-environment retarget rollouts with seed 421 and the same
execution input packet: two `structured-profile=zero` replays and one each
with a single active finger coordinate (index 6) set to `+0.08` and `-0.08`
for command tick 50 only. The runner saves the full commanded action, actor
action, structured residual, native active PD target (`task.real_pd_tar`),
state frame, and actor observation. State frame `t` is paired with command
`t`; post-pulse state is reported only as a diagnostic.

Engineering gates are: 543 finite state frames and 542 commands; exact
requested/applied action composition; zero clipping; identical seed, actor,
and frozen input hashes; bitwise-equal zero replay; and bitwise-equal control
and pulse prefixes through the intervention frame. A pulse comparison is
contract-valid only if the requested residual and captured action/PD target
match the registered value exactly.

The visible matched-state screen remains hand RMS <=3 mm, q RMS <=.03,
dq RMS <=.10, with the saved force-pair field retained as a diagnostic. This
pilot stops if the prefix/twin gates fail or if it cannot provide the command
contract; it does not use post-pulse divergence as a positive scientific
result.

## Stop conditions and boundary

Stop on incomplete/nonfinite packets, clipping, action-composition mismatch,
hash mismatch, or any prefix drift. Do not train R, H-to-tau, a decoder, or
run the pulse as a native behavior/success experiment. Even a valid serial
prefix is only `PROMISING` for future data collection; a later matched-state
pilot would still need at least 20 matched rows and 10 nontrivial command
differences per phase before expansion.

## Artifacts

Runner profile:
`src/task/consequence-evaluator/tools/run/run_hand_bridge_rollout.py`

Audit entry point:
`src/task/consequence-evaluator/tools/audit/audit_serial_finger_pulse.py`.

Outputs:
`outputs/consequence-evaluator/serial-finger-pulse-zero-20261010-r1/`,
`outputs/consequence-evaluator/serial-finger-pulse-zero-20261010-r2/`,
`outputs/consequence-evaluator/serial-finger-pulse-plus-20261010-r1/`,
`outputs/consequence-evaluator/serial-finger-pulse-minus-20261010-r1/`.

## Result

All four bounded rollouts completed on GPU2 in about 41 seconds each. The two
zero-residual replays were bitwise identical for every saved state, command,
actor observation, force proxy, and native active PD target. This is a valid
deterministic serial prefix/twin contract, not a hidden-PhysX state fork.

Both pulse branches also passed the engineering gates: 543 finite state
frames, 542 commands, zero clipping, exact action composition, bitwise-equal
prefix through tick 50, and exact requested/applied `0.08` or `-0.08` residual
on native finger coordinate 6. The corresponding active PD-target change was
about `0.064` in the saved `real_pd_tar` coordinate.

The post-pulse diagnostic differs by sign. The zero control has pair rates
`0.65 / 0 / 0` in onset/contact/hold windows; the `+0.08` branch has
`0.60 / 0.033 / 0`, while the `-0.08` branch has `0.875 / 0.208 / 0.033`.
These are one-seed, one-tick feedback trajectories and are not success,
preload, or causal estimates; the post-pulse hand/object/force changes are
explicitly diagnostic only.

The Probe is therefore `PROMISING` only for retaining the serial paired-capture
data contract. It does not reopen the execution decoder or native route. The
next collection would need multiple contact and hold states, with the
pre-registered visible matched-state thresholds and at least 20 matched rows
and 10 nontrivial command differences per phase before any decoder fit or
scientific effect screen.
