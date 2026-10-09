---
schema: ref2dex.probe.v2
probe_id: P-20261010-finger-command-coverage
experiment_id: P-20261010-finger-command-coverage
date: 2026-10-10
task: consequence-evaluator
branch: main
git_commit: pending-clean-commit
claim_id: C3
hypothesis_family: HF-hand-action-retarget
probe_index_in_family: 6
seed_pool: probe
seeds: [412, 413, 414]
decision_changed_if_positive: retain the structured packets as a basis for a narrow commanded-finger/contact decoder design
decision_changed_if_negative: keep native R and H-to-hand integration frozen and require new contact-stage data before decoder work
status: PLANNED
run_id: finger-command-coverage-20261010-r1
---

# Do the structured rollouts identify commanded finger/contact labels?

## Motivation and decision note

The ref7_2 execution gate and the fixed-wrist attribution audit failed, while
the contact-proxy decoder Spike was only a 1.19% offline improvement. The
remaining unclosed question is narrower: do the existing structured rollouts
contain enough variation in the *commanded* finger PD targets to justify a
future contact-stage decoder, or are the labels effectively ambiguous under the
available geometry and state? This is a read-only audit; it does not train a
new retargeter and does not launch native execution.

A positive result would preserve the packets for a carefully specified
commanded-finger follow-up. A negative or unclear result keeps R, H-to-hand,
PointWorld, selector, and native integration frozen and points to a new
contact-stage collection contract instead.

## Frozen protocol

The audit reads the history-preserving retarget packets from the 96/64/64
episode split (seeds 412/413/414). It verifies that the saved action is the
actual clipped `actor_action + structured_residual`, then reports action and
residual variation separately for approach (ticks 0--119), contact
(120--239), and hold (240--541). It also reports the native force-pair proxy
rate and its agreement with the saved `pair` field.

For a state-ambiguity screen, current hand points are transformed into the
current object frame. At each tick, an episode pair is considered near when

```
hand RMS <= 3 mm, q RMS <= 0.03 rad, dq RMS <= 0.10 rad/s,
and the saved pair proxy agrees.
```

The label difference is the RMS of the 12 finger action coordinates; a
difference above `0.05` normalized action units is called nontrivial. The
pre-registered ambiguity screen requires at least 50 near pairs in each of
contact and hold and at least 10 nontrivial pairs in each phase. This threshold
is an engineering data-coverage screen, not evidence that the command is
physically identifiable.

## Stop conditions and boundary

Stop on incomplete episodes, nonfinite arrays, action-composition mismatch,
split/actor hash mismatch, or a pair/force inconsistency. Do not fit a decoder,
add future labels, alter thresholds after seeing results, or run a native
behavior screen from this audit.

## Artifacts

Audit entry point:
`src/task/consequence-evaluator/tools/audit/audit_finger_command_coverage.py`.
Output:
`outputs/consequence-evaluator/finger-command-coverage-20261010-r1/`.

## Result

Pending read-only audit.
