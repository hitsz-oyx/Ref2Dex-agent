---
schema: ref2dex.probe.v2
probe_id: P-20261010-paired-triplet-contact-hold
experiment_id: P-20261010-paired-triplet-contact-hold
date: 2026-10-10
task: consequence-evaluator
branch: main
git_commit: pending
claim_id: C3
hypothesis_family: HF-hand-action-retarget
probe_index_in_family: 8
seed_pool: probe
seeds: [211, 212]
decision_changed_if_positive: retain a multi-state contact/hold paired-capture contract for a future commanded-finger decoder Probe
decision_changed_if_negative: close the contact/hold paired-data route and keep R, H-to-tau, PointWorld, selector, and native integration frozen
status: PLANNED
run_id: paired-triplet-contact-hold-20261010-r1
---

# Can triplet prefixes provide matched contact and hold command rows?

## Motivation and decision note

The preceding serial Probe established exact zero replay and one-tick action
injection, but it yielded only one state and the tick-50 pulse was an onset
diagnostic rather than a registered contact/hold row. Existing structured
rollouts have contact and hold coverage but no paired command perturbations.
This bounded follow-up asks only whether a deterministic triplet schedule can
join those two contracts: each group has a control, `+d`, and `-d` branch with
the same random structured prefix, then receives one active-finger pulse at a
fixed contact or hold tick.

A positive result keeps a data-collection route alive; it does not establish
preload causality, decoder identifiability, or native task benefit. A failed
prefix, clipping, or row-count gate closes this route without touching the
main R/H/PointWorld claims.

## Frozen protocol

Run two fresh GPU2 retarget rollouts with 90 environments each (30 triplets),
seed 211 for contact and seed 212 for hold, using the same frozen execution
input packet and `--structured-profile paired-triplet`. Within each triplet the
random residual schedule is duplicated across `control/plus/minus`; the
intervention overwrites independent finger coordinate 6 for one command tick
with `0/+0.08/-0.08`. Contact pulse ticks are 120--149; hold pulse ticks are
240--269. No post-reset jitter is used, so the triplet prefix itself remains
bitwise comparable; distinct groups receive distinct structured residual
prefixes.

The saved state frame `t` is paired with command `t`. A row is eligible only
when all three branches have bitwise-equal state/action prefixes through `t`,
the pre-action saved pair proxy agrees, visible hand/q/dq differences remain
within the registered thresholds, and the changed independent finger
coordinate has captured action difference > `.05`. The engineering screen
requires at least 20 matched rows and 10 nontrivial rows in each phase.
All arrays must be finite, 543/542 aligned, exactly composed, zero-clipped,
and include native active PD targets. Post-pulse state/force divergence is
reported only descriptively.

## Stop conditions and boundary

Stop on any prefix drift, action-composition mismatch, nonfinite data, clipping,
hash/input mismatch, or incomplete episode. If either phase misses the
matched/nontrivial row screen, mark the Probe `UNCLEAR` and do not train a
decoder. Do not call triplets a PhysX fork or counterfactual causal estimate;
the hidden solver state remains unavailable.

## Artifacts

Runner profile:
`src/task/consequence-evaluator/tools/run/run_hand_bridge_rollout.py`

Audit entry point:
`src/task/consequence-evaluator/tools/audit/audit_paired_triplet_pilot.py`.

Outputs:
`outputs/consequence-evaluator/paired-triplet-contact-20261010-r1/`,
`outputs/consequence-evaluator/paired-triplet-hold-20261010-r1/`,
`outputs/consequence-evaluator/paired-triplet-contact-audit-20261010-r1/`,
`outputs/consequence-evaluator/paired-triplet-hold-audit-20261010-r1/`.

## Result

Pending bounded contact/hold capture.
