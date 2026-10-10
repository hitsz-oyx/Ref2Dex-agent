---
schema: ref2dex.probe.v2
probe_id: P-20261011-tau-residual-basin
experiment_id: P-20261011-tau-residual-basin
date: 2026-10-11
task: consequence-evaluator
branch: main
git_commit: pending
claim_id: C3
hypothesis_family: HF-reference-tracking-control
probe_index_in_family: 5
seed_pool: probe
seeds: [286]
decision_changed_if_positive: retain a bounded thumb residual basin and use it as the low-level adapter for the ref2 goal route
decision_changed_if_negative: stop fixed thumb-offset diagnostics and preserve the retained tail canonicalization candidate without more PPO
status: PLANNED
run_id: ref2-tau-residual-basin-20261011-r1
---

# Fixed residual-basin diagnostic around the retained tau executor

## Motivation / Decision Note

The retained tau-only tail canonicalization candidate passed the unchanged
majority gate (`12/16` long-held terminal, clipping `6/8672`), while its single
finger-only PPO continuation regressed to `5/16`. The remaining four drops are
not explained by a current command-chain defect. This Decision Probe asks the
cheapest next question: does a small, fixed thumb residual direction preserve a
feasible contact basin around the successful bounded-residual controller?

It uses the same REGRIND-style contract already present in the tau route:
GT hand-derived geometry as the nominal reference, live measured state on every
control, and a bounded residual. The perturbation is added to the frozen policy's
12-D latent output only after policy inference. It is a diagnostic intervention,
not an input, a learned parameter, a contact label, a phase/clock signal, or a
replacement for the goal-conditioned policy contract. No outcome is used to
select a row or tune a direction.

## Fixed protocol

Use the retained checkpoint
`outputs/consequence-evaluator/ref7_4-tau-finger-fit-20261010-r3/final.pt`
with the verified tau geometry, CPU-source reference, native Inspire controller,
future-only wrist feedforward, and all live q/dq/hand/object/velocity inputs
unchanged. Randomize 16 physical rows into four separate-live roles, four rows
each:

| Role | Fixed latent delta |
| --- | --- |
| `baseline` | zero |
| `thumb_pitch_plus` | residual coordinate 11, `+0.05` |
| `thumb_pitch_minus` | residual coordinate 11, `-0.05` |
| `thumb_yaw_plus` | residual coordinate 10, `+0.05` |

The values are chosen before execution and are not fit from this rollout. The
run records raw intended/applied commands, native PD targets, clipping, full
physical outcomes and provenance hashes. It does not alter nominal geometry,
the observation contract, or reset semantics.

## Gates and interpretation

The baseline must have at least `3/4` long-held terminal episodes to calibrate
the wave. A perturbation direction is useful only if it has at least `3/4`
long-held terminal episodes and clipping below `1%`; then the probe is
`PROMISING` for a bounded residual basin. If baseline calibrates but no fixed
direction qualifies, the probe is `UNPROMISING` and no additional offset sweep
or PPO run is launched. This does not prove a causal improvement, generalize to
new tau, or establish Cm utility.

Budget: one idle GPU, one eight-minute smoke/full wall-time envelope, below
2 GiB output and no foreign-process interference. The full run is a single
16-env/542-control wave; no seed or offset sweep follows this card.

## Results

Pending the fixed smoke and full execution. The retained tail candidate remains
the declared comparison arm regardless of this diagnostic.

## Limitations / follow-up

This is a single motion and a separate-live descriptive comparison. It cannot
distinguish a thumb offset from a missing state/history adaptation mechanism,
and it does not change the direct-action BC/PPO conclusion. A positive result
would justify exposing the shared bounded residual adapter to the ref2 goal
policy; a negative result closes only these four fixed offsets.
