---
schema: ref2dex.probe.v2
probe_id: P-20261011-tau-residual-basin-repeat
experiment_id: P-20261011-tau-residual-basin-repeat
date: 2026-10-11
task: consequence-evaluator
branch: main
git_commit: 00859cf
claim_id: C3
hypothesis_family: HF-reference-tracking-control
probe_index_in_family: 6
seed_pool: probe
seeds: [287]
decision_changed_if_positive: retain thumb_pitch_minus as a candidate bounded-residual adapter for one later goal-conditioned integration Probe
decision_changed_if_negative: close the fixed thumb offset route and preserve only the retained canonicalized tau candidate
status: PROMISING
run_id: ref2-tau-residual-basin-repeat-20261011-r1
---

# One-launch repeat of the fixed tau residual basin

## Motivation / Decision Note

The preceding fixed-offset wave was positive only for a single motion and one
fresh launch: the retained actor with no external offset and
`thumb_pitch_minus (coord11=-0.05)` each reached `4/4` long-held terminal rows,
while the yaw direction failed and the pitch-plus direction saturated. Before
exposing the candidate offset to the ref2 goal route, this one-run repeat tests
whether the low-clipping pitch direction survives a fresh launch seed.

This is a matched robustness Probe, not an offset or seed sweep. It keeps the
same frozen checkpoint, geometry, reference packet, native adapter, four roles,
16 separate-live rows, 542 controls and outcome gate. It uses no training, no
future q/object, phase/clock/contact input and no packet actions/q labels.

## Fixed protocol and gates

Run `probe_reference_tracking.py --residual-basin` with seed287 and the same
four roles (`baseline`, `thumb_pitch_plus`, `thumb_pitch_minus`,
`thumb_yaw_plus`), four rows per role. The baseline must have at least `3/4`
long-held terminal rows. The repeat confirms the candidate only if
`thumb_pitch_minus` has at least `3/4` long-held terminal rows and clipping below
1%. If the baseline calibrates but the candidate fails, the probe is
`UNPROMISING`; if the baseline itself fails, it is `UNCLEAR`. No additional
offset, seed or PPO run follows this card.

Budget: one idle GPU, one eight-minute wall-time envelope, below 2 GiB output.

## Results

The repeat completed in
`outputs/consequence-evaluator/ref2-tau-residual-basin-repeat-20261011-r1/`
with GPU2, seed287, 16 environments, 542 controls, 65.7 s wall time and
148,584,960 bytes peak Torch allocation. The baseline had `3/4` long-held
terminal rows. `thumb_pitch_minus` had `3/4` long-held terminal rows with
`5/2168 = 0.2306%` clipping, so it passed the predeclared repeat gate.
`thumb_pitch_plus` had `3/4` terminal rows but `52.58%` clipping. The yaw
direction happened to reach `3/4` terminal rows in this launch, compared with
`0/4` in the first wave; this launch-to-launch variation is retained as a
limitation rather than a positive yaw claim. All `16/16` rows again had
`success=false` for complete controlled placement.

The independent audit
`outputs/consequence-evaluator/ref2-tau-residual-basin-repeat-audit-20261011-r1/audit.json`
is `PASS`; delta, latent, applied-command and native-PD errors are zero and
feature replay maximum is `1.91e-6`. The repeat is therefore `PROMISING` for
retaining `thumb_pitch_minus` as a candidate bounded-residual adapter, still
only on this single motion.

## Limitations / follow-up

Even a positive repeat remains single-motion and separate-live evidence. It can
authorize one later goal-conditioned integration Probe through the shared
bounded-residual executor, but cannot establish full placement success,
cross-motion generalization, causal preload effect or Cm utility.
