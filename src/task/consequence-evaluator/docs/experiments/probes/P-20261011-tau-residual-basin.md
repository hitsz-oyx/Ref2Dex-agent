---
schema: ref2dex.probe.v2
probe_id: P-20261011-tau-residual-basin
experiment_id: P-20261011-tau-residual-basin
date: 2026-10-11
task: consequence-evaluator
branch: main
git_commit: 00859cf
claim_id: C3
hypothesis_family: HF-reference-tracking-control
probe_index_in_family: 5
seed_pool: probe
seeds: [286]
decision_changed_if_positive: retain the bounded thumb residual as a candidate adapter and run one matched robustness Probe before exposing it to the ref2 goal route
decision_changed_if_negative: stop fixed thumb-offset diagnostics and preserve the retained tail canonicalization candidate without more PPO
status: PROMISING
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
| `baseline` | zero external offset; the retained actor latent is still active |
| `thumb_pitch_plus` | residual coordinate 11, `+0.05` |
| `thumb_pitch_minus` | residual coordinate 11, `-0.05` |
| `thumb_yaw_plus` | residual coordinate 10, `+0.05` |

The values are chosen before execution and are not fit from this rollout. The
run records raw pre-clamp intended commands, the clamped/applied `action`, native
PD targets, clipping, full physical outcomes and provenance hashes. It does not alter nominal geometry,
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

The engineering smoke completed first; it is retained only as a wiring check. The
fixed full wave completed in
`outputs/consequence-evaluator/ref2-tau-residual-basin-20261011-r1/` with
GPU2, seed286, 16 environments, 542 controls, 66.2 s wall time and 148,584,960
bytes peak Torch allocation. The retained actor with no external offset
(`baseline`) reached long-held terminal in `4/4` rows with zero clipping.
`thumb_pitch_minus` also reached `4/4`, with `4/2168 = 0.1845%` clipping;
`thumb_pitch_plus` reached `4/4` but clipped `1200/2168 = 55.35%`; and
`thumb_yaw_plus` reached `0/4` terminal-held rows, with median held duration
131 frames and mean object-position RMSE 0.4197 m. All `16/16` rows had
`success=false` under the complete controlled-placement outcome, so the gate
only concerns sustained holding, not full task completion.

The predeclared gate therefore labels this fixed-offset basin Probe
`PROMISING`: the baseline calibrated and at least one perturbation direction
(`thumb_pitch_minus`) met the 3/4 terminal-held and <1% clipping thresholds.
The independent audit
`outputs/consequence-evaluator/ref2-tau-residual-basin-audit-20261011-r1/audit.json`
is `PASS`: delta, latent, applied-command, native-PD reconstruction errors are
zero and the tau/live-state feature replay maximum is `2.86e-6`.

## Limitations / follow-up

This is a single motion and a separate-live descriptive comparison. It cannot
distinguish a thumb offset from a missing state/history adaptation mechanism,
and it does not change the direct-action BC/PPO conclusion. A positive result
would justify one matched robustness Probe before exposing the shared bounded
residual adapter to the ref2 goal policy; a negative result closes only these
four fixed offsets.
