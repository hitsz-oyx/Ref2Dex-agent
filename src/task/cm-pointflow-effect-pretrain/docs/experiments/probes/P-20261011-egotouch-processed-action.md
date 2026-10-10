---
schema: ref2dex.probe.v2
probe_id: P-20261011-egotouch-processed-action
experiment_id: P-20261011-egotouch-processed-action
date: 2026-10-11
task: cm-pointflow-effect-pretrain
branch: cm-pointflow-effect-pretrain
git_commit: TBD
claim_id: C1
hypothesis_family: HF-egotouch-pressure-contact
probe_index_in_family: 3
seed_pool: probe
seeds: [230]
decision_changed_if_positive: keep an offline action/hand-trajectory-conditioned pressure-contact token as a teacher-side route
decision_changed_if_negative: close this future-hand oracle recipe and keep pressure-map use limited to representation/audit evidence
status: PLANNED
run_id: egotouch-processed-pressure-action-20261011-r1
---

# Action-conditioned processed pressure/contact probe

Class: Decision. The preceding processed-pressure history Probe tested only a
causal history-to-future forecast and lost to persistence on all five held
tasks. This probe asks the next narrower question from ref10: can a processed
pressure/contact map be predicted when an offline future hand trajectory is
provided as an action/geometry proxy?

The future hand trajectory is an oracle available only in the recorded sample;
it is not a deployable robot action and cannot support a policy claim by
itself. A shuffled future-hand arm is required to test whether any gain comes
from temporal alignment rather than extra hand capacity.

## Decision note

**Question:** does aligned future hand shape provide information about the
future processed pressure/contact map beyond pressure history and persistence?

**Evidence:** the processed grid is a real masked `(21,21)` signal, but the
history-only arm was worse than persistence on every held task. ref10 defines
`(H,A/tau) -> future pressure/contact`; the current pack has hand geometry but
no executable action labels, so future hand is the cheapest offline proxy.

**Cheapest test:** reuse the immutable pressure target pack, side-specific H4/K8
windows, one seed and 500 GPU updates. Compare pressure-only, past-hand,
aligned future-hand and donor-shuffled future-hand arms with the same target,
past-only normalization, and held-task persistence.

**Stop/gate:** stop on source hash drift, invalid masks, nonfinite loss or GPU
contention. A positive teacher signal requires the aligned future-hand arm to
beat persistence by at least 10% on at least three held tasks, with no held task
more than 20% worse and no corresponding aligned gain reproduced by the
shuffled donor. Even a positive result remains an oracle/representation result;
native fusion and a deployable action contract require a separate matched Probe.

## Fixed protocol

- Source pack: `outputs/cm-pointflow-effect-pretrain/egotouch-pressure-targets-20261011-r2`;
  manifest and per-record hashes are checked before training.
- Split: the frozen five fit and five held-task records; donor windows stay
  inside the same split and come from a different record/task.
- Inputs: past processed pressure values and finite-cell masks, side identity,
  optional past hand shape, and optional future hand shape. Future hand is
  normalized with its own past-window hand scale; no future pressure statistic
  is used as an input.
- Target: masked future pressure for K=8 plus the relative activation proxy
  `valid & (pressure/scale >= 0.05)`. The proxy is not physical contact GT.
- Compute: seed 230, batch 32, AdamW, 500 updates, one GPU; output is a new
  run directory and does not overwrite prior checkpoints.

## Results

To be filled after the fixed run. Report held per-record pressure MAE,
persistence MAE, contact F1 and the aligned-versus-shuffled donor comparison.

## Limitations / future evidence

The future hand is retrospective human geometry, not a robot action, and EgoTouch
pressure remains normalized rather than calibrated Newtons. This probe does not
test object-world contact, CHORD net wrench, T-Rex F6, RBO FT or downstream Cm
utility.
