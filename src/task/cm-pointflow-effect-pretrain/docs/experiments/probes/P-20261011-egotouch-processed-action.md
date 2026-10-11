---
schema: ref2dex.probe.v2
probe_id: P-20261011-egotouch-processed-action
experiment_id: P-20261011-egotouch-processed-action
date: 2026-10-11
task: cm-pointflow-effect-pretrain
branch: cm-pointflow-effect-pretrain
git_commit: 73c7900
claim_id: C1
hypothesis_family: HF-egotouch-pressure-contact
probe_index_in_family: 3
seed_pool: probe
seeds: [230]
decision_changed_if_positive: keep an offline action/hand-trajectory-conditioned pressure-contact token as a teacher-side route
decision_changed_if_negative: close this future-hand oracle recipe and keep pressure-map use limited to representation/audit evidence
status: UNPROMISING
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

The fixed run used all 3,166 fit windows and 3,924 held windows, seed 230,
batch 32, AdamW and GPU1 for 500 updates per arm. The output is
`outputs/cm-pointflow-effect-pretrain/egotouch-processed-pressure-action-20261011-r1/result.json`;
the pack manifest hash is
`315a32ef58d3ebe660c40638602988a3ea6bd741cd697326f36b8343504941cf`.

Original-scale held pressure MAE / persistence MAE was:

| held task | pressure | pressure + past hand | aligned future hand | shuffled future hand | persistence |
|---|---:|---:|---:|---:|---:|
| drag chair | 0.04849 / 0.02993 | 0.04677 / 0.02993 | 0.04690 / 0.02993 | 0.04788 / 0.02993 | 0.02993 |
| flip book pages | 0.07162 / 0.03302 | 0.07130 / 0.03302 | 0.07048 / 0.03302 | 0.07185 / 0.03302 | 0.03302 |
| badminton racket | 0.06085 / 0.01822 | 0.05982 / 0.01822 | 0.05965 / 0.01822 | 0.06057 / 0.01822 | 0.01822 |
| shop for snacks | 0.03989 / 0.01802 | 0.04060 / 0.01802 | 0.04045 / 0.01802 | 0.04021 / 0.01802 | 0.01802 |
| squeeze pliers | 0.02865 / 0.01162 | 0.02867 / 0.01162 | 0.02902 / 0.01162 | 0.02954 / 0.01162 | 0.01162 |

All four learned arms lost to persistence on all five held tasks. The aligned
future-hand arm did not beat the shuffled donor; their held pressure MAE values
differed by at most 0.00099 in original-scale units. Contact F1 was also below
persistence on every held task (for example, aligned future-hand F1 ranged
0.454--0.769 versus persistence 0.737--0.914). The result is therefore
`UNPROMISING` for this offline action/hand-trajectory-conditioned pressure
recipe. It closes this teacher-side forecast recipe, not calibrated tactile
representation, T-Rex per-fingertip wrench, RBO measured FT or simulator CHORD.

## Limitations / future evidence

The future hand is retrospective human geometry, not a robot action, and EgoTouch
pressure remains normalized rather than calibrated Newtons. This probe does not
test object-world contact, CHORD net wrench, T-Rex F6, RBO FT or downstream Cm
utility.
