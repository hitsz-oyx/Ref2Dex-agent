---
schema: ref2dex.probe.v2
probe_id: P-20261011-egotouch-pressure-auxiliary
experiment_id: P-20261011-egotouch-pressure-auxiliary
date: 2026-10-11
task: cm-pointflow-effect-pretrain
branch: cm-pointflow-effect-pretrain
git_commit: 8a044c8
claim_id: C1
hypothesis_family: HF-egotouch-pressure-contact
probe_index_in_family: 1
seed_pool: probe
seeds: [229]
decision_changed_if_positive: use processed hand-surface pressure/contact as a candidate auxiliary representation before native point-cloud fusion
decision_changed_if_negative: keep pressure-map auxiliary deferred; retain T-Rex/RBO and simulator wrench routes as separate candidates
status: UNPROMISING
run_id: egotouch-pressure-20261011-r1
---

# EgoTouch processed pressure/contact auxiliary

Class Decision. This probe asks a narrow question: with the same past hand-local
condition, can the released processed `(21,21)` pressure maps predict future
pressure better than persistence across held tasks? It does not claim calibrated
force, 3-D contact ground truth, CHORD wrench supervision, or native policy gain.

## Decision note

**Current question:** should the previous 512-channel raw-glove negative close the
tactile route, or should the processed pressure map become the first auxiliary
target?

**Evidence:** the old learner consumed `jq_pressure.json` bytes and never read
`pressure_grids.npz`. The audited grids contain finite measured cells and repeated
nonzero pressure, while `manual_contact=false` is semantically unqualified.
TouchAnything/EgoTouch exposes dense pressure supervision, and T-Rex independently
shows that temporal tactile force can help contact-rich control. Therefore the raw
negative cannot close the processed-pressure route.

**Action:** run the cheapest local pressure-only probe on the ten already acquired
official TRAIN recordings. Build an immutable, side-specific masked window pack;
compare persistence, history-only, and history+past-hand models. Keep the pressure
map in hand-surface coordinates and call thresholded activation a `contact_proxy`.

**Cost and stop conditions:** CPU preparation/tests only, then one small GPU seed
(H=4, K=8, 500 updates, at most 20 minutes and 2 GiB artifacts). Stop on missing
source files, non-monotonic clocks, all-invalid masks, nonfinite loss, or GPU
contention. A positive result opens a hand-surface token/fusion probe; a negative
result leaves the calibrated T-Rex/RBO and simulator CHORD routes open.

## Fixed protocol

- Source revision: `zhouzhoujy/EgoTouch@cfdbb0ac31cc2af4247943820aa250575e7e6637`;
  only the existing local downloads are used.
- Split: the pre-existing 5 fit-task / 5 held-task separation. No record or task
  is moved after looking at future values.
- Window: 30 Hz source rows, history `H=4`, target horizon `K=8`; each hand is
  sampled independently so missing opposite hands do not discard valid pressure.
- Target fields: `pressure[T,21,21]`, `pressure_valid` from finite grid cells,
  `frame_index`, `timestamps`, and optional past wrist-relative hand shape. NaNs
  become zero only after the validity mask is stored.
- Activation: fixed `tau=0.05` on the published normalized pressure scale;
  `contact_proxy = pressure_valid & (pressure >= tau)`. This is not physical
  contact GT and is not chosen from held-task values.
- Inputs never use future normalization maxima. Pressure history is omitted from
  the first arm to avoid global-max leakage; the condition is past hand-local
  shape only. The history-only arm receives the same hand-validity mask.
- Loss: masked Huber pressure regression plus optional proxy BCE. Report all-valid
  and proxy-positive MAE, macro task H8 MAE, proxy F1/PR-AUC when prevalence is
  sufficient, and persistence errors.
- Gate: `PROMISING` requires history+hand to improve masked H8 MAE by at least 10%
  over persistence on at least 3 held tasks, no held task >20% worse, and proxy
  metrics no worse than persistence. Sparse proxy labels yield `UNCLEAR`, not a
  negative contact conclusion.

## Limitations / future evidence

Pressure-grid units are normalized sensor values, not Newtons. EgoTouch has no
audited object-world calibration, so this probe cannot feed CHORD support vectors
or native object points. A future T-Rex arm can add per-fingertip 6D wrench history;
an RBO/simulator arm is required for object-COM wrench labels. Any native fusion
must use a matched Cm-on/Cm-off downstream comparison.

## Results

The prepared pack used all ten already acquired records (5 fit / 5 held),
side-specific H4/K8 windows, finite-cell masks, and raw sensor inputs divided by
255. Two matched GPU arms used seed 229, batch 32, AdamW, 500 updates, and the
same architecture except for the past-hand input. Each arm completed in about
8 seconds on an otherwise idle RTX 3090; peak `nvidia-smi` memory returned to
2 MiB after exit.

The raw-only and raw+hand pressure MAE both lost to persistence on every held
task. The exact held pressure MAE / persistence MAE pairs were:

| held task | raw | raw+hand | persistence |
|---|---:|---:|---:|
| drag chair | 0.0754 / 0.0299 | 0.0655 / 0.0299 | 0.0299 |
| flip book pages | 0.0915 / 0.0330 | 0.0917 / 0.0330 | 0.0330 |
| badminton racket | 0.0727 / 0.0182 | 0.0731 / 0.0182 | 0.0182 |
| shop for snacks | 0.0528 / 0.0180 | 0.0514 / 0.0180 | 0.0180 |
| squeeze pliers | 0.0401 / 0.0116 | 0.0399 / 0.0116 | 0.0116 |

This probe is `UNPROMISING` for the raw-sensor-to-future-pressure recipe. It
does not close the processed pressure route: the model did not consume a
processed pressure history, and the persistence comparison uses the released
processed scale. It also does not test T-Rex wrench streams or RBO object-frame
wrenches. Those remain separate follow-up evidence routes.
