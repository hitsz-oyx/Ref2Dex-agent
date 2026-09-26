# P-20260924-multiaxis-h10-cm

date: 2026-09-24
branch: `agent/cm-multiaxis-long-horizon`
classification: Decision

## Question and next decision

Can an action-conditioned Cm trained on real randomized wrist x/y/z
transitions predict a useful *ten-step* effect on a new simulation
seed, beyond a same-capacity state-only model? If yes, design a minimal
matched policy-decision Probe. If no, do not spend PPO compute merely
because the physical average effect is nonzero.

## Frozen protocol

Train on completed seeds161/162 from
`P-20260924-multiaxis-h10-effects`; collect seed163 with the identical
64-env, 16-step, ±0.1, 10-step randomized protocol solely for testing.
Do not change architecture or training after viewing seed163 outcomes.

Three models share a fixed 500-update, batch96 AdamW schedule and
one-step object translation + 10-step object translation + 10-step
contact-fraction targets:

1. Raw action-conditioned MLP, width64 (`RawContactAwareCm`).
2. Same raw MLP initialization/capacity with its action input zeroed.
3. Six-region geometric contact-aware Cm (`ContactAwareCm`, width32),
   using the existing pinned online handflow calibration.

The action-conditioned models receive *executed* action at training;
candidate contrasts at test vary only one wrist axis of the
pre-treatment base action by ±0.1. No post-treatment state enters
model input. Train loss is 0.5 scaled one-step smooth-L1 + 0.5 scaled
ten-step smooth-L1 + 2 contact MSE. CPU-only, 2 threads.

Primary held-out metric: factual ten-step world-object x displacement
RMSE on test rows assigned to the x-axis ± arms. Continue to a policy
Probe only if the seed163 randomized x effect is ≥5 mm and at least
one action-conditioned model beats state-only RMSE by ≥5%, while its
mean predicted ±0.1 x contrast is positive and lies within 0.5–2×
the observed step-stratified x effect. Also report y/z displacement,
contact RMSE and action-contrast calibration, but do not select a
model on those exploratory metrics. If x RCT is too weak, call it
`UNCLEAR`; if it is strong but no model passes, call it `UNPROMISING`
for these fixed models and do not attach them to PPO.

## Budget and stop

One new GPU collection run (<30 min, <100 MB), one CPU training and
analysis run (<10 min, <100 MB), total <60 min; no policy training.
Stop on source/hash drift, assignment mismatch, non-finite targets,
GPU conflict, target leakage or budget breach. All run IDs and exact
input SHA256s go in run manifests.

## Result

Status: `PROMISING` for held-out multiaxis ten-step Cm prediction,
not for grasp policy utility. Code commit: `34d14fc`.

Test collection `agent_multiaxis_h10_s163_n64` completed with 803
valid, unclipped randomized contact treatments. It was not inspected
for effect before the models, schedule and gate were committed. The
CPU training/evaluation run
`agent_multiaxis_h10_cm_s161162_train_s163_test` completed in 7.3 s
with 1574 train and 803 held-out samples. Exact commands, input
SHA256s, output checkpoint SHA256s and status are in run manifests;
all three model checkpoints and `report.json` are in the model run.

| Test seed163, 10-step x outcome | State-only | Raw action Cm | Six-region Cm |
| --- | ---: | ---: | ---: |
| Factual x RMSE (mm) | 28.16 | 26.69 | 25.25 |
| RMSE improvement vs state-only | — | 5.22% | 10.35% |
| Predicted ±0.1 x effect (mm) | 0 | 17.49 | 21.41 |

The step-stratified observed ±0.1 x effect was +16.76 mm.
Both action-conditioned models pass the fixed ≥5% RMSE and 0.5–2×
contrast calibration gate. The geometric model was best on x, but
had worse held-out y/z RMSE (34.32/33.67 mm) than the raw model
(27.21/30.04 mm); therefore this does **not** establish a generally
superior geometry representation. Raw and geometry contact RMSE were
.118/.150, versus state-only .128. Single-seed test and training loss
cannot establish robust generalization or policy benefit.

## Decision update

Proceed to one minimum matched online decision Probe with a frozen
model. Before choosing a candidate-control score, inspect the
randomized cross-axis object/contact effect matrix to avoid treating
lateral displacement as grasp progress. Do not tune any model weight,
loss or architecture on seed163. Use new policy-evaluation seeds.
