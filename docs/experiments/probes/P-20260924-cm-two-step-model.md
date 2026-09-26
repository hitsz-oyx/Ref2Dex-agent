# P-20260924-cm-two-step-model

Date: 2026-09-24. Branch: `agent/cm-two-step-sequence`.
Classification: Decision.

## Question

Can a sequence-conditioned Cm predict the additional ten-step object
effect of a second controlled action on an unseen simulator seed? The
physical one-vs-two-step randomized Probe on seed168 passed its gate,
but this alone does not establish a usable Cm or policy benefit.

## Fixed minimal comparison

Collect the same four-arm randomized 64-env design on new seed169.
Train one compact raw-state Cm on seed168 selected contact rows for
500 AdamW updates and test only on seed169. Input is pre-treatment
`q`, joint velocity, object state, actor base action and the two
planned signed wrist-x doses (second dose 0 for one-step arm).
Targets are actual one-step and ten-step local object translations and
ten-step contact fraction. Compare same-architecture, same-initialized,
same-schedule state-only control whose two dose inputs are zeroed.
No post-treatment second-step actor action is used as a model input.

Primary gate: on held-out seed169, the action-conditioned model's
factual ten-step world-x RMSE improves ≥5% over state-only, and its
predicted two-step-minus-one-step signed x interaction has the same
sign and magnitude between 0.5× and 1.5× of the randomized observed
interaction. Both conditions must hold before any planning integration.
No architecture search or threshold adjustment on seed169. If the
physical interaction on seed169 itself disappears or flips, stop and
classify the result as uncertain before model-based planning.

One GPU for collection (<30 min, 100 MB); CPU 2 threads for training
(≤10 min, 100 MB). Stop on input/commit drift, nonfinite tensors,
improper dose, leakage or budget overrun. This is model usefulness,
not Cm policy utility; matched online Cm-on/off remains necessary.

## Result

Status: `UNPROMISING` for the fixed raw-concatenation model, while
two-step physical effects remain present. Code commit `b7f037f`.
`agent_two_step_cm_s168_train_s169_test` completed 500 CPU updates
using 719 seed168 train and 766 new seed169 test contact rows.
Held-out ten-step world-x RMSE was 28.08 mm vs state-only 31.71 mm
(11.43% better). But predicted signed second-step interaction was
6.28 mm vs randomized observed 19.84 mm (0.316×), below the fixed
0.5× gate. Do not connect this checkpoint to an online planner or
tune it on seed169. `report.json`, checkpoints and SHA are in the
run directory. A separate structured-effect architecture is tested
on a new seed170 in its own pre-registered card.
