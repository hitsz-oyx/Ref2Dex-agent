# P-20260925-cm-postcontact-three-arm-value

date: 2026-09-25
branch: `agent/cm-postcontact-heldlift-value`
code commit: `988ce52`
classification: Decision Probe
status: **UNPROMISING for the current post-contact wrist-z value route**

## Question

Can a stricter three-arm randomized design connect a post-contact wrist-z Cm
representation to final held-lift value after removing the two-arm base-versus
candidate replay ambiguity? Each eligible environment is assigned exactly one
of `-z`, base, or `+z` at step 50, then runs to the complete episode outcome.
The assignment is balanced within each simulator batch. This is a same-batch
population comparison; it is not a same-state counterfactual for an individual
environment.

## Data and audit

The collector now exposes `--three-arm-randomized`, records
`intervention_valid`, and preserves the final outcome fields already audited
for the two-arm probe. Training uses seeds 250/251 and held-out evaluation uses
252/253. After filtering `pre_contact & intervention_valid`, the arm counts are:

| split | seed | −z | base | +z | selected rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 250 | 21 | 21 | 21 | 63 |
| train | 251 | 20 | 19 | 19 | 58 |
| test | 252 | 21 | 21 | 21 | 63 |
| test | 253 | 21 | 21 | 21 | 63 |

All four payloads are `COMPLETED`, contain full episode labels, and pass the
actual executed-action delta audit. The analysis tool also checks finite state
features, per-file arm propensities, and the three-arm schema before fitting.

## Held-out offline value

The model uses the same 69-dimensional pre-action state as the preceding
probe. `cm_aware` adds a three-way treatment indicator and state interactions;
`state_only` has no treatment input; `action_shuffled` breaks the treatment to
outcome correspondence in the training files. Values use a multi-arm Hájek
estimator and environment-cluster bootstrap.

| model | policy value | cluster 95% interval | selected −z / base / +z |
| --- | ---: | ---: | ---: |
| state-only | 0.6667 | [0.5227, 0.8182] | 0 / 0 / 126 |
| Cm-aware | **0.6809** | [0.5500, 0.8049] | 35 / 41 / 50 |
| action-shuffled | 0.5349 | [0.3749, 0.6765] | 55 / 32 / 39 |

The Cm-aware gain over state-only is **1.42 percentage points**, below the
pre-registered 5-point minimum. The intervals overlap substantially. The
held-out factual AUROC/Brier are 0.762/0.207 for Cm-aware and 0.799/0.183 for
state-only, so the small policy-value difference is not supported by a better
factual predictor. The arm-wise held-lift rates in the test files are −z
0.429, base 0.762, and +z 0.667; this also shows that the state-only policy's
training-set choice of +z does not identify the best held-out arm.

As a descriptive secondary check, the pooled held-out continuous outcomes also
favor leaving the action unchanged: mean maximum contact-supported lift is
0.179 m / 0.248 m / 0.187 m for −z / base / +z, and mean contact fraction is
0.512 / 0.570 / 0.490. These are arm means from 42 rows per arm, not a new
formal gate, but they make the binary result directionally consistent.

The pre-registered offline gate requires Cm-aware to beat both controls and
to exceed their maximum by at least 5 points. It therefore **failed**. No
frozen online Cm-on/Cm-off run was started from this artifact.

## Decision update

The three-arm design resolves the main protocol weakness identified in the
two-arm probe, but it does not produce enough held-lift policy value to justify
an online selector for this wrist-z intervention. Stop this post-contact
single-step wrist-z value family for now. Preserve the collector and complete
episode labels for a future action family only if a new representation or
supervision target changes the decision question. This Probe covers one
airplane distribution and one contact time; it does not evaluate Cm in general
or cross-object transfer.

## Artifacts and checks

* `third_party/DExplore/dexplore/evaluate_randomized_action.py`
* `src/task/CmResidual/tools/probe_postcontact_three_arm_value.py`
* `src/task/CmResidual/tests/test_probe_postcontact_three_arm_value.py`
* `outputs/CmResidual/agent_postcontact_three_arm_value_probe_v2/report.json`
* `outputs/CmResidual/agent_postcontact_three_arm_value_probe_v2/models.joblib`

Input payload SHA-256 values and output artifact hashes are recorded in the
probe `run_manifest.json`. The focused randomized/action/history/post-contact
test set passes **20 tests**; both collector and analysis tools pass
`py_compile`.
