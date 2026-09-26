# P-20260925-cm-supported-rise-event

- Classification: Decision Probe, CPU screening first.
- Policy: frozen self-trained airplane e260. No official actor in this
  data or final-policy route.

## Question and decision

Earlier Cm heads predicted physical displacement or mean contact, but
their action information did not reliably improve complete grasping.
Does a task-aligned ten-step event label retain information about the
*randomized* wrist x/y/z action? The label is object-z increase >=3 cm
**and** hand/object contact in >=5 of the next ten steps. This is a
short-horizon surrogate, not the strict full-episode five-consecutive-
step held-lift metric.

Fit one fixed small histogram-gradient-boosting classifier on existing
randomized seeds161/162. Inputs: pre-action q, dof velocity, object
state, base actor action, plus the executed wrist xyz perturbation.
Controls use the identical model and state: `blind` zeroes the
perturbation; `shuffled` permutes training perturbations within the
same intervention step. Evaluate untouched seed163 as a cheap
development screen. The model, feature set and thresholds are frozen
before reading prediction scores.

Pass the screen only if action-aware Brier is >=5% lower than both
blind and shuffled, AUROC exceeds blind by >=0.02, and predicted
plus-minus event-rate signs agree with observed randomization for at
least two axes whose observed absolute contrast is >=3 pp. If it
passes, collect new randomized seed237 with identical protocol and
repeat the frozen fit/test procedure; only a second pass justifies an
online action-selector Probe. If it fails, stop this H10 event target.
The test is of action information, not policy utility.

CPU <=10 minutes, <20 MB output. No GPU unless the existing-data
screen passes. Stop on input manifest drift, nonfinite features,
incomplete follow-up or missing treatment cells.

## Results

The existing-data screen completed at commit `1e86238` with 1,574
randomized training rows and 803 development rows (297 and 156
positive contact-supported rise events, respectively). The identical
fixed classifier gave:

| Input | Brier | AUROC |
| --- | ---: | ---: |
| State + actual perturbation | 0.06307 | 0.95190 |
| State, action blind | 0.06375 | 0.95111 |
| State + shuffled training perturbation | 0.06371 | 0.95156 |

Action-aware Brier improved only **1.08%** versus blind, below the
predeclared 5% gate, and AUROC improved only 0.00079 versus the
required 0.02. The observed plus-minus event-rate contrasts were
x +0.82 pp, y −2.61 pp and z +8.26 pp; the model predicted x −0.14,
y 0.00 and z +1.12 pp. Only z met the 3 pp observed-contrast and
direction condition. The joint gate failed (`UNPROMISING`) for this
H10 event target under the fixed learner. No new seed237 was collected
and no online selector was launched. This does not rule out a longer
decision horizon or a different contact/grasp representation.

Report: `outputs/CmResidual/agent_cm_supported_rise_event_dev_s163/report.json`.
