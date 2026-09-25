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

Pending.
