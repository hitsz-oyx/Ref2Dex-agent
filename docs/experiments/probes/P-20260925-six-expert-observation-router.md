# P-20260925-six-expert-observation-router

- Classification: Decision Probe.
- Cm: off. Six self-trained PPO checkpoints.

## Question and decision

Can the initial 1442-D actor observation recover the newly frozen
six-expert object route, including cup, on unseen seed232? The prior
training feature exports from seeds214/215 are independent of which
expert was run; relabel them using the new route. Export initial
features on seed232 with the new route. Train the previously selected
balanced SVC with the same hyperparameters, without tuning on seed232.

Pass if held-out accuracy is >=90% and all airplane, cup, duck, mug
and toothpaste examples choose the specified expert. If passed,
evaluate online observation routing on fresh seed233 and compare
expert choices and held-lifts with the fixed six-expert route. If
failed, inspect only the confusion matrix before choosing a different
representation. One GPU for <=10 minutes and <100 MB outputs; stop
on incomplete episodes or config/checkpoint drift. This is a Probe,
not policy-utility Validation.

## Results

Pending.
