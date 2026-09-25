# P-20260925-observation-route-identifiability

- Classification: Decision Probe.
- Cm: off. Experts and object route frozen by the preceding Probe.
- Question: Can the actor's initial 1442-D observation identify the frozen
  expert route without reading simulator object or motion IDs at inference?

## Decision rule

Collect first pre-action observations from the same twelve converted
motions on seeds214–216 (64 environments each). Train a `StandardScaler`
plus `SVC(C=1, kernel="rbf", gamma="scale")` on seeds214/215 and evaluate
once on seed216. Labels are the frozen route's expert IDs, derived from
simulator motion IDs **only for training and scoring**. Do not include the
motion ID, environment ID, start frame or outcome as model inputs.

If heldout expert accuracy is >=90% and there are no mistakes among the
four object identities that passed the previous coverage gate (airplane,
duck, mug, toothpaste), build and evaluate an observation-driven router.
Otherwise inspect which objects confuse the model before spending GPU on
Cm routing. This is a feasibility Probe, not a claim of grasp improvement.

Use one idle GPU for observation export, <15 minutes total and <200 MB
outputs. Stop if checkpoint/input drift, incomplete episodes, or GPU
conflict. Record the actual route accuracy and per-object confusion.

## Results

Pending.
