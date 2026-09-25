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

Runs214–216 completed; the first pre-action 1442-D observation was
exported for 64 environments per seed. Training on seeds214/215 and
testing on seed216 gave **62/64 (96.9%)** correct expert labels.
Airplane15/15, duck6/6 and toothpaste5/5 were correct, but mug was only
3/5. The two mug errors were assigned to the default source expert.
The prespecified gate therefore **failed** despite high overall accuracy.
The two misclassified mug episodes had start frames8 and0; the frozen
privileged router failed on the first and succeeded on the second. This
Probe is `UNCLEAR` for observation-driven routing, and does not test Cm.

Offline diagnostic on the already exposed data: changing only the SVC
class weight to `balanced` improved leave-one-seed-out results across
seeds214/215 from 117/128 to 123/128 overall, and from 4/10 to 8/10 on
mug. On seed216 it achieved 64/64, but this is exploratory because the
model choice followed inspection of seed216. Test that single fixed change
on a fresh seed before using it in an online router.

Artifacts: `outputs/CmResidual/agent_observation_route_s214/` through
`..._s216/`, and
`outputs/CmResidual/agent_observation_route_probe_20260925/report.json`.
