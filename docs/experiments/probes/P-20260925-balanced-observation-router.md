# P-20260925-balanced-observation-router

- Classification: Decision Probe.
- Cm: off.
- Single change: use `class_weight="balanced"` in the observation SVC.

## Question and gate

Does balancing the expert labels remove mug's initial-observation routing
errors on an **unseen** seed? Keep the same frozen 12 motions, expert
checkpoints, normalizer, 1442-D input, `StandardScaler`, RBF kernel, C=1
and gamma=`scale`. Train only on seeds214/215, then export and classify
64 first observations on new seed217. The observations contain no object
or motion ID as features; those IDs supply labels after export.

Pass if accuracy >=90% and airplane, duck, mug and toothpaste have no
expert classification errors. If passed, build an online observation
router and compare grasp results with the fixed privileged route; if not,
inspect observation representation rather than tuning this SVC on seed217.
One idle GPU, <=5 minutes, <100 MB outputs. Stop on input drift, GPU
conflict or incomplete feature export.

## Results

The new seed217 completed with 64 first observations. The exact
predeclared balanced classifier trained only on seeds214/215 classified
**64/64** expert labels correctly, including airplane15/15, duck6/6,
mug5/5 and toothpaste5/5. The feasibility gate **passed**. Result:
`PROMISING` for an online observation-driven expert route on these
same known converted motions. The test used simulator IDs only to score
predictions, and is not evidence of grasp improvement or transfer to
unseen motions/objects. An online grasp comparison is still required.

Artifacts: `outputs/CmResidual/agent_observation_route_s217/` and
`outputs/CmResidual/agent_observation_route_probe_balanced_20260925/report.json`.
