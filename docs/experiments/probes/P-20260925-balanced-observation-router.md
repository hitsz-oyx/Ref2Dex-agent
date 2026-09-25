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

Pending.
