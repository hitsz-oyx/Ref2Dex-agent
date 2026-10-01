# P-20261001-actuation-effect-factorization

Decision Probe. Design fixed before model fitting; reused historical data, not
independent policy Validation or evidence of a novel general factorization principle.

## Question, decision and cheapest test

Do predictable realized hand motions carry useful one-step physical information
beyond native commands and current state? A useful available predictor justifies
fresh counterfactual/contact-conditioned collection. If only future realized motion
helps, its privileged information is unavailable and no controller is justified.
If neither helps, do not collect a large new action bank for this factorization.

The existing1920complete episodes contain current/next physical state and actions.
Use them instead of immediately collecting new simulation. Geometry audit shows
that the old force-proxy pulse test changes geometric regimes across actors;
this screen evaluates current-state proximity explicitly.

## Frozen data and targets

- Read-only original HF08 r7 `collect_s283` fit and `collect_s284` test, each960
  complete episodes. These data were examined and used in earlier research;
  they are episode-disjoint for this fresh fit, not pristine independent test data.
- Verify every shard against its completed collection manifest; retain rows at
  step%16==0, exclude terminal transitions. No future-object outcome chooses rows.
- Current geometry from pinned Inspire/airplane URDF and seed42 surface sampler.
  Distance uses193hand/128object points (stride8, fixed approach convention).
  Primary stratum is current sampled gap<=20mm. Require>=1024test rows and
  >=64test episodes in this stratum; otherwise UNCLEAR/INSUFFICIENT_COVERAGE.
- Predict current-object-local position residual over constant-velocity motion:
  `R_t^T * (p_next-p_t-v_t/30)`. Response reconstruction is baseline plus this
  predicted residual. Report position error in mm, full and near strata separately.
- State input:15wrist/finger angles,18joint velocities (wrist linear velocity
  rotated to object frame),6object-local linear/angular velocities, gravity axis
  in object frame, object world height,6object-relative palm/tip positions,2force
  proxies. No future object state, reference identity or future reference pose.

## Models and controls

- Passive analytic constant-velocity predictor: zero residual.
- Same63-D state and18-D action-feature slots for all effect MLPs; two128-unit
  SiLU layers and3outputs. State-only fills action slots with zero.
- `raw_command`: normalized18-D native command.
- `nominal_motion`:18-D current-object-local palm/tip displacement from exact
  native action-to-PD-target mapping plus FK. The six overwritten dependent
  finger commands have no nominal effect; this known kinematic fact is not new.
- `learned_motion`: FK displacement of predicted next joints. Separate actuator
  MLP (two128-unit SiLU layers) takes current q/dq,PD-target error and the same
  current physical/geometry state features (contact feedback can affect q), and learns
  joint residual over constant velocity. Fit uses only collection283; normalization
  and residual scales use fit only. Its future hand motion is available by prediction.
- `oracle_motion`: FK displacement from measured next joints. This privileged
  diagnostic is not deployable; hand motion may reflect object contact feedback.
  A favorable oracle result does not establish an independently predictable mediator.
- Seeds411/412/413; each actuator and effect MLP has1000Adam updates,lr0.001,
  batch256. Effect sampling is half current-near rows and half all fit rows.
  Fixed final checkpoints; no test selection or additional seeds/updates.
- Strong nonlearned baselines and every method/seed are reported. Same effect
  capacity/updates/data; factorized method has extra actuator fit compute, reported
  separately and never called free data or compute.
- Geometric transport controls use a current-distance-weighted mean of the six
  palm/tip displacement vectors (only centers within20mm of the sampled object
  surface contribute; exponential distance weights with scale20mm). If no center
  qualifies, use passive motion. These assumptions can fail with slip or palm-side
  contact; the controls are label-free approximations, not ground-truth dynamics.

## Fixed gate and budget

PROMISING requires learned-motion seed-mean near RMSE at least10% below EACH of
state-only, raw-command,passive and learned-motion geometric-transport predictors,
with nonnegative improvement
versus raw-command in all three optimization seeds. Otherwise UNPROMISING.
Oracle/nominal changes are diagnostics and cannot substitute for this gate.

One idle GPU4,2CPUthreads,<=900seconds,<=500MiB outputs/cache. Input or source
drift, insufficient geometry, nonfinite tensors or occupied GPU stops the run.
Frame samples are temporally correlated; counts are not independent trajectories.
There is one object/three reference motions; no generalization, policy benefit,
cross-embodiment transfer or journal-ready result is claimed by this screen.
