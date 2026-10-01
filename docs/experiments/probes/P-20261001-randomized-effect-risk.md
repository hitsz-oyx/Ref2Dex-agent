# P-20261001-randomized-effect-risk

Decision Probe. Freeze before new simulation. Old paired run remains invalid.

## Question and cheapest valid design

Do existing nominal-motion models predict conditional physical action effects
better than command models and zero effect on new randomized states? Exact
cold-replay pairing failed its matching gate. Randomization identifies a risk
DIFFERENCE without separately identifying individual paired outcomes or absolute
conditional-effect RMSE. Do not claim to measure those unavailable quantities.

## Estimand

Let X be the complete available pretreatment features at an acquired state. T is
independently assigned tozero,x+/x-,y+/y-,z+/z- with known nonzero propensities.
Y is one-step world object displacement over constant velocity. For each axis,
`Z_axis=I(T=axis+)/p_plus*Y-I(T=axis-)/p_minus*Y`.
Randomization implies E[Z|X]=tau(X), the three-axis conditional contrast matrix.
For frozen predictors f,g,
`E[ ||f||²-||g||²-2(f-g)·Z ] = E[||f-tau||²-||g-tau||²]`.
The common pseudo-outcome variance cancels. This elementary identity and inverse
probability weighting are not claimed as new methods. Partial/hidden simulator
state is integrated into the conditional response distribution, not cloned.

## Fixed acquisition and assignment

- Actors286/287, NEW initial seeds492/493,768environments per panel,256permotion.
  Same checkpoint/config/object/motions. Four batches,total3072environments.
- Within each motion, a CPU Generator seeded9900+evaluation_seed independently
  permutes balanced labels0..6 (counts37/37/37/37/36/36/36per256). Known label
  probability is count/256, recorded per environment. No global physics/actor
  RNG is consumed by this assignment; no outcome influences assignment.
- Assignment is made after native reset but before any physics step; it does
  not modify actions until the acquired trigger. Each environment receives at
  most one treatment. Prehistory/current X is therefore pretreatment.
- Trigger: first EVEN current pre-action step>=10with both force proxies and
  full sampled hand/object gap<=20mm,seed42. Geometry construction preserves
  native RNG. Check only untreated environments. Same 1538/1024surface assets.
- One-step pulse+/-0.01in named WORLD wrist-translation axis;zero leaves action
  unchanged. Exact native scales1, no clipping, no other action modifications.
  Subsequent commands are actor closed-loop actions; primary outcome is one step,
  before any subsequent actor action. Five-step outcomes are optional diagnostics.
- Require complete acquired one-step windows for ALL768environments in EVERY
  panel. No partial-panel or post-treatment window exclusions. Stop at360steps
  or first termination; unmet coverage gives UNCLEAR, no model ranking.
- Record physical pretreatment/current/future states, original/action commands,
  arm,propensity,trigger,motion,panel,geometrygap,initial RNG and actor/RMS hashes.
  No raw-action recovery objective or new training is introduced.

## Frozen predictors, scoring and gate

- Nominal/command models411–413from archived factorizationr1, hashes pinned before
  simulation. At each common observed X evaluate all six plus/minus actions;
  subtract predictions to form f,g. CURRENT object rotation only; no future
  hand/object state is an input. Zero-effect matrix is the strong third control.
- Report identified seed-mean risk differences (mm²) nominal-minus-command and
  nominal-minus-zero, perseed,perpanel/axis/motion. Also raw-minus-zero diagnostic.
  No absolute effect RMSE or paired-effect cosine is inferred.
- Descriptive fixed-substrate stratified bootstrap:2000replicates,seed12026.
  Within each of six evaluation-seed/motion strata, resample environment indices
  jointly across the two actors, preserving shared initial-seed/assignment blocks.
  Each panel/motion keeps its fixed sample count; matching IDs/motions are checked.
  Report95%one-sided upper quantiles and central95%intervals; these do not
  establish cross-object/population generalization or a formal Validation result.
- PROMISING requires BOTH nominal-minus-command and nominal-minus-zero upper
  quantiles<0and nominal-minus-command point estimate<0in each optimization seed.
  Otherwise UNPROMISING. No model/seed/axis/horizon selection or refitting.

One idleGPU4,2threads,<=3600s,<=2GiB; admission before every process. Own outputs
only, no other-user interference. Source/input SHA checked; retain failures.
This tests a statistical substitute for unavailable clone matching. It remains
one-object predictive exploration, without a novel-method or policy-utility claim.
