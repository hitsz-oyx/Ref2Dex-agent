# Frozen randomized task selection implementation

Implements the previously frozen task Probe. Online scores average the three
simple factual checkpoints; seven global control scores use FIT outcomes only.
Native/random/global/conditional policies share the geometry gate, clipping
mask, model inference and10-event/6-step spacing budget. All first native
episodes are followed independently to completion; no partial cohort selection.
Compact height/contact/active/done traces permit independent NumPy reconstruction
of stable success, retained success, drop and hold metrics.

Six meaningful checks pass: candidate clipping/control preservation, tie handling,
invalid random choice rejection, and three45-step-success/six-step-drop boundary
cases against the existing HoldTracker. New scripts compile; diff formatting clean.
Controller/geometry initialization captures and restores native RNG. Subsequent
model scoring neither samples nor reinitializes networks. Policy assignments use
an independent CPU generator; random correction uses separate per-env generators.

Before any new test outcome, the per-env random seed expression was clarified to
21000+1000*evaluation_seed+env, avoiding equal streams across neighboring envs in
different evaluation cohorts. GPU4became occupied by an unrelated process during
implementation. The resource card now explicitly permits another admitted idle
GPU under the same budget; use GPU5if still idle at launch. No process is stopped.

Classification: Decision. This experiment can stop the simple predictive controller
or justify broader task/method work. It does not establish journal readiness alone.
