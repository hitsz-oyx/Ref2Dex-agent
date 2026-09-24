# P-20260924-expert-physical-cm-data

Date: 2026-09-24. Branch: `agent/cm-cross-object`.
Classification: Decision Probe.

## Question

Two randomized action families around the current self-trained airplane
actor produced weak task-aligned effects on most other objects, despite
the objects being physically graspable. Is the data distribution, rather
than merely the Cm architecture, withholding productive grasp transitions?

## Minimal protocol and boundary

Use the read-only official Inspire actor checkpoint **only as a simulator
data collector**. It may generate successful physical contact states and
randomized action transitions for Cm training/diagnosis. Its weights must
not initialize the final self-trained actor or appear in the final
Cm-on/off policy comparison. Any Cm trained from this data has explicit
expert-origin provenance; policy utility must still be demonstrated on a
self-trained actor.

Fix the same object-disjoint Cm split: airplane/mug/toothpaste for model
training, apple entirely held out from Cm fitting, normalization and model
selection. At contact states from the official actor, randomize wrist-z
`+0.1` vs `-0.1`, follow five executed simulation steps, 64 envs,
steps50..200 stride10, seeds186 train/187 heldout. The collector must pin
the official checkpoint SHA and motion split SHA, record actor role, exact
executed dose, object ID and five-step contact-supported object z.

Train-side continue gate: >=50 treated samples per object and pooled
contact-supported z contrast >=5 mm, with at least two of three objects
having positive point effects. If not, do not train another Cm on this
fixed action family; reconsider action distribution/target. If yes,
train a *shared* object-conditioned geometric/action model on train
objects only and evaluate apple against action-blind/state-only and raw
state+action controls. No offline accuracy result alone can prove policy
utility. A held-out run is only needed once the train-side gate passes.

One idle GPU per data run, <=30 min, <=100 MB output. Stop on source hash
drift, non-finite states, incomplete followup, dose mismatch, GPU conflict
or budget overrun.

## Physical result (Probe, not policy utility)

The source-actor and split hashes matched the pinned manifests; both runs
completed with exact `+/-0.1` wrist-z dose. Train run
`agent_expert_crossobject_randomized_train_s186_h5` produced 992 treated
contact rows: airplane 352, mug 336, toothpaste 304. Contact-supported
five-step object-z contrasts were +12.09, +8.67, and +11.45 mm respectively;
the object/step-stratified pooled effect was +10.73 mm, environment-cluster
95% interval [+4.57, +16.63] mm. The predeclared train-side continuation
gate passed.

The independent held-out apple run
`agent_expert_crossobject_randomized_apple_s187_h5` produced 971 treated
contact rows. Its contrast was +7.95 mm, environment-cluster 95% interval
[+3.71, +12.27] mm. Apple data remain excluded from any Cm fit,
normalization, or model selection. These randomized results establish that
the simulator contains action-dependent object motion in this *expert-actor
state distribution*. They do not establish Cm learnability, cross-object
prediction, or PPO benefit.

## Next decision test fixed before model fitting

Fit one shared object-conditioned six-region geometric Cm on the three train
objects only. Calibrate one-step action-to-executed-hand flow on those train
transitions only. Compare the same-capacity geometry model with action-flow
zeroed, a raw-state+action MLP, and a constant train-side action-effect
baseline. The primary held-out diagnostic is whether the geometric model
predicts a positive apple `+` versus `-` contrast near the randomized
7.95 mm effect, while action-blind predicts near zero. A constant effect
alone is insufficient: the geometry model must additionally separate
high- and low-benefit apple states in a frozen quartile uplift analysis,
and beat the raw-state+action control on that separation. If it does not,
label the shared geometric Cm `UNPROMISING` for this action/target setting;
do not infer policy utility or start PPO from offline fit alone.

## First shared-model result

The 300-update `agent_expert_crossobject_geometric_cm_s186187` run used only
train-object rows for hand-flow calibration, normalization and fitting.
On apple, geometric factual five-step translation EPE was 31.41 mm versus
zero-motion 37.76 mm and raw-state+action 39.16 mm. However, the geometric
counterfactual effect mean was only +3.48 mm versus the randomized +7.95 mm;
raw-state+action predicted +10.10 mm, action-blind exactly zero. The
geometric high-minus-low ranked treatment effect was only +0.86 mm with a
wide corrected environment-cluster interval [-12.50, +13.29] mm; raw +3.51 mm,
also uncertain. Thus factual prediction improved, but action-specific
cross-object utility was not established: `UNPROMISING` for this direct
supervised architecture, not a policy result.

## Architecture diagnostic fixed before further fitting

Test **one** causal-effect-oriented architecture on the train objects only:
an action-blind state baseline plus a separate signed treatment-effect head
with six-region geometry (versus equally budgeted raw-state and constant
effect controls). Use three leave-one-object-out folds on the original
train objects; fit hand-flow and normalization anew without each held-out
object. The gate to an independent apple test is positive predicted effect
on at least two of three held-out objects and a lower object-level mean
absolute ATE error than **both** raw-state and constant-effect controls.
If this gate fails, stop local architecture tuning and revisit data/action
target at a higher level. Apple s187 results above must not select epochs,
hyperparameters or model. If the gate passes, acquire an independent apple
seed for a single frozen evaluation.

The initial CI implementation used treated-row index modulo 64 instead of
the original simulator environment ID. A corrected rerun
`agent_expert_crossobject_geometric_cm_s186187_correctci` leaves fitted
parameters and point estimates unchanged and supplies the corrected
environment-cluster intervals above.

The leave-one-object-out run
`agent_expert_crossobject_causalhead_loo_s186_retry` completed after a
first run stopped on a sparse-quartile diagnostic (no change to fitting).
Predicted geometric action effect was -8.94 mm for held-out airplane,
+5.20 mm for mug, and -6.39 mm for toothpaste, whereas their randomized
effects were +12.09, +8.67, and +11.45 mm. Object-level ATE MAE:
geometric 14.11 mm, raw-state 13.61 mm, constant train-side effect
2.09 mm. Only 1/3 signs were positive. The fixed gate failed. Stop this
local architecture/action-target route; no new apple test or PPO run is
justified. This does not refute Cm generally: the fixed wrist-z action may
simply lack useful cross-object choice heterogeneity.
