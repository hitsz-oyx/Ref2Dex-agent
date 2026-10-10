---
schema: ref2dex.probe.v2
probe_id: P-20261010-tau-geometry-tracking
experiment_id: P-20261010-tau-geometry-tracking
date: 2026-10-10
task: consequence-evaluator
branch: main
git_commit: 0003ab0
claim_id: C3
hypothesis_family: HF-reference-tracking-control
probe_index_in_family: 3
seed_pool: probe
seeds: [276, 277, 278]
decision_changed_if_positive: retain tau geometry plus closed-loop preload learning before high-level proposal integration
decision_changed_if_negative: isolate geometric reference error from missing feedback learning rather than refute tau representation
status: UNCLEAR
run_id: ref7_4-tau-tracker-train-20261010-r1
---

# Remove measured q and future-object reference from the execution controller

## Motivation / Decision Note

User confirms: preserve24-step11-point tau, remove true q_ref and future object
reference; ref7_4 motivates geometry retarget plus learned closed-loop preload.
No tactile input is introduced. This serves C3 execution and changes neither
Mission nor Cm claim. Geometry does not uniquely prescribe force/preload; the
controller learns physical feedback, not a unique inverse command label.

Question: after the measured-reference holding gate passed31/32, can tau-only
geometry provide a sufficient kinematic base for the learned tracker?
Classification Decision. Cheapest test: build geometry without later-q/object
labels, test the migrated frozen controller, then one fixed short PPO fine-tune
with a tau-only reward. Positive -> retain and later integrate proposals;
negative -> separate geometry approximation from feedback learning limits.

## Frozen input/action contract

Same native owned inputs, single s3_airplane_lift, CPU tensor exchange/GPU
PhysX. Future hand comes from the frozen measured teacher packet used in
[feedforward Probe](P-20261010-reference-tracking-feedforward.md).
Only initial live q/hand calibrate static wrist-root geometry. Wrist pose comes
from rigid alignment of palm/finger roots; six active fingers from bounded
coupled URDF FK fitting with reset/midrange starts. No later q/dq, future object
poses, teacher controls, forces or tactile labels fit the geometry reference.
Coupled geometry is approximate for contact-loaded joints; report fit error.

Student inputs: live q/dq, hand points, object pose/velocity, future tau,
geometry-derived next q error, previous residual. No true q_ref, future object
pose/error or tactile input. Action: tau-derived kinematic target + wrist
velocity feedforward derived from that geometry + learned12-coordinate bounded
residual, encoded into native full18 command. Actual measured q never supplies
a future PD base. FK reference fitting is independently frame-local; only past
wrist geometry chooses a continuous Euler branch. It caches geometry, not
command labels. No post-fit selection using later true q or execution outcome.

Actor/critic migrate the owned fixed checkpoint20073fc165e13f00f4e8f9f7b7f25d8b686faa3d6e5b98dbc8a6a9e867fe16b5
by dropping the12 future-object-error input columns. Preserve all other weights;
new input dimension897. Actor's joint-error fields contain only tau-derived q.
Old controller remains an independent privileged comparison arm. In training,
future q/object fields are not materialized for the student.

Reward also drops future-object reference: absolute hand tracking, object lift
following tau palm displacement, and existing actual-lift/force-pair holding
proxy; small residual penalty. Force proxy is training reward/statistics only,
not a new sensory input. No force or commanded-PD supervision. Keep native
PD gains, bounds, coupling, PPO hyperparameters and fixed final checkpoint.
128updates x32steps x64env, seed277. Smoke debug43 crosses2 PPO updates.

## Screen and discrimination

Frozen comparison seed276, then fixed-final comparison seed278, each64env,
542controls. Randomized16rows per role: privileged old reference tracker,
tau geometry with zero residual, tau closed-loop tracker, tau tracker with
reference shifted271frames. Shift affects both future hand and its derived
kinematic base; it is a dependency control, not an equally feasible task.
No baseline actor action enters student rows. These are descriptive rows,
not PhysX state forks or independent scientific seeds.

Container needs>=8/16 old-oracle hold45. Original strong holding threshold
remains>=90% archived481 (>=433), qualifying>=8/16 tau tracker, no recorded
intermediate loss in qualifying rows, median>=nominal+45, clipping<1%.
Report terminal hold, actual unsupported separation, tau-shift behavior and
command differences on the same live states. Shift sensitivity cannot prove
held-out tau generalization; later wrong-reference execution is diagnostic.
No original placement completion: the measured tau reference ends holding.

## Budget / stops

One idle physical GPU2, <=512MiB passive contexts, <=10% util, >=20GiB free;
preserve foreign processes, stop own run on new heavy foreign GPU use.
Geometry<=120s; smoke<=120s; frozen eval<=300s; train<=1200s; finaleval<=300s:
<=34 GPU minutes total, <=2GiB outputs. Nonfinite, input drift, requested/applied
mismatch, early terminal or deadline stops own run. Preserve failed artifacts.
No unbounded learning-rate/seed/model sweeps. A clear wiring bug may be repaired
and rerun into a new run_id within this budget.

## Limitations / future evidence

Single-motion GT tau upper bound, approximate coupling, fixed reset calibration,
short warm-started learning. No tau proposal quality, held-out motions/seeds,
Cm benefit, tactile dependence or raw controlled-placement claim. Geometry-fit
quality is not commanded force/preload correctness. Formal conclusions require
later matched Validation; Probe tags only PROMISING/UNPROMISING/UNCLEAR.

## Runs

Pending. Actual implementation commit is captured in run manifests.


Geometry r1 (0a79260) completes300iterations/14.31s onGPU2 (~337MiB).
Coordinate RMSE2.47mm, fixed roots<0.0001mm, thumb tip point RMSE13.80mm;
other fingertip point RMSE0.51–2.79mm. Coupled geometry is not loaded-q recovery.
Warmstart weights carry prior oracle training information; only current student
execution and fine-tuning inputs/reward remove that privileged information.

Frozen comparison r1 (75ff26e,seed276): oracle median484/near43316/16/terminal16;
tau nominal median0/near0/terminal0; migrated tau tracker median478/near13/16/
terminal13; tau shifted median207/near0/terminal15. Do not use end-hold alone
to infer tracking correctness. Actual student897 input reconstruction maxerror
3.81e-6, command maxerror2.38e-7, nativePDtargeterror0, reset alignment bitwise.
The strong screen remains UNPROMISING because tracker clipping3556/8672=41.01%.
3547thumb-pitch and44thumb-yaw coordinate overdrives; no wrist clipping.
The original threshold is not relaxed; saturated commands are not concealed by
preprojecting desired targets or changing which controls count as clipping.

Decision before training: retain geometry + feedback hypothesis; wrist works
without measured q_ref but the old preload is incompatible with the new thumb
base. Use the single predeclared128update seed277 fine-tune with a smooth
native-command excess cost: reward minus0.20*max(abs(requested-clamped)).
This is computed before native dispatch from intended commands, with no future
labels or added tactile state. Clamped effective control is unchanged; the
penalty favors valid equivalent preload over unnecessary overdrive. All other
PPO settings stay fixed. Record this targeted reward change before training;
geometry fitting is not redone or chosen by rollout result.
Frozen artifacts: outputs/consequence-evaluator/ref7_4-tau-tracker-frozen-{eval,audit}-20261010-r1/.
