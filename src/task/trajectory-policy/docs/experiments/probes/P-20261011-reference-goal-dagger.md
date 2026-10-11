---
schema: ref2dex.probe.v2
probe_id: P-20261011-reference-goal-dagger
experiment_id: P-20261011-reference-goal-dagger
date: 2026-10-11
task: trajectory-policy
branch: main
git_commit: 27ac34fe
claim_id: C3
hypothesis_family: HF-trajectory-policy-reference-goal-dagger
probe_index_in_family: 1
seed_pool: probe
seeds: [183, 184, 185, 186]
decision_changed_if_positive: retain reference-conditioned autonomous trajectory baseline and validate before PointWorld
decision_changed_if_negative: separate reference conditioning from learner-state PD correction and remaining R domain limits
status: PLANNED
run_id: reference-goal-bc-fit-20261011-r1
---

# Authorized human/object reference goals and one-round PD DAgger

## Motivation / Decision

User explicitly authorizes future human-hand and object planned references as G,
following official DExplore t+1/t+16 anchors, and permits DAgger-style training.
Earlier endpoint-G restriction was an agent assumption, now superseded; no
change to Mission core question/claim or Campaign. Actual future measured
hand/objects and expert actions remain labels or explicit oracle controls.
Does reference conditioning fix task-progress ambiguity? If remaining learner
states diverge, can teacher PD queries on those states repair closed-loop
behavior? One fixed reference BC followed, only if needed, by one learner-state
collection and fixed matched continuation is the cheapest discriminator.

## Frozen reference fit and execution

Same original audited official teacher source r2, env train0:16/val16:24/test24:32;
same fixed R SHA414411a0c16d92a9e43c3b7a40615f8995ec4ea3b66bcaca3dc1a0fcae10181f.
G408 concatenates reference human16 point positions/orientations and object pose
at t+1,t+16, expressed in current measured object frame, clipped to terminal
reference frame. These are pose fields from the pinned raw interaction motion,
verified against native processed hoi_data; no measured future robot q, force
or contact labels. Use pose anchors first, not the entire official observation:
its velocity/contact fields are not part of this initial adapter. H328 unchanged,
736->256Tanh->256Tanh->24x33 independent actor, train-only normalization.
Seed183,3000updates,Adam3e-4,batch512, fixed startup sampling and lambdaPD=.1 as
prior matched tau-only/joint fit. Per-control queries; frozen R gradients flow
into actor. Select own lowest validation total objective every250updates, no
physical/test model selection. Offline screen unchanged: startup first8 RMS
<=20mm, clipping<1%, finite, immutableR, genuine PD gradient.

Physical seed184,24env/542controls:teacher4,GT-R4,tau_only8,joint8, same donors
and frame0 reset. Calibration>=3/4 hold45 and controlled place for each control;
joint>=6/8 each, clipping<1%, medianheld>=80%GT-R. Also record full-wave object
reference position RMSE and rotation geodesic mean; require joint group position
RMSE<=1.5*teacher and angle mean<=1.5*teacher+.1rad for baseline readiness.
All->PROMISING; valid calibration with joint failure->UNPROMISING; calibration
failure->UNCLEAR. No claim exact goal/generalization or formal G causal effect:
input width/parameter count changes, and this is one fixed reference/seed Probe.

## Conditional one-round DAgger-style correction

If reference BC fails, retain its checkpoint and run one fixed roll-in seed185.
Query official teacher on every visited state before action application and
record its native PD target alongside learner requested/applied actual PD.
Teacher labels are native-controller targets for that live state, not future q.
Use joint learner with beta=.25 independent Bernoulli expert intervention per
control; record masks/actions/probabilities. Keep pure learner and teacher/GT-R
rows as diagnostics. Interventions are collection only; evaluation is fully
independent learner execution. Never label mixed-policy future hand as expert tau.

Warm-start both continuation arms from the same reference joint checkpoint:
offline-control uses original qualified teacher data; DAgger arm uses the same
original tau/PD data plus visited-state teacher PD loss through frozen R.
Fixed3000updates,batch512:256original +256queried learner-state rows, lambdaPD=.1;
control uses512original rows. Hold all normalization fixed at initialization.
Query split by whole rollout env; training joint rows16:22, validation22:24,
with original held-out source validation. PD-only correction has no tau label.
Use original val objective +.1*heldout query PD objective for DAgger checkpoint
selection; physical/test never select. No refitR or second aggregation sweep.
Compare both frozen continuations in24env/542control seed186 with same gates.
Report extra labels/data/compute; PD queries alone are not complete expert
trajectory distillation and do not establish full classical DAgger guarantees.

## Resources / provenance / stops

Remaining shared competent-route20GPUmin/1GiB budget: this round at most9GPUmin,
<=600MiB new outputs. Each fit<=300s, each native run<=300s, each audit<=120s.
One currently idle GPU (prefer3); pre-inspected <=512MiB idle contexts allowed
only with launch utilization<=5% and total prior memory<=512MiB, record current
PID/command. Stop on new foreign PID, context>512MiB, nonfinite/reset/controller
or reference drift. No other-user process operations. Repo-owned tmp/outputs,
main branch, explicit staged files, no push, no Campaign changes or ref additions.

## Results

Pending.

## Limitations / future evidence

Original small single-wave/collection-seed source, sparse anchor goals, measured
history328 versus complete official observation. G pose fields are authorized
exogenous task inputs. Teacher PD queried on student states may be imperfect;
intervened future tau is deliberately unsupervised. Conditional comparison is
exploratory, not multi-seed validation, reference conditioning generalization,
Cm utility or RL completion. Formal matched validation follows a competent
baseline, and PointWorld remains deferred.
