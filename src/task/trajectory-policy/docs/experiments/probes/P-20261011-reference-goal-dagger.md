---
schema: ref2dex.probe.v2
probe_id: P-20261011-reference-goal-dagger
experiment_id: P-20261011-reference-goal-dagger
date: 2026-10-11
task: trajectory-policy
branch: main
git_commit: 3f07b895
claim_id: C3
hypothesis_family: HF-trajectory-policy-reference-goal-dagger
probe_index_in_family: 1
seed_pool: probe
seeds: [183, 184, 185, 186]
decision_changed_if_positive: retain reference-conditioned autonomous trajectory baseline and validate before PointWorld
decision_changed_if_negative: separate reference conditioning from learner-state PD correction and remaining R domain limits
status: PROMISING
run_id: reference-goal-dagger-execution-20261011-r1
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

Reference export: `outputs/trajectory-policy/competent-reference-20261011-r1/`;
raw-source/SciPy pose audit and native processed-reference comparisons pass
(native max error0). BC fit `reference-goal-bc-fit-20261011-r1`, seed183,
3000updates/28.10s GPU3: joint val tauMSE .00612650, PD MSE .02297259,
startup RMS .618mm; tau-only PD MSE .03224009. Fit audit PASS. Offline
improvement does not demonstrate closed-loop reference competence.

Physical `reference-goal-bc-execution-20261011-r1` completed542controls/543states
but failed only in result postprocessing (NumPy advanced index moved env axis).
Original FAILED manifest/log/trace preserved. `...-r1-recovered` symlinks the
original immutable trace and authenticates original input hashes against producer
commit cba9436b; recovery commit7f67d38f. Independent full feature/action/native-PD,
R/actor inference and physical outcomes audit PASS. Teacher hold4/4,place1/4;
GT-R hold4/4,place3/4; joint and tau-only hold0/8,place0/8. Calibration failed,
so status UNCLEAR, not negative evidence against reference conditioning. Joint
clipping3.506%, ref position RMSE .1793m, teacher .2310m. Retain all controls.

Decision: proceed with already specified one-round PD-query correction because
student autonomous grasp is absent and the user authorized DAgger. Collection
is not evaluation; teacher placement weakness is a limitation on teacher-query
quality and interpretation. Do not select a new seed or checkpoint based on
physical outcomes. No change to gates, R or budget. The one-round DAgger sequence is now complete.

Collection `reference-goal-dagger-rollin-20261011-r1`, seed185,91.26s GPU3:
4336 joint learner-state PD queries,3252 train/1084 validation, actual expert
intervention fraction .2442343. Intervened joint hold/place7/8; teacher and
GT-R4/4 each. Full independent query/native PD, masks, learner/R/features and
outcome audit PASS. This remains UNCLEAR collection, not autonomous success.
Trajectory SHA a5d0e5fe684d7f25f1ae2b250344020f57de15779d717764f179e0fe76786a9d.

Continuation `reference-goal-dagger-fit-20261011-r1`, seed183,3000updates/36.83s:
both arms warm-start the same reference joint checkpoint, R/statistics frozen.
Selected offline-control3000, DAgger250 by predeclared own validation criteria.
Independent training/statistics/PD label/objective selection/gradient audit PASS.
The producer manifest's checkpoint-selection description omitted the extra query
term; implementation and card already used it, independently recomputed by audit.
Forward metadata wording is fixed at1d274d0d; no measurements or selection changed.
`tau_only.pt` now means offline joint continuation (not tau-only supervision),
SHA4b646e964ec6d517859a2c9c710aa1e168e22569a8ebd536dab3ab583e8a6c64;
`joint.pt` means PD DAgger,
SHA7c2d1fe98a27c7c3ba72ba44a217c631aea5e46b0c74384a0cc7c9d4f692677e.

Autonomous `reference-goal-dagger-execution-20261011-r1`, seed186,98.13s,
producer commit3f07b895,24env/542controls/543states, teacher intervention OFF:

| Arm | Hold45 | Controlled placement | Median held frames | Clipping | Reference position RMSE | Reference angle mean |
| --- | --- | --- | --- | --- | --- | --- |
| Official teacher | 4/4 | 4/4 | 440.5 | 0% | .06968m | .37167rad |
| GT actual hand tau / frozen R | 4/4 | 4/4 | 436 | .0923% | .07411m | .44120rad |
| Offline joint continuation | 7/8 | 6/8 | 92 | .6919% | .75693m | .95631rad |
| One-round PD DAgger | 8/8 | 8/8 | 376 | 0% | .07574m | .49079rad |

DAgger passes all frozen gates (median376>=348.8, reference RMSE .07574<=.10452m,
angle .49079<=.65750rad), status PROMISING. All action/applied/native PD replay
exact; independent H/G/plan/actor/R max errors<1e-6 and all physical metrics
recompute exactly. Trajectory SHA
11e4be256fd15a9a620a9e63fb37b1ecf259333288f5afbb49da89a060f2701f.
Full audit at `reference-goal-dagger-execution-20261011-r1/audit.json` and
behavior curve `behavior.png` (all paths under `outputs/trajectory-policy/`).

Additional fixed query replay `reference-goal-dagger-fit-20261011-r1/audit_query.json`
preserves the pinned original audit. Query-train PD MSE warm .42311 -> DAgger
.12525 (offline-control .42188). Query-val mean9.50430 ->9.34694, dominated by
failed roll-in env23 (held15frames): per-env MSE22/23 warm .22687/18.78173,
DAgger .09611/18.59776, offline .23308/18.82995. Recovery from catastrophic
failed states remains unproven; this is not evidence that low mean PD loss alone
explains autonomous success. No relabeling of the failed env or exclusion from
selection. New outputs452.1MiB; GPU producer wall time334s plus bounded audits,
inside9GPUmin/600MiB and shared20GPUmin/1GiB envelope. GPU3 utilization ~16-48%,
training memory<=1.1GiB, native own memory18.1GiB. All own GPU jobs ended.
35 Task tests pass; no Campaign/Mission edits, branch change or push by agent.

## Decision / next

Retain G pose anchors and PD DAgger as a candidate autonomous baseline. Do not
add another aggregation round or refit R now: the fixed positive Probe provides
information to change the next decision. Before PointWorld or a formal claim,
freeze both checkpoints/inputs/gates for matched repeat/Validation, assess live
launch variation, multiple seeds and eventually held-out reference motions.
Do not equate single-wave8/8 placement with exact goal attainment, complete
classical DAgger efficacy or Cm-on/off training benefit.

## Limitations / future evidence

Original small single-wave/collection-seed source, sparse anchor goals, measured
history328 versus complete official observation. G pose fields are authorized
exogenous task inputs. Teacher PD queried on student states may be imperfect;
intervened future tau is deliberately unsupervised. Conditional comparison is
exploratory, not multi-seed validation, reference conditioning generalization,
Cm utility or RL completion. Formal matched validation follows a competent
baseline, and PointWorld remains deferred.
