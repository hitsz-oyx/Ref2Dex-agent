# Standalone Contact-Response Research State

Updated: 2026-10-02. Worktree `Ref2Dex-agent-contact-response`, branch
`agent/contact-response-cm`. This branch is an independent research track.
The user's original worktree progresses independently (observed on
`agent/cm-executable-options` on2October); this session does not switch its branch
or modify its files. Its initial changes were preserved. Its checkpoints, prior results and motion inputs are read-only.

## Current objective

Under the user's authorization to autonomously extract ideas, experiment and
write a journal-level paper, investigate whether reliably measurable physical
action effects can be learned and ultimately improve dexterous decisions.
Existing literature means generic particle world models, value awareness,
action recovery and controllability representations cannot be claimed as new.

## Confirmed observations

- Physical pulse/repeat experiment completed16/16runs,384windows across four
  actor/evaluation panels and three airplane motions; code466e80d. Five/ten-step
  position contrasts7.15/18.83mm exceed observed zero-repeat contrasts0.10/0.44mm.
  Vertical effects have mixed signs; the full predeclared screen is UNPROMISING.
  This distinguishes physical response from beneficial lift, not policy utility.
- A separate documented vector-prediction screen completed9/9fits with fixed
  three optimization seeds,1000updates each; code44f22ea. Differential contrast
  errors10.11/26.99mm do not beat factual or global-fit-mean controls. Sign
  agreement32.6%/31.1% fails its75%gate. This implementation is UNPROMISING.
- All learned test panels were part of the earlier physical pilot. Results are
  exploratory reused-data evidence, not independent Validation.
- Current-geometry audit:318/384old triggers are within20mm. Actor287/motion1
  has no near examples in either old test panel (gaps55–105mm), exposing a
  systematic geometry shift. This does not prove the cause of learning failure.
- Archived one-step factorization screen (`bbd5ba4`) completed53.64s. Predicted
  hand-motion effect RMSE15.07mm versus native-command16.44mm near the object;
  its8.33%gain fails the fixed10%gate, UNPROMISING. The separately reported
  available PD-target-motion variant scores14.60mm (11.18%gain), a secondary
  hypothesis requiring fresh causal testing. Future-hand oracle is privileged.
-30factorization RMSE metrics independently recomputed from saved predictions;
  feature/acquisition tests pass. No formal scientific conclusion is upgraded.
- Exact-null inverse-recovery screen (`33d1085`) completed12fits;32896states
  pass exact PD-target invariance. Full inverse null-MSE1.0019does not beat the
  population floor1and null response is below factual-only in everyseed.
  UNPROMISING for the proposed interference mechanism; abandon this explanation.
  Hard command canonicalization enforces null invariance but is not a new claim.

## Current decision

Freeze these exact designs and preserve their failed gates. Do not increase
updates, choose a better seed, or start PPO from a failed predictor. The frozen
nominal/command/state models are now tested on fresh geometry-conditioned
plus/minus wrist interventions in all three world axes. No refitting or model
selection is allowed in this transfer screen. This is a Decision Probe, not
policy Validation or new-object generalization.

Fresh `P-20261001-fresh-causal-transfer-r1` completed32/32physical arms and384/384
geometry-conditioned windows, then FAILED the strict pre-intervention match:
seven arms in panelt286_s491 exceed joint/orientation limits, including zero_b.
Do not widen the tolerance, exclude arms, or report a model winner. Scientific
transfer label UNCLEAR; frozen nominal model remains untested causally. All36
native output files and102input hashes independently verified after closure.

Randomized trial `P-20261001-randomized-effect-risk-r2`, code510a00e, COMPLETED:
3072/3072new windows, four768-environment panels,137.84s/8.29MiB. The r1device
alias failure (35.85s) is preserved. All assigned propensities and action changes
are audited. Nominal-minus-command risk difference+0.1223mm², upper95%=1.0126;
nominal-minus-zero-0.3261mm², upper95%=1.1242. All three gates fail, UNPROMISING.
This is an identified risk DIFFERENCE, not absolute effect RMSE. No subgroup
or seed replaces the failed gate. Stop direct control use of these archived
factual predictors; their causal advantage is not supported by the new trial.

Direct randomized learner (`2075314`) COMPLETED:3072fit and3072newtest windows,
initial seeds494/495,IIDseven-arm assignment. All five gates fail, UNPROMISING:
direct-minus-factual risk+23.9773mm²,upper95%=44.2011; every seed worse.
Compute-matched/global/zero controls also not beaten. Six nuisance-fold scales,
16fit model/data files,135test input hashes, predictions and risk/bootstrap arrays
pass independent CPU/NumPy audits. Fit/test30.31/134.49s;3.41/11.89MiB. Stop
this exact learner; no further updates or selected subgroups.

POST-HOC candidate evidence: the simpler factual1000-update control has
factual-minus-zero/global risk-3.4359/-3.1576mm²,upper95%-1.1063/-1.2216,
negative in each fixed seed. This does not upgrade the failed direct primary
gate. No robust benefit of3000versus1000updates appears. The baseline is not
novel and has not demonstrated task benefit. Next frozen Decision Probe tests
equal-weight factual selection against actor/random/global controls on new
seeds496/497, full first-episode retained success and drops, bounded corrections.
No PPO yet.13focused learner checks and6task-controller checks pass.
Task Probe `P-20261001-randomized-task-selection-r1` COMPLETED, code c0ec955,
3072first episodes,465.87s/44.13MiB on admitted GPU5. All four gates fail:
conditional/actor retained success0.803%/1.166%; difference-0.364pp. Stable
success14.004%/11.509%does not replace retention; drop increase+2.859pp,
upper95%+5.478fails2ppgate.3072trajectories,30440policy choices,142protected
inputs independently audited. End this exact greedy controller/protocol.

Task-specification BLOCKER: immutable reference lift intervals last at most
36/28/25frames, below45-step hold standard, and all return to table beforeend.
Full-episode nonreturn conflicts with perfect reference tracking. This should
have been checked before freezing the last gate; preserve its failure, do not
claim that every counted return is accidental loss or that all models are useless.

Next design: explicitly synthetic90-row holding plateau at each FIRSTlift
interval's label-only peak. Generation COMPLETE (`b2b26e0`), initial/remainder
preserved, generated files4.68MB. A new actor-only feasibility Probe is frozen:
actors286/287,new seeds498/499,four192-env panels;75consecutive held steps within
inserted phase, pooled>=10%and each motion>=5%. Physical collection COMPLETE (`40e73f2`):768 first
episodes,5/768 retained75 (0.651%),19/768 stable45 (2.474%), all four gates fail.
UNPROMISING for direct transfer. Original reporting job FAILED on duplicate
keyword fields; preserved. Separate identical-score correction (`3f4ae77`) audits
all768 actual-progress trajectories without new physics; all protected inputs
unchanged. Baseline curriculum COMPLETED (`e770d4b`):300 NEWepochs/460800 interactions,
1120.95s/106.30MiB, final-only evaluation500/5010/384 retained75 and0/384 stable45.
All178 inputs unchanged; exact continuation UNPROMISING and stopped.
Native reset audit (`da39660`) changes raw finger joints but preserves sampled
contact flags; no causal explanation of failure follows. Static unmodified PD
(`1e38cf6`) COMPLETED192 elevated90-step trajectories,0/192 necessary height/proxy
retained75. Original table-localZ clearance INVALID_TABLETOP_AXIS; immutable
original gate and inputs retained. Separate68bf178 full-pose correction uses actual
thin localY normal; correct initial clearance307/131/154mm, still0/192; post-hoc
only. Wrist joint5 is continuous; geometry helper +/-pi fallback is not its bound.
Current distinct Decision Probe is bounded fixed finger preload0/.05/.15/.30rad,
new seeds504/505, corrected tabletop geometry; no more PPO of the failed recipe.
Question: can added closure establish mechanical retention before a pickup teacher?
Run P-20261002-finger-preload-feasibility-r1 COMPLETED (`5d7c375`),192 fresh
90-step trajectories,76.99s/9.43MiB, all189 inputs unchanged. Retained75 counts
0/0/6/13 out of48 per dose; motion1 all zero. All candidate multi-motion gates
fail, UNPROMISING; stop this exact static family. Positive suspended-start cases do not establish pickup. A distinct original-frame0
reference controller (`d15ef0f`) with post-hoc.30rad candidate finishes192trajectories
on506/507; unchanged0/96 vsclosure5/96 strictretained75. All fullgates not passed,
UNPROMISING. Original parent resource FAILED after one panel; verified continuation
(`3c8c78f`) runs ONLY missing panel on freeGPU1, all198 inputs unchanged, no reruns.
Active parent time119.23s. Geometry-only SECONDARY on closure motion1 is32/32,
strictforce75 only5/32. Native mass2.5936g, gravityon, weight.025443N; historical
.1N force threshold equals3.9303weights. This does not yet establish false negatives.
Prior static POSTHOC component audit also separates geometry/strictforce counts.
Prospective support witness (`49cf8b3`) COMPLETED102.35s/17.03MiB,193inputs
unchanged,192fresh202-tick trajectories508/509. Predeclared motion1 all64prior
geometry75; all32controls retain lategeometry, all32release lose it and retract
>=15cm. Heightcontrast147.113/144.379mm, all10gates pass: PROMISING mechanical
witness only. Strictforce75 misses17of these64geometricallyheld rows. No old
failedgate upgraded; no forceclosure/learnedpolicy/Validation claim. Motion0/2
remain unsuccessful. Next priority is a freshly trained observation policy on
verified holding physics, withprospective physical105-tick retention andcurrent
force proxy retained assecondary. Scratch BC9c0a218then COMPLETED149.84s/44.10MiB,all200inputs unchanged: physical105andstrictforce75both0/192;motion1 0/64,UNPROMISING. Late wrist-target RMSE~171mm, no normalization clipping onmotion1; descriptive posthoc diagnostic only. Stop this exactfit. Next distinct Probe is one fresh on-policy aggregation513/514, independenttest515/516, fixed2000updates; oldtest511/512never used in fit. No active simulation/training at this checkpoint.
Original-worktree data remain read-only and all outputs remain isolated here.

## Paper and evidence

- [Results](research/20261001-contact-response-results.md)
- [Literature/novelty screen](research/20261001-contact-response-literature.md)
- [Vector-learning decision](decisions/D-20261001-vector-response-learning.md)
- [Factorization results](research/20261001-actuation-effect-results.md)
- [Fresh causal-transfer decision](decisions/D-20261001-fresh-causal-transfer.md)
- [Fresh matching failure](research/20261001-fresh-causal-transfer-results.md)
- [Randomized estimand decision](decisions/D-20261001-randomized-effect-risk.md)
- [Randomized trial results](research/20261001-randomized-effect-risk-results.md)
- [Direct randomized results](research/20261001-direct-randomized-response-results.md)
- [Task-selection decision](decisions/D-20261001-randomized-task-selection.md)
- [Next task Probe](experiments/probes/P-20261001-randomized-task-selection.md)
- [Task result](research/20261001-randomized-task-selection-results.md)
- [Hold-task decision](decisions/D-20261001-reference-hold-task.md)
- [Next hold feasibility Probe](experiments/probes/P-20261001-hold-plateau-substrate.md)
- [Holding curriculum result](research/20261001-hold-plateau-curriculum-results.md)
- [Static mechanical result and corrected geometry](research/20261002-static-hold-feasibility-results.md)
- [Finger-preload result](research/20261002-finger-preload-feasibility-results.md)
- [Finger-preload decision](decisions/D-20261002-finger-preload-feasibility.md)
- [Frame0 tracking and measurement result](research/20261002-frame0-tracking-feasibility-results.md)
- [Next support-removal witness](experiments/probes/P-20261002-support-removal-witness.md)
- [Support witness result](research/20261002-support-removal-witness-results.md)
- [Scratch policy result](research/20261002-observation-hold-baseline-results.md)
- [Novelty update](research/20261002-contact-response-novelty-update.md)
- [Current manuscript](../paper/manuscript-v5.tex) and [PDF review copy](../paper/manuscript-v5.pdf)

The manuscript reports actual methods and negative pilot results. Journal
readiness is NOT READY: distinctive method, task-level matched policy benefit,
independent object/task/hand evidence and hardware evidence remain missing.
Original North-star scientific claims are not upgraded by this branch's pilots.
Resource and external-data protection boundaries remain in CAMPAIGN.md.
