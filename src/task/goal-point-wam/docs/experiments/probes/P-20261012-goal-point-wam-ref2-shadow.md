---
schema: ref2dex.probe.v2
probe_id: P-20261012-goal-point-wam-ref2-shadow
experiment_id: P-20261012-goal-point-wam-ref2-shadow
date: 2026-10-12
task: goal-point-wam
branch: main
git_commit: 23947ec4
claim_id: C3
hypothesis_family: HF-goal-point-wam-three-domain
probe_index_in_family: 16
seed_pool: probe
seeds: [289]
decision_changed_if_positive: run one bounded robot-simulation comparison with frozen R after source and embodiment review
decision_changed_if_negative: retain shadow interface and diagnose robot-domain or retargeter-distribution mismatch before closed-loop execution
status: UNPROMISING
run_id: wam-ref2-shadow-20261012-r2
---

# Frozen WAM -> R -> PD shadow admission

Decision: test whether the final10000 human-data WAM candidate can consume
measured robot H4 and a planned object endpoint and preserve the retained R's
command distribution. This serves the trajectory-policy self-trained robot
baseline subgoal; it does not establish Cm utility or a deployable actor.
The cheapest discriminating check is offline replay, rather than physics or a
new retargeter fit. User explicitly requested parallel ref2 integration.

## Frozen protocol

One seed289, idle GPU2, <=300s inference, <=100MB new artifacts. No training,
new checkpoints, real robot or remote training changes. Final WAM checkpoint
`outputs/goal-point-wam/goal-memory-ddp-20261012-r3/train/final.pt`, step10000.
Its original model.py is extracted from Git e8b81994 and checked against the
checkpoint's model SHA before importing it into an independent output/code
directory. The video route's edited model.py is not used for WAM inference.
Retargeter is the exact `competent-pd-fit-20261011-r1/tau.pt` authenticated by
the prior physical calibration manifest; R and normalization remain frozen.

Teacher source is `competent-pd-teacher-20261011-r2`, held-out env24:28,
query ticks0,1,2,4,8,16,32,64,96,128,192,256,320,384,448,518:64 windows.
History at startup repeats the actual initial state. Goal uses only the known
planned object pose t+24 from `competent-reference-20261011-r1`, paired with
the frozen512-point airplane mesh and32 fixed canonical point indices. Real
future hand is an evaluation/R-teacher label only; actor receives measured
H4 plus the object goal. No future q/contact or measured object motion input.

Native points are hand_base_link plus interleaved finger proximal/tip links.
Explicit map `[0,1,3,5,7,9,2,4,6,8,10]` converts native points into WAM
wrist/5-first-joints/5-tips order; active right hand is selected by name, absent
left hand masked. This is semantic role correspondence; human wrist/joint
origins and robot link origins are not anatomically identical. No learned or
fitted embodiment transform is claimed.

All measured history/hand and the planned endpoint use one current measured
object inverse. Independently transform actual future world points into that
frame, subtract transformed current points, reorder WAM then back to native,
and compare against R's existing displacement implementation: max<=1e-6m.

Shadow emits WAM and GT trajectories through the same live-state/frozen R and
native PD conversion, without applying the WAM commands. Report h1/2/4/8/24
and first8/full24 EPE, PD XYZ norm/rotation max-axis/finger max-coordinate p95,
native clipping and unclipped native action->PD roundtrip error.
Predeclared conservative screen: frame<=1e-6m, PD XYZ p95<=10mm, rotation
and finger p95<=.05rad, clipping<1%. Only all-pass permits consideration of
one small simulation comparison; thresholds are engineering screens, not
validated success guarantees. No thresholds selected from observed results.

Entry: `src/task/trajectory-policy/tools/audit/audit_wam_ref2_shadow.py`.
Reusable adapter: `src/task/trajectory-policy/src/trajectory_policy/wam_shadow.py`.
Contract tests: named hand selection, missing-point rejection, global rigid
coordinate invariance and no measured future-hand inputs.

## Results

Fixed shadow r2 completed64 windows in12.81s, GPU2 peak233.22MiB, no applied
WAM actions or new checkpoint. Original model source SHA matches checkpoint;
frozen-model repeated prediction is exactly equal in the diagnostic run.
Frame max3.6622e-7m, semantic reorder inverse exact, native unclipped PD
roundtrip max1.1921e-7: coordinate/index/PD wiring gates pass.

Robot-domain h1/2/4/8/24 EPE12.10/27.52/70.77/172.40/311.24mm,
nearest8 mean86.53mm, all24 mean205.51mm. Immediate requested-PD difference
versus GT-R: XYZ p95288.67mm, wrist max-axis p951.0948rad, finger
max-coordinate p95.7158rad. Native clipping is0 for both WAM and GT;
XYZ action scale1m makes zero clipping a weak usability criterion. The
predeclared PD screen fails, so no simulation or physical execution was run.
UNPROMISING applies to this zero-shot checkpoint/robot-input shadow screen,
not to the overall WAM->R research route or video point supervision.

Source teacher dt1/30s matches WAM and R. Repeated-start H4 (tick<3,
12 windows) has nearest8 EPE180.95mm/h1 31.71mm. Genuine H4 (tick>=3,
52 windows) has64.74mm/h1 7.57mm; later than tick8 has56.29mm/h1 5.23mm.
Failure persists with real history. Planned goal pose t+24 differs from actual
measured pose t+24 by p9590.98mm in position: the known reference is a valid
goal input, not an exact replay of human-data actual-endpoint labels.

Frozen R diagnostic (4.68s, same64 windows, no fit): replacing WAM's first8
with oracle GT leaves PD XYZ p95129.88mm; replacing only its tail16 with GT
leaves219.54mm. Both prefix and tail errors affect the immediate command.
Zero trajectory has305.46mm PD difference. Thus nearest-h1 plausibility alone
does not establish safety/usefulness of this R, which reads all24 horizons.
GT-R versus true source PD has p9510.59mm/.02377rad/.05516rad; the retained
prior GT-R physical calibration is still the appropriate empirical boundary.

The exact same cached scene with static object goal gives first8 EPE85.63mm
versus real goal86.53mm; full24 is239.68mm versus205.51mm. Goal intervention
changes first8 prediction by16.58mm but does not resolve this shadow screen.
Measured normalized scene velocity/acceleration peaks reach32/71 std units,
which flags input distribution risk without proving it causes the error.

Contract tests plus existing native PD target tests:5 passed (CPU index/pose
arithmetic); model computation uses GPU. Engineering r1 failed before model
prediction due a missing collate goal_object_slot field; r1 manifest is FAILED,
the adapter was corrected, and r2 is the actual scientific shadow observation.

Raw evidence: `outputs/trajectory-policy/wam-ref2-shadow-20261012-r2/` contains
manifest.json, result.json, shadow.npz, diagnosis.json, diagnosis.npz,
history_strata.json and authenticated code/model.py. Both runs total<1MiB.
Diagnostic entry: `src/task/trajectory-policy/tools/audit/diagnose_wam_ref2_shadow.py`.

Next minimum useful action: a separately declared robot-domain decoder and
observed-hand adapter calibration with scene/PTv3 and R frozen, original human
checkpoint preserved, and actual robot hand/PD supervision from the already
qualified teacher split. Do not truncate24-step R input or run sim simply on
the basis of human validation nearest8=13.46mm.

## Limitations / future evidence

GT-R's prior success only calibrates that source trajectory distribution and
controller, not arbitrary WAM outputs. A WAM-R failure can reflect WAM robot
domain shift, embodiment semantics, goal/history shift or R extrapolation.
It cannot automatically be attributed only to human-data trajectory EPE.
One nominal motion and64 fixed windows do not establish perturbation,
multiple motion, contact robustness, real robot safety or scientific efficacy.
