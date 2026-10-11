---
schema: ref2dex.probe.v2
probe_id: P-20261011-official-reference-screen
experiment_id: P-20261011-official-reference-screen
date: 2026-10-11
task: trajectory-policy
branch: main
git_commit: 82bf96ad
claim_id: C3
hypothesis_family: HF-official-reference-coverage
probe_index_in_family: 1
seed_pool: probe
seeds: [191]
decision_changed_if_positive: use competent references as candidate multi-task teacher sources before WAM
decision_changed_if_negative: restrict teacher collection to competent references and diagnose failed tasks before imitation training
status: UNCLEAR
run_id: official-reference-screen-20261011-r1
---

# Current corrected12 references: official actor physical screening

User requests actual official DExplore performance on repository reference tasks,
temporarily deferring WAM/PW. Decision: which references provide competent teacher
behavior? Cheapest useful evidence: one seed191 wave per corrected reference,
8 synchronous frame0 environments, deterministic official Inspire checkpoint,
native GPU PhysX, full frames-1 controls, no early termination or interventions.
No training. These evaluation traces are training_allowed=false.

Inputs: corrected-expert-inputs-20261007-r2 manifest (12 motions/10 objects),
native-scale corrected assets, ballSize=1, existing inspire environment configuration.
Official checkpoint SHA8f6823db752288f1bddd6d042981d33514e29dac5a68e58726e76215fea6d553;
archived Torch2.0.1/rl_games1.1.4 with single RMS compatibility and padded64-row
actor calls. One separate process per reference to avoid carrying simulator state.

Record actual object/hand pose, velocity, q/dq, requested/applied actions and native
PD targets, unsigned sampled surface proximity, table support and footprint.
Report geometric lift>=3cm with hand proximity<=1cm and no table support, held5,
held45, maximum consecutive held frames, controlled placing+terminal settle15,
object reference position RMSE/rotation mean and terminal error. Placing is reported
separately: a clip whose reference ends elevated is not required to place. Determine
reference terminal support from the same native-scale mesh/table geometry, record it.
Native survival success_rate is never manipulation success. Independent CPU summary
recomputes metrics and controller mapping from the retained traces; probe counts
are correlated repeats of fixed starts, not independent generalization estimates.

Task screen PROMISING if >=6/8 hold45, else UNPROMISING; full task behavior requires
reference-compatible terminal completion and acceptable tracking, reported separately.
Aggregate UNCLEAR when task coverage is mixed. No formal scientific conclusion.
No checkpoint or seed selection based on results; retain every failure.

Budget20 GPUmin/1GiB new output; one inspected idle GPU3. Bound each native process
300s and total worker elapsed<=1200s. Recheck GPU before each launch; existing foreign
PID2378894 is an inspected Isaac Sim job active on other GPUs with130MiB idle contexts
here. Allow only that existing context with GPU utilization<=5% and prior total<=512MiB;
stop on new PID/context growth>512MiB, nonfinite, unexpected reset, source drift or
budget. Do not operate foreign processes. No Campaign/ref modifications or new branch.

## Execution compatibility note

First cubesmall attempt stopped before any controls: native default hard-object
oversampling repeats the same input path because its name contains `small`.
The one-motion guard treated this alias as a second reference. Preserve FAILED
manifest/log; not a physical negative result. Permit loader aliases only after
every resolved path, frame count and processed reference tensor matches the
single authenticated input. Retry cubesmall in a fresh directory; no changes to
native simulation, sampling defaults, checkpoint, seed or evaluation thresholds.

## Results and scope

All12 complete: first worker82bf96ad, subsequent f40e7313, cubesmall retry and
remaining workers62060ae2. Per-run manifests pin actual source Git/input SHA.
Summary/trace/controller/reference audit:
`outputs/trajectory-policy/official-reference-screen-20261011-r1/summary.json`
and `summary.md`; figure `comparison.png`. Each8env, seed191, complete frame0
reference. All12 references end geometrically supported at the source table.
No evaluation trace is training-eligible. Failed pre-control cubesmall attempt
retained alongside the single declared retry; no outcome-based run selection.

| Reference | Held>=45 | Controlled place | Object position RMSE cm | Rotation mean rad | Endpoint position error cm |
|---|---:|---:|---:|---:|---:|
| s3_airplane_lift | 8/8 | 8/8 | 7.46 | .464 | 12.99 |
| s7_airplane_lift_Retake | 8/8 | 1/8 | 44.27 | .812 | 145.30 |
| s9_airplane_lift | 8/8 | 8/8 | 7.11 | .505 | 2.20 |
| s7_apple_lift | 8/8 | 5/8 | 24.54 | 1.100 | 59.39 |
| s1_mug_lift | 8/8 | 6/8 | 24.28 | 1.174 | 22.30 |
| s1_toothpaste_lift | 8/8 | 8/8 | 6.58 | .602 | 3.97 |
| s1_alarmclock_lift | 8/8 | 7/8 | 5.37 | .831 | 3.43 |
| s1_cubesmall_lift | 8/8 | 6/8 | 8.42 | .858 | 7.26 |
| s1_cup_lift | 8/8 | 7/8 | 11.32 | .521 | 13.33 |
| s1_duck_lift | 8/8 | 2/8 | 11.74 | 1.242 | 24.78 |
| s1_phone_lift | 2/8 | 2/8 | 155.60 | 1.787 | 124.41 |
| s1_waterbottle_lift | 8/8 | 8/8 | 5.28 | .765 | 7.43 |

All96 briefly lift>=5frames;90/96 held45;68/96 controlled place. These sums are
inventory counts, not IID success-rate estimates. Under the predeclared grasp
screen11 references PROMISING, phone UNPROMISING. Aggregate teacher quality
UNCLEAR: complete reference quality is mixed, and placing does not establish
accurate object-pose tracking. In particular successful s3 placements still have
12.99cm mean terminal error. Sampled unsigned hand/object gap and table bounding
box support are geometric proxies, not exact contact/force/stability proof.

Independent CPU analysis verifies all source/checkpoint/mesh hashes (earlier
entrypoint/card versions authenticated from producer Git), raw-vs-native reference
poses, complete finite traces, action capture equality, native finger coupling and
every actual PD target (maximum absolute error0), recomputed held runs/placing and
position RMSE. These checks authenticate rollout semantics; they do not explain
phone/late-placement failure causes. No formal scientific conclusion or route
refutation. Initialization/generalization, repeated seeds and failed-task causal
diagnosis are deferred until required for teacher collection or a formal claim.

Decision: candidate competent sources s3/s9 airplane, toothpaste and waterbottle;
alarmclock/cubesmall/cup remain secondary candidates. Apple/mug track poorly,
Retake/duck have weak placement, phone is weak even at stable holding. Do not
label every official rollout a competent demonstration or use reference object
poses as if they were actual successful outcomes. User defers WAM implementation;
final target remains RL. This assessment starts no training or new collection.

Successful native in-process elapsed sum752.15s, failed pre-control run11.31s,
peak native device occupancy about18,366MiB (17.9GiB), ordinary utilization
31–50%, total new artifacts272MiB. GPU3 released; foreign idle context untouched.
Within20GPUmin/1GiB budget; legacy imports/startup add bounded overhead.
