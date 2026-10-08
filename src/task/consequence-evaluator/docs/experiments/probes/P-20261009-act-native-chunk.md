---
schema: ref2dex.probe.v2
probe_id: P-20261009-act-native-chunk
experiment_id: P-20261009-act-native-chunk
date: 2026-10-09
task: consequence-evaluator
branch: main
git_commit: 79cdbf3
claim_id: C3
hypothesis_family: HF-consequence-act-proposal
probe_index_in_family: 1
seed_pool: probe
seeds: [282]
decision_changed_if_positive: run a native GPU open-loop chunk behavior screen before returning to GT candidate ranking
decision_changed_if_negative: keep the reactive policy as the only behavior baseline and defer PointWorld/chunk candidates
status: UNCLEAR
run_id: gate1-gpu-serial-engineering-20261009-r30
---

# Does a one-shot native 24-step proposal have a usable behavior contract?

## Decision Note

The synchronous native GPU group preserves the reactive policy's full grasp
behavior, but its contact-stage replay is not an exact twin and candidate
effects are not uniformly above the zero-pair displacement. The next question
is whether freezing a complete action chunk at a decision state removes the
reactive observation-to-action amplification enough to make a later candidate
comparison interpretable. This probe therefore trains a small ACT-like decoder
from one raw observation `H_t` to the next 24 native controls and first audits
action-space predictability. It does not train the consequence evaluator and it
does not produce a GT-value claim.

The target is the `action[t:t+24]` field captured at the native
`pre_physics_step` boundary (`native_post_noise_pre_physics_control`). The
`residual_plan` field is excluded: it is a requested residual contract, not the
native action chunk. The simulator remains responsible for Inspire's downstream
PD/coupling conversion. Windows never cross an episode boundary and use
`history[t]` with `action[t]`.

## Data and split

The primary engineering source is the labeled hold audit
`outputs/consequence-evaluator/labeled-hold-audit-20261008-r1/`: 21 clean
`expert_success` airplane episodes, all `s3_airplane_lift` under the frozen
`airplane_base` route. Its manifest is `training_allowed=false` by design, so
any use is explicitly `--allow-audit-only`, remains `engineering_only`, and
cannot enter evaluator training. A deterministic episode holdout is used for
the offline screen. The broader labeled continuous train/val/test source is
kept available for a later route-conditioned proposal, but mixes expert
checkpoints and motions and is not the first native parity source.

The native checkpoint `running_mean_std` is frozen when supplied, including the
native `sqrt(var+1e-5)` scale and the actor's `[-5,5]` normalized-observation
clip. Output remains raw 18-D normalized control in `[-1,1]`. The first r1--r5
artifacts predate the clip metadata; they remain reproducible engineering
artifacts but are not treated as the corrected normalization contract.

## Probe and stopping rules

First run the CPU/data-contract tests and a bounded imitation fit (one GPU,
less than 900 s, no checkpoint overwrite). Report MSE/MAE, first-action error,
chunk error, and comparison with a train-mean chunk. A lower offline error only
marks the route `PROMISING` for an engineering behavior screen; it is not task
success. Stop if the source contract drifts, the audit-only provenance is
silently treated as trainable, or the GPU becomes occupied by a foreign task.

Before any candidate ranking, integrate the frozen proposal into the existing
native GPU group as a behavior-only screen: reset through the native path,
generate one 24-step chunk from the pre-action observation, feed raw actions
open loop for the chunk, and compare env0 hold/lift/contact traces with the
reactive teacher. A first-action match does not establish chunk behavior. Only
if this parity gate is acceptable will the group/noise harness be extended to
candidate chunks.

## Limitations

The hold audit is same-motion engineering evidence, not an independently
qualified training corpus. Contact-driven branches produce nonzero action and
observation variation even under nominal controls, and overlapping windows can
inflate apparent sample count; episode holdout and first-action/open-loop
metrics are required. The proposal has no execution model, PointWorld future,
or evaluator ranking yet.

## Results

The task-local contract tests pass (`5 passed`). On the 21 clean hold-audit
episodes, a stride-8 fit with 17 training and 4 held-out episodes reduced
action-chunk MSE from a train-mean `9.38e-4` to `1.61e-4`; first-action MAE was
`0.01090` in the held-out episodes. This is an action-space `PROMISING`
engineering signal only. The source remains audit-only and the output is
`engineering_only`.

The native GPU behavior screen used four roles in one 542-step process:
reactive teacher, ACT chunk, reactive repeat, and ACT repeat. With
`open_loop24`, the ACT role reached `0.7939 m` maximum lift and held 478 frames,
versus 0.8134 m and 483 frames for the reactive teacher; both reached the
controlled final geometry. Each 24-step proposal was generated once and its
raw 18-D controls were fed unchanged through the native pre-physics boundary.
The packet is
`outputs/consequence-evaluator/act-native-chunk-engineering-20261009-r2/`.
That packet predates the later repeated-chunk arm cleanup; the teacher and
single ACT-role behavior numbers remain engineering observations, while its
ACT-repeat drift is not used as a solver-noise estimate.

The final-chain `receding8` schedule failed the behavior gate: the same
checkpoint held zero frames and never lifted (r3). A full-fit all-episode
upper-bound imitation reduced offline MSE to `4.53e-5`, but its receding8
screen also held zero frames (r5). Thus the failure is not explained by the
held-out fit alone: re-planning every eight steps moves this direct imitation
proposal out of its action/state distribution before grasp. These packets are
`engineering_only` and do not refute the open-loop24 proposal or the value
route.

### Decision Note

Open-loop24 is retained as the only current ACT-like behavior baseline. The
4--8-step receding schedule is closed for this direct imitation checkpoint;
do not attach PointWorld candidate ranking or formal Gate1 claims to it. The
next affordable route is to use a recorded nominal chunk with matching prefix
provenance, or to collect a reactive-deployment distribution before retraining
a route-conditioned proposal. Y, reference-bank labels, and policy weights
remain unchanged.

## Query-window candidate follow-up

The first candidate implementation used the original GPU PhysX/GPU tensor
pipeline and kept the reactive policy prefix through tick47. It then broadcast
one frozen chunk to four roles and added the positive/negative residual only at
tick48. The runner now records and checks the prefix/replay controller hash,
physics hash, backend, seed, source packet hash, and native action semantics.

The learned proposal is not yet a valid nominal baseline on the reactive query
state. The r6 full-group screen (the pre-clip r1 checkpoint) had zero lift for
the broadcast ACT nominal. With the same reactive prefix as the successful r9
GPU group, r7/r8 reached only about `0.012 m` and had pre-query zero-pair drift
up to `0.0786 m` in object pose, about `20` in q velocity, and `26--32` in
history. Offline replay gives the same diagnosis: the r1 proposal differs from
the reactive query controls by max `0.0567`/MAE `0.0147`, while it matches the
successful ACT-state chunk at about `5e-7` max error. This is a state-distribution
gap, not evidence that the native feeding boundary is wrong.

The corrected RMS-clipped fit is in
`outputs/consequence-evaluator/act-native-chunk-engineering-20261009-r10/`;
its held-out MSE is `1.77e-4`, but the 542-step behavior screen r11 again had
zero ACT lift. A recorded ACT chunk from r2 (r13) reached only `0.150 m` and
held 7 frames. A recorded reactive nominal chunk from the successful r9 prefix
(r15) reached `0.240 m` and held 11 frames, while its zero pair still had
`0.0785 m` pre-query object drift and `0.338 m` post-query object displacement
p95. The explicit zero-pair and nominal-behavior gates therefore both fail;
the candidate effects in r6--r15 are engineering diagnostics and are not
interpreted as candidate value evidence.

The next probe must first pass both gates with a recorded prefix/chunk and
source hashes. If a nominal chunk cannot retain behavior or a zero pair cannot
stay within the declared noise tolerances, the ACT candidate route remains
closed and no PointWorld/evaluator fitting or formal Gate1 ranking is started.

## Multi-zero GPU group follow-up

To test whether a larger same-process group supplies a usable noise floor, r17
ran the native `gpu_physx_gpu_pipeline` contract with 16 environments for the
72-step engineering window. The fourteen nominal roles were env0, env1, and
env4--env15; env2/env3 were reserved for candidates. The saved packet is
`outputs/consequence-evaluator/gate1-gpu-group-engineering-20261009-r17/group.pkl`.
Its env0 nominal reached only `0.2536 m` and held 10 frames, so this launch did
not restore the full GPU grasp behavior.

The follow-up r18 replayed a recorded reactive nominal chunk from that exact
16-env prefix, selected env1/env4 as the reported zero pair, and added the
positive/negative residuals only after tick48. The packet is
`outputs/consequence-evaluator/gate1-gpu-group-engineering-20261009-r18/group.pkl`.
The selected nominal roles reached `0.2743 m`/12 frames and `0.2617 m`/10
frames; positive and negative candidate roles reached about `0.253 m`/13 and
`0.259 m`/9 in the same bounded window. The behavior gate was not evaluated
for a 72-step packet and `candidate_calibration_valid=false`; these numbers do
not show behavior parity or candidate success.

The selected pair had small pre-query p95 drift (object pose
`1.83e-5`, hand `3.29e-6`, q position `8.89e-6`, q velocity `1.43e-3`, and
history `3.91e-5` in their native units), but max/contact spikes remained. The
runner now records a common-pair p95 summary rather than combining the best
pair independently for each field, and requires at least 80% of the common
zero pairs to meet the field tolerances before a multi-zero gate can pass.
Offline recomputation of the saved r18 packet gives 62/91 all-field pairs
(68.1%), so its multi-zero gate is diagnostic/failed. The current packet still
has selected-pair post-query ratios only; a future multi-zero candidate probe
must add post-query query-relative noise for every zero pair before it can
support a candidate comparison.

This keeps the native GPU group as an engineering container, but closes the
16-env candidate interpretation for this launch. No formal Gate1, GT scoring,
evaluator fitting, Y/reference-bank change, or policy-weight change follows
from r17/r18.

## Four-env behavior and candidate follow-up

The original four-env GPU group was rechecked after the pair-selection and
audit-metadata changes. In r19 the env0 nominal reached `0.8262 m`, held 481
frames, and passed the full native behavior gate; this keeps the four-env GPU
process as a usable behavior container. The env0/env1 pair still had
pre-query p95 drift above the declared robust tolerances for hand/q/velocity,
so behavior parity did not become a same-state contract.

Using r19's tick0--47 prefix and its recorded nominal tick48--71 chunk, r21
ran the four-env candidate window with the selected zero pair env0/env1. The
nominal role reached only `0.2607 m`/11 frames in this fresh 72-step process;
the behavior gate was therefore unevaluated and `candidate_calibration_valid`
was false. Candidate-vs-zero post-query p95 ratios were positive/negative
`2.52/1.10` for object pose, `2.06/1.33` for hand points, and `5.20/1.72`
for joint position, but only `1.07/0.79` for joint velocity and `1.14/0.69`
for history; contact ratios were also below one for both arms. The effect
margin gate failed, so r21 remains an engineering noise probe rather than
candidate value evidence. Packets are in
`outputs/consequence-evaluator/gate1-gpu-group-engineering-20261009-r19/` and
`...-r21/`.

The native GPU group can preserve the reactive behavior in a successful
four-env launch, but the same prefix/chunk does not provide a stable nominal
candidate window across fresh processes. The next implementation step is to
record post-query zero displacement for every zero role and compare each arm
against that distribution, or redesign the replay contract; formal Gate1 and
evaluator fitting remain stopped.

## Physical-bank GT-value audit

The engineering packets were then scored with the frozen physical-reference
bank and TCC phase encoder using
`tools/audit/audit_gpu_group_gt_values.py`. This audit accepts engineering
packets only, records the current-progress spread, and never feeds them to the
strict score worker. Its output is
`outputs/consequence-evaluator/gate1-gpu-group-engineering-20261009-r22/gt-value-audit.json`.

For r19, the four role values were
`[0.041369, 0.042313, 0.017920, 0.041562]`; for r21 they were
`[0.042281, 0.044033, 0.040786, 0.016099]`. In both packets the progress
start range was `4.90e-6`, above the strict `1e-9` same-state tolerance. The
raw argmax was the zero-repeat role in both cases, while the frozen `0.01`
deadzone selector chose baseline in both cases. This is useful contract
evidence—the current native GPU residual arms do not produce a reproducible
non-baseline GT-value choice—but it is not a Gate1 utility result because the
same-state and nominal behavior contracts are still not jointly satisfied.

## Post-query zero-noise audit

The missing multi-zero diagnostic was added as the read-only
`tools/audit/audit_gpu_group_noise.py` tool (`110cd13`). It validates the
candidate-role packet, the shared prefix controls/done signals, native 30 Hz
timestamps, and all zero-role pairs before reporting query-relative state
displacements. The output is
`outputs/consequence-evaluator/gate1-gpu-group-engineering-20261009-r24/noise-audit.json`.

For r18 (14 nominal roles, 91 pairs), the median zero-pair post-query p95 was
`0.2522 m` for object pose, `0.2621` for joint position, `5.472` for joint
velocity, and `0.8532` for history; the corresponding p90 values were
`0.5375 m`, `0.5151`, `9.851`, and `1.4236`. Candidate-vs-zero incremental
p95 divided by the all-pair median was only descriptive: the positive and
negative object-pose ratios were `1.96`/`1.74`, while history was `1.55`/`1.81`.
For r21 there was only one zero pair; its object-pose noise p95 was `0.2579 m`,
with positive/negative ratios `2.02`/`0.86` and history ratios `1.21`/`0.87`.
These ratios use candidate incremental displacement over the all-pair median;
they are not the runner's selected-pair effect-margin statistic and do not
establish candidate ≫ solver noise.

The same audit on the full-behavior r19 packet (one zero pair) gave object-pose
noise p95 `0.1195 m`, with positive/negative ratios `4.28`/`1.95`; history was
`1.42`/`0.81`, and joint velocity was `1.27`/`0.93`. Thus even the one launch
that preserves the native hold does not separate both residual arms across all
physical fields. Its diagnostic is in
`outputs/consequence-evaluator/gate1-gpu-group-engineering-20261009-r26/`.

The audit therefore confirms that the proposed synchronous group has a
measurable noise floor, but the current residual effects are not uniformly
separated from it. No formal Gate1, evaluator fit, PointWorld ranking, or
reference/Y change follows from r24.

## Fixed actor-batch contract follow-up

To isolate a possible group-size-dependent actor GEMM shape, r25 temporarily
changed the group runner to infer env0 with the verified single-env/64-row
actor call and broadcast that action. The native backend and seed stayed fixed
(GPU PhysX/GPU tensor pipeline, seed282, four environments, 542 steps). This
did not preserve the behavior contract: the baseline reached only `0.0473 m`
and held 4 frames, versus r19's `0.8262 m` and 481 frames under the prior
group execution. The packet records `actor_inference_batch=64` and is
`outputs/consequence-evaluator/gate1-gpu-group-engineering-20261009-r25/group.pkl`.

The fixed-batch change was reverted in `b0c4b33`/`ecc4def`; r25 remains a
negative engineering probe. Its contact-stage divergence means actor batch
shape and PhysX scheduling were not cleanly separable, so it cannot explain
the old r17/r19 difference or justify a new backend. The native GPU group
remains an engineering container only; a same-state fork or a statistically
powered repeated-baseline/candidate design is still required before Gate1.

The old group contract was then repeated for seed283 (r27), without the
fixed-batch change. It also failed the full behavior screen: env0 reached only
`0.0891 m` and held 5 frames, while the negative residual role reached
`0.5663 m`/398 frames. The zero pair's post-query object-pose displacement p95
was `0.8247 m`; positive/negative candidate-vs-zero incremental ratios were
`0.82`/`1.29` for object pose and `1.11`/`1.68` for history. This contrasts
with r19's successful env0 (`0.8262 m`/481 frames) under the same nominal
contract. The fresh-process behavior spread is therefore itself larger than
the candidate comparison signal, and r27 is engineering evidence of an
unstable GPU contact execution distribution, not a candidate result. Its
packet and noise audit are in
`outputs/consequence-evaluator/gate1-gpu-group-engineering-20261009-r27/`.

The frozen physical-bank audit was extended to r27 with
`tools/audit/audit_gpu_group_gt_values.py`; output is
`outputs/consequence-evaluator/gate1-gpu-group-engineering-20261009-r28/gt-value-audit.json`.
Its four role values were `[0.016152, -0.000431, 0.020753, 0.036116]`, with
progress-start range `1.72e-6`. The raw argmax and frozen `0.01` selector both
chose the negative residual (`+0.019965` over baseline), whereas r19 and r21
both selected baseline after their zero-repeat role won the raw argmax. Because
all three packets violate the strict same-state contract, this choice flip is
evidence of execution sensitivity, not a GT-value or utility result.

## Same-process serial reset/replay follow-up

### Decision Note

The current decision was whether a single native GPU simulator could provide a
cheaper paired execution contract by resetting one environment in place and
replaying the recorded executed controls. The key evidence was that fresh
GPU-process group behavior varied from full grasp to near-zero lift, while the
strict same-state requirement rejected even small contact-stage drift. I ran one
bounded engineering Probe at seed 282 with five serial arms (baseline, two zero
repeats, and positive/negative residuals), keeping the original GPU PhysX/GPU
tensor pipeline and the 542-step behavior screen. A failure would close serial
reset as a useful engineering container; a successful behavior/zero-geometry
replay would retain it for field-specific noise calibration. No formal Gate1 or
external resource boundary was involved.

The implementation was run at commit `bf96546` (the current serial worker also
guards this mode to `gpu_physx_gpu_pipeline` in `11fd62d`). The raw packet is
`outputs/consequence-evaluator/gate1-gpu-serial-engineering-20261009-r29/serial.pkl`.
The derived, read-only diagnostics are
`serial-noise-audit.json` and `serial-gt-value-audit.json` in the same
directory. Both are engineering artifacts and are rejected by the strict score
worker.

The native GPU baseline reached `0.8241 m` maximum lift and held 484 frames;
both fixed-control zero repeats produced the same held/lift outcome and the
same action stream. Positive and negative residual arms reached `0.7971 m` /
485 frames and `0.6489 m` / 310 frames respectively. These outcome numbers are
behavior diagnostics only; all five arms failed the terminal settle criterion,
so they do not form a success claim. The reset setter did not advance the
absolute simulator frame (`0`, `1084`, `2168`, `3252`, `4336` before and after
the five resets), and the visible RNG streams were restored to the same reset
anchor for every arm.

The zero pair was exact over the recorded geometric execution fields: object
pose, hand points, surface/support gaps, table footprint, q position/q velocity,
and object velocity all had zero post-query p95 displacement. The exposed
contact buffers did not match: zero-pair post-query p95 was `15.82` for hand
contact force and `17.15` for object contact force. The task observation/history
buffers also differed immediately after reset (the first state trace difference
was stale `_curr_obs` at tick 0; history zero-pair p95 was `2.0`). The native
root/dof/rigid-body state views remained equal, but these measurements do not
cover PhysX warm-start, contact-manifold, or island caches. The baseline versus
zero RNG trace also includes the actor-inference RNG consumption; the two
fixed-control zero arms share the reset anchor and are the relevant pair.

The frozen physical-bank/TCC audit assigned identical `Y=0.0411695` to
baseline and both zero repeats. Positive and negative residuals scored
`-0.0145127` and `0.0044708`, respectively; the deadzone selector chose
baseline and the progress-start range was exactly zero. This is a useful
label-contract diagnostic, not a Gate1 utility result: contact/history state is
not a strict twin and only one serial launch was run.

The serial route is therefore retained as an engineering harness for frozen
control and field-specific noise measurements; the r29 single launch preserved
the native baseline behavior. It does not satisfy the strict same-state Gate1 contract and
cannot justify reactive-policy candidate ranking. The next route must either
fork/restore the hidden PhysX contact state or use a predeclared repeated
same-process statistical design that treats contact/history as noise; reference
bank, Y labels, evaluator training, PointWorld, and policy weights remain
unchanged.

### Arm-order follow-up (r30)

To test whether the serial result was tied to the warm-cache arm order, commit
`79cdbf3` added an explicit `candidate-first` schedule. A fresh process ran
`baseline → positive → negative → zero_repeat_1 → zero_repeat_2`, while the
saved packet remains in canonical role order and records `execution_order`.
The packet and read-only audits are in
`outputs/consequence-evaluator/gate1-gpu-serial-engineering-20261009-r30/`.

The comparison is not valid as a causal order test: the fresh r30 baseline did
not match r29's baseline despite identical seed, checkpoint, backend, and reset
contract. r29 reached `0.8241 m`/484 held frames; r30 reached `0.8281 m`/483,
with action and physical divergence before the candidate window. Thus any r29
versus r30 candidate/noise difference is confounded by the known fresh-process
GPU reactive baseline spread. Within r30, the two fixed-control zero arms still
had zero post-query p95 on the geometric fields, while contact/history remained
non-exact. Its frozen-bank values were baseline/zero
`0.0437091`, positive `0.0189845`, and negative `-0.0122399`, with zero
progress-start range and baseline selected; this remains an engineering label
diagnostic only.

The arm-order question is therefore `UNCLEAR`, and no further order-only runs
are planned without a frozen baseline control stream. Serial reset remains a
single-run frozen-control calibration container, not a fresh-process twin or a
formal Gate1 route.

## 8-env GPU group decision note

The next execution-contract probe uses the native `gpu_physx_gpu_pipeline` only. It
will run two fresh 72-step launches with eight environments: env0 is the nominal
baseline, env2/env3 receive the positive/negative residual, and the remaining
environments are zero roles. Actor inference remains on the legacy 256-row contract
by using 32 copies per environment, so group size and actor GEMM rows are not changed
together.

This is a decision probe, not a relaxed Gate1. A launch is useful only if its initial
semantic state is exact, all roles receive the same prefix controls and done flags,
and env0 reaches a clearly nontrivial query-window lift. If either launch collapses
near zero, group expansion stops and the route returns to serial frozen-control
calibration. Passing the short screen would justify four full542-step launch clusters;
it would still require launch-level paired statistics before candidate ranking. Hidden
PhysX state remains outside the public Isaac Gym API, so these packets cannot enter the
strict scorer.

### r31 short screen

The fixed-row implementation was run once at commit `01c085a` with seed282,
eight environments, 32 actor copies per environment, and a 72-step bounded
window. The initial semantic tensors were exact and every role received the
same prefix controls and done flags. That did not preserve the usable native
behavior: env0 reached only `0.1439 m` maximum lift and held 8 frames; the
selected zero env reached `0.1537 m`/8, while candidate roles were essentially
flat. Query-relative zero noise was already large (object pose displacement
p95 about `0.200 m`, q velocity p95 about `0.151`, history p95 `0.0231`).
The frozen TCC audit had progress-start range `1.86e-4` and selected baseline
after the deadzone. The packet and read-only audits are in
`outputs/consequence-evaluator/gate1-gpu-group-engineering-20261009-r31/`
and its `r31-audit` sibling.

This launch crossed the predeclared stop line (`tick72` lift below `0.20 m`),
so no second 8-env launch or 542-step group panel was run. The group expansion
is closed; the result is engineering evidence only and does not enter strict
Gate1, evaluator training, or PointWorld.

## Serial frozen-control cluster decision note

The first serial 72-step follow-up (r32) reached `0.2724 m` teacher lift, above
the `0.20 m` short-window screen. Its five-arm packet still showed exact
geometric zero-pair replay through the query window, while exposed contact
buffers and history diverged. This separates a useful frozen action stream from
the hidden-state problem, but does not establish same-state twins.

Commit `8f935b8` therefore adds one bounded cluster Probe. A reactive teacher
first produces the executed `[72,18]` stream. One process then resets and
replays the stream in the fixed order
`frozen_zero_1, positive_1, frozen_zero_2, negative_1, frozen_zero_3,
negative_2, frozen_zero_4, positive_2, frozen_zero_5`. Zero arms replay all
teacher actions; candidates use the clipped residual only at ticks 48--71.
The teacher is stored separately and is never treated as a counterfactual zero.
The packet and audit remain engineering-only; they report descriptive zero-pair
noise and candidate-versus-zero effects without treating arms as independent
samples or calling `choose_candidate`.

The hard stops are teacher lift below `0.20 m`, failed reset/action/provenance
contracts, mechanical zero noise above the existing field thresholds, or no
repeat-consistent candidate effect reaching roughly `2x` the zero floor on a
primary mechanical field. A pass would justify at most one additional launch
cluster for launch-level evidence; it would not open strict Gate1, evaluator
training, PointWorld, or a reference/Y/policy change.

### r33 result and bounded follow-up

The r33 launch ran at commit `e1c71b4` and passed the execution contract. The
teacher reached `0.2706 m` by tick72. Prefix/suffix controls, done flags,
requested residuals, teacher-action hash, fixed schedule, reset frame/RNG
anchors, and the visible reset contract all passed audit. Five frozen zeros had
zero query-relative p95 on object pose, hand geometry, gaps, q/dq, and object
velocity; contact/history remained a separate hidden-state noise channel.

Both positive repeats and both negative repeats were mechanically repeat-
consistent and separated from the zero floor. The physical-bank/TCC diagnostic
gave teacher/zero `0.0051948`, positive `-0.0117869`, and negative `-0.0129771`,
with zero progress-start range. This is an open-loop residual effect diagnostic;
it does not select a candidate or establish a reactive-policy GT utility.

Because the teacher and mechanical zero/effect screens passed, one fresh r34
cluster is predeclared to test launch-level repeatability. It keeps the exact
r33 schedule and 72-step limit. A failure closes the serial-statistical route;
a pass still remains engineering-only and cannot start evaluator, PointWorld, or
strict Gate1 work.

### r34 result and route closure

The fresh r34 launch passed the same packet and audit contracts. The teacher
reached `0.3001 m` by tick72 and held 13 frames; this clears the short-window
screen but is not a grasp-success result. All five frozen zeros again had zero
query-relative p95 on the primary mechanical fields. Contact buffers retained
hidden-state noise, while the two positive and two negative repeats reproduced
their respective mechanical effects. The descriptive TCC values were zero
`-0.0094461`, positive `-0.0117899`, and negative `-0.0131493`; neither
candidate exceeded the zero value.

The predeclared serial-statistical budget is now exhausted. The route is closed
as an engineering calibration: it demonstrates repeatable visible mechanical
replay for one frozen executed stream across two fresh launches, but it does not
restore PhysX hidden state or satisfy strict same-state Gate1. No evaluator,
PointWorld, Execution Bridge, or MPC work follows from r33/r34. The next route
must supply a new verifiable execution contract, such as an ACT/open-loop chunk
design or a native hidden-state fork/restore; Y, reference-bank labels, and
policy weights remain frozen.

## Deployment-conditioned proposal Probe

The existing clean hold-audit proposal has a deployment-distribution gap: when
its frozen checkpoint is evaluated on the r33/r34 reactive-teacher query
histories, 24-step chunk MSE is approximately `4.2e-4--6.3e-4`, compared with
`1.6e-4` on its clean holdout. The next bounded offline Probe uses the two
serial-cluster teachers as launch-level episodes and performs leave-one-launch-
out fitting with the same native 24-step decoder. The standardizer is fit only
on the training launch, and evaluation reports full-chunk, first-action, and
mean-baseline-relative error on the held-out launch.

This tests deployment coverage only. It does not restore hidden PhysX state,
run a new simulation, alter Y/reference/policy, or authorize evaluator,
PointWorld, or formal Gate1 work. A positive result justifies collecting more
reactive-deployment episodes; a negative result closes this proposal route until
a different execution contract is available.

## Clean-only source audit

The continuous manifest labels ordinary rollouts as `unlabeled`, so selecting
that quality alone can mix in intervention episodes. The implementation adds an
opt-in `clean_only` filter and verifies `assigned_phase=clean`,
`perturbation_tick=-1`, a zero finite residual plan, and fully-known plan
entries. On `continuous-20261008-{train,val,test}-r1`, filtering to
`expert=airplane_base` yields 8 clean episodes and 1204 windows per split; the
exact `s3_airplane_lift` motion contributes only 1 episode per split. The
manifest source entry for the frozen `GRAB_00000260.pth` also matches the SHA
used for the history RMS.

This is provenance evidence, not a behavior or utility result. It justifies one
small episode-held-out offline fit across the clean airplane-base episodes; it
does not justify route-specific generalization, a new simulator launch, or
evaluator/PointWorld/Gate1 work.
