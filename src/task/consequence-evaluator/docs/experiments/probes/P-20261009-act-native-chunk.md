---
schema: ref2dex.probe.v2
probe_id: P-20261009-act-native-chunk
experiment_id: P-20261009-act-native-chunk
date: 2026-10-09
task: consequence-evaluator
branch: main
git_commit: c45d4a4
claim_id: C3
hypothesis_family: HF-consequence-act-proposal
probe_index_in_family: 1
seed_pool: probe
seeds: [282]
decision_changed_if_positive: run a native GPU open-loop chunk behavior screen before returning to GT candidate ranking
decision_changed_if_negative: keep the reactive policy as the only behavior baseline and defer PointWorld/chunk candidates
status: UNCLEAR
run_id: gate1-gpu-group-engineering-20261009-r27
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
