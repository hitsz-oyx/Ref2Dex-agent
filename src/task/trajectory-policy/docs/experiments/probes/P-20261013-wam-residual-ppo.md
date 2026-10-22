---
schema: ref2dex.probe.v2
probe_id: P-20261013-wam-residual-ppo
experiment_id: P-20261013-wam-residual-ppo
date: 2026-10-13
task: trajectory-policy
branch: main
git_commit: ea2575aa
claim_id: C3
hypothesis_family: HF-wam-residual-ppo
probe_index_in_family: 1
seed_pool: probe
seeds: [292, 296]
decision_changed_if_positive: keep this frozen WAM+old-chunk pair and spend a longer residual-PPO budget on airplane
decision_changed_if_negative: stop this alpha and reward, and do not add a world critic on top of a residual that still throws
status: UNPROMISING
run_id: wam-ref2-execution-rppo-20261014-r7
---

# Residual PPO on the frozen context pair

Freeze the context actor and the old chunk R that reached a 0.91 mm gap and
then threw the airplane to 277 mm. PPO outputs only a 12-D native residual.
The executed command is clip(base + 0.15 * residual). WAM, R, and World stay
frozen. World does not enter the actor. The reward is object progress,
contact, hold, a residual penalty, and a slip penalty. It does not imitate
an expert action.

Training resets use demonstration frames 0, 32, 48, and 62, plus 0.01 rad
joint noise and 5 mm object noise. The reported gate is still one frame-0
rollout. A 64-step zero-residual rollout must reach a gap under 30 mm before
any PPO update.

## Decision Note

The question is whether a small residual, trained on task outcome from
phase-randomized resets, can keep the context pair's contact without the
throw. Six updates of 48 steps on four environments is a probe, not a
validation. Cost is one GPU and one server, about 3576 WAM queries, wall
clock under 35 minutes. Success is frame-0 hold of at least 45, or a higher
close-range object rise than 42.5 mm without leaving the table. A near-zero
residual that still throws is UNCLEAR. A clearly nonzero residual that still
throws is UNPROMISING for this alpha and reward. No R swap, no world critic,
and no BC-overfit track in this run.

`wam-ref2-execution-rppo-20261013-r1` confirmed the zero-residual base still
reaches a 0.60 mm gap, then the server aborted at demonstration phase 93
because 9 scene points fell outside the grid and the inference cap was 8.
The rerun keeps that cap at 8 for a normal episode and allows 64 dropped
points only for these open queries. A cloud that leaves the grid still
aborts.

## Completed result

`wam-ref2-execution-rppo-20261013-r2` finished 3576 queries on GPU0 in 295 s,
about 18.6 GB, utilization around 20%. The zero-residual diagnostic still
reached 0.60 mm. Six updates did not produce a hold. The sampled residual
stayed near its initial scale, about 0.14, and the frame-0 mean residual
was 0.012.

The four frame-0 copies started from the same state and the same first
command, then diverged. The best copy rose 17 mm while the gap was within
10 mm and stayed on the table. One copy left the table and finished at
-757 mm. Hold was 0 on every copy. Local label UNCLEAR: the mean residual
did not leave the frozen base, so this budget neither supports nor refutes
the reward. World stays out of the actor. R is unchanged.

## Long run

The six-update budget was too short to move the residual mean, and the first
KL reject left the learning rate at 7.5e-5 for every later update. The user
asked for a large PPO run on the same frozen pair. `wam-ref2-execution-rppo-20261013-r3`
uses 200 updates of 96 steps on 4 environments, about 77k training steps.
Each update starts again at learning rate 3e-4; a KL reject only shrinks
that update. Episodes are long enough for a frame-0 reset to reach the
known contact. One GPU, wall-clock cap 4 hours. WAM queries stay serial,
so this is the large run that fits in that cap, not millions of steps.
The frame-0 gate is unchanged. World and R stay frozen.

`wam-ref2-execution-rppo-20261013-r3` stopped in update 5. The phase-reset
function was never called, so physics continued from frame 0 and hit the end
of the clip. `r4` calls that reset at every update.

## Batched inference

`r4` was still querying one environment at a time. It was stopped during
update 70. The kept weights are `update0050.pt`; updates 51–70 were not
checkpointed. WAM weights are 52M parameters, 0.19 GiB. A forward benchmark
on GPU0 (`wam-ref2-execution-rppo-bench-20261013-r1`) compared the same four
scenes:

| mode | group wall | samples/s | peak GPU |
|---|---:|---:|---:|
| serial-4 | 262 ms | 15.2 | 555 MiB |
| batch-2×2 | 162 ms | 24.7 | 557 MiB |
| batch-4 | 92 ms | 43.7 | 557 MiB |
| batch-8 | 113 ms | 71.1 | 581 MiB |
| batch-16 | 161 ms | 99.5 | 601 MiB |
| batch-32 | 252 ms | 127 | 661 MiB |

Batch 4 is 2.9× the serial-4 rate. Batch 32 did not run out of memory.
Against the matching serial rows, the largest trajectory difference was
1.46 mm and the largest base-action difference was 5.2e-4. That is bf16
batch noise, not a swapped row. `r5` therefore uses one batch-8 WAM+R
call per physics step, eight environments, and resumes from update 50 for
150 further updates. The frame-0 zero-residual gate is unchanged. World
and R weights stay frozen.

`r5` passed that gate: the zero-residual diagnostic gap was 0.42 mm, and
the server base action matched native R within 4.7e-5. One control step
is about 0.17 s, against about 0.27 s for the old 4-env serial step, so
environment steps per second are about 3×. GPU0 stays near 18.7 GB:
Isaac about 18034 MiB, the WAM server about 602 MiB. The earlier 18.6 GB
reading was this Isaac allocation, not WAM weights. Utilization during
the step is about 24%. Batch 16 and 32 still fit in the WAM process;
they were not used for this run.

## Completed long run

`r5` finished 200 updates on GPU0 in 2562 s. The frame-0 evaluation is
UNPROMISING for this alpha and reward. Eval residual rms is 0.136. The best
environment held 5 frames, rose 93 mm while close, then left the table
(terminal −765 mm). Other copies stayed on the table with almost no lift.
Hold 45 did not occur.

Reward and the hold fraction were flat from update 51 through 200: hold
stayed near 0.006–0.010 of each rollout, mean reward between −0.03 and
−0.01. Residual rms crept from 0.155 to 0.209, so the weights were still
moving, but the task outcome was not. Another block of the same updates is
not the next step.

## 2026-10-14 ref6 implementation correction

Historical r2–r5 traces remain evidence of their executed behavior, but their
negative results cannot close the intended residual method: truncated rollout
bootstrap was zero, intermediate resets fabricated H4 and previous command,
and lift/held reward state used the reset height. A regression reproduced LR
attempts [3e-4, 7.5e-5, 7.5e-5, 7.5e-5]; corrected backtracking progressively
quarters each attempted rate and restores both weights and Adam moments.
λ defaults to 1, following the existing long-credit mechanism evidence.

Decision Note: this is a Blocker correctness repair for the Task's autonomous
grasp baseline and global Objective A. First establish that measured prefix
context and true terminal bootstrap reach the implemented PPO path; then ask
whether a fresh 20-update probe produces hold beyond the old floor. The
cheapest useful check is tiny deterministic contract tests plus one 4-env GPU
run, rather than another 200-update run. WAM, World, old Chunk-R, alpha=.15
and reward coefficients stay frozen. Phase snapshots now come from a bounded
62-control teacher prefix, with matching measured past history/actions; this
changes the reset physical source as well as its policy context, so r5/r6
is not a single-variable causal comparison. Training perturbs only the reset
frame; diagnostic resets have no perturbation. Held initialization uses the
current measured gap/support and task-start lift; there is no expert command
inside the training/evaluation rollout.

Planned run `wam-ref2-execution-rppo-20261014-r6`: fresh initialization,
GPU6 if free, phases 0/32/48/62, 96 diagnostic controls per phase in one
batch, 20×96 training controls plus 20 actual post-control bootstrap queries,
542 frame-0 eval controls. Total 2578 RPCs / 10312 model-query samples,
plus 62 teacher prefix controls. One GPU, expected 5–10 minutes, hard wall
cap 4 hours, outputs under 100 MB. Abort on any invalid execution/context,
nonfinite value, unexpected early done, foreign GPU process or any phase
with no gap under 30 mm. A successful engineering gate permits only this
small probe; no permission/resource expansion is needed.

Success/failure: retain frame-0 hold45/close-rise criteria as a Probe screen;
report training held/slip, residual mean/std, per-DOF saturation/margins,
per-phase returns and value explained variance. If hold remains near 1%,
inspect these diagnostics before spending more updates; stop this block.
No formal RL/WAM claim follows from this rerun. The old failed implementation
is not negative evidence for a clean residual method.

v2 checkpoints include optimizer state, update, all relevant RNG streams and
the measured teacher prefix. v1 cannot be resumed as a strict continuation.
GPU simulator contact caches are not serialized; resumed reset rollouts may
still differ physically. Tiny tensor tests use CPU because they finish in
seconds and GPU startup would dominate; model training/physics use GPU.

Limitations/future evidence: exact frozen-pair fixed-residual basin screening,
joint-wise feasible authority, observation normalization and matched multi-seed
validation remain deferred. Gaussian-latent PPO remains mathematically defined
through the deterministic clipping map; saturation can still hide useful
exploration and is now explicitly measured. Do not transfer the old tau
thumb-pitch basin to this WAM+R pair without testing it.

## Completed corrected short probe

Fix commits: `b201396b` (context/reward/bootstrap/λ/KL), `a2f85d4b`
(legacy teacher batch initialization). The first attempt `r6` stopped before
any WAM rollout or PPO update: the legacy player's batch marker had not been
initialized, flattening a 64×1442 teacher observation. Its FAILED manifest
is retained. `r7` ran the corrected implementation at `a2f85d4b`, fresh
actor/value, on GPU6; 20 updates and frame-0 evaluation completed in 492 s
after native/server readiness. GPU memory was 18832 MiB (18.4 GiB),
utilization usually 17–34%. Output size 36 MiB; global outputs+tmp
approximately 293 GiB, within the 400 GB budget.

All four 96-control zero-residual phase diagnostics passed the <30 mm gate:

| phase | minimum gap (mm) | held fraction |
| --- | ---: | ---: |
| 0 | 0.423 | 0 |
| 32 | 0.598 | 0 |
| 48 | 0.000 | 0.396 |
| 62 | 1.563 | 1.000 |

The phase62 reset was already held, 92.7 mm above task-start height.
Its restored previous_held is true; previous-action RMS at phases32/48/62
is .333/.206/.277, rather than zero. The phase0 value is correctly zero.
These actual reset measurements validate why the prior reset-local lift and
cleared reward state were semantically wrong.

Training mean held fraction was .2346 (per-update range .1146–.3672), slip
fraction .00208, and average active-DOF clipping .00289. Last-update value
explained variance was .3459. High held fractions predominantly come from
the already-grasped phase62 reset; they are not startup grasp progress.
The final deterministic frame0 residual RMS was .0706, with held0 on all
four environments, maximum object rise <=1.53 mm. All four objects remained
on the table; no throw/drop was detected. Under the retained Probe screen,
this fresh 20-update recipe is locally UNPROMISING for startup grasp. This
does not refute residual RL or attribute failure to WAM alone, and it is not
a controlled causal improvement over r5.

`tools/audit/audit_wam_residual_ppo.py --run outputs/trajectory-policy/wam-ref2-execution-rppo-20261014-r7 --device cuda:0`
was run with CUDA_VISIBLE_DEVICES=6 after the training processes exited.
It independently checked 22 reset contexts, all four zero-residual phase
traces, task-start lift/initial-held state, all 20 actual final observations
(NumPy reconstruction), and all 20 saved pre-update value checkpoints.
Bootstrap maximum discrepancy was 1.91e-6; all training endpoints are
nonterminal. Both Adam states, update count, phase layout and all five RNG
streams are present in final.pt. This is an engineering audit, not scientific
validation or a bitwise physics-resume claim. Related contract tests: 17 pass.

Artifacts: `outputs/trajectory-policy/wam-ref2-execution-rppo-20261014-r7/`
contains manifest/result/audit, reset-prefix and reset captures, four-phase
diagnostic, per-update bootstrap input/value, full evaluation and v2
checkpoints. Raw native/server output is also preserved in
`tmp/wam-ref6-monitor/r7.log` and the run's server.log.

Next decision: stop this block at 20 updates. Before a longer PPO block,
screen small fixed residual directions on this exact frozen WAM+R pair from
frame0 and determine whether startup has a usable basin. Current measured
saturation is low on average, so do not assume clipping alone is the blocker.
No further experiment was started in this bug-fix task.

## 2026-10-22 负结果 CPU 审计：结论范围修订

r2–r5受bootstrap、reset H4/previous、lift基准与optimizer/KL回退等实现问题污染，移出清洁方法负证据；原轨迹可保留为故障现象。r6初始化前失败不算训练。r7真正执行版本a2f85d4b的20更新frame0无持握可以保留为局部短预算结果；phase62已持握reset主导训练held，不能当作起抓收益。修复也改变初始化分布，不把r5→r7当单变量因果对照。不是WAM v6/PPO路线的普遍否决。

本追加不改变原始数字、gate或运行产物；完整身份、版本边界与排除项见[逐卡证据审计](../../../../../../src/task/ptv3-shared-wam-v6/docs/audit/negative-result-card-audit-20261022.md)。
