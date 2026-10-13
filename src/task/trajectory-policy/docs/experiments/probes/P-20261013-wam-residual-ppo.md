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
run_id: wam-ref2-execution-rppo-20261013-r5
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
