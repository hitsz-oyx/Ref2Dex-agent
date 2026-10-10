---
schema: ref2dex.probe.v2
probe_id: P-20261010-generated-tau-native-execution
experiment_id: P-20261010-generated-tau-native-execution
date: 2026-10-10
task: consequence-evaluator
branch: main
git_commit: 3ed2404
claim_id: C3
hypothesis_family: HF-generated-tau-execution
probe_index_in_family: 1
seed_pool: probe
seeds: [298]
decision_changed_if_positive: collect real generated-candidate outcomes before scorer adaptation
decision_changed_if_negative: identify proposal or executor distribution mismatch before more scorer training
status: UNCLEAR
run_id: generated-tau-native-execution-20261010-r1
---

# Can pure measured-history generated tau drive the frozen native executor?

## Motivation / Decision Note

Blocker for ref8 H->candidate tau->Y->selector->tau*->A->Z: no generated tau
has actually been executed. Frozen score/geometry audit shows selected18/18
geometrically usable, but learned raw tau4/18; forecast-substitution failure
cannot label newly generated alternatives. Cheapest decision is one small
native execution wave, with frozen weights and repaired geometric outputs.
This serves the full chain; PointWorld/E necessity remain frozen. No new model,
seed sweep, core Mission/claim change, branch or strict same-state assertion.

## Protocol

Use existing owned CPU tensor exchange/GPU PhysX native single-motion task,
aligned frame0 reset and actual canonical fingerfit controller SHA79ee282e...
(ref7_4-tau-finger-fit-20261010-r3/final.pt). Fixed four roles,4env each16total,
randomized by seed298: tau_gt independent upper-bound, persistence,
learned displacement, scored ten-candidate pool (persistence+learned+K8).
Roles have separate live physical states; do not claim same-state causal utility.

At every8control steps, generated roles read only their measured t-3:t
hand/object poses, finger q/dq and object velocity. Bootstrap by repeating the
earliest measured state until four exist. No true future/actor observation,
clock/phase/action or tactile input to proposal/retrieval/choice. Retrieve only
train-bank displacements, one per episode; frozen T scores raw generated tau.
The runtime never admits a query-GT future into the ten-candidate pool.

Project the chosen future using live current hand/q and static URDF,300steps
(60step shortcut failed its geometry tolerance and will not be deployed),
two starts, independent native finger bounds/coupling. Use the resulting rigid
hand points and geometric q to condition the frozen897dim tau controller.
Next-frame tau-derived wrist velocity feedforward, original native residual
limits/clamp/coupling; no extra command clipping. Consume first8steps before
replanning;24future input is shifted and terminal-padded within the chunk.
Clock schedules replanning/independent GT comparison only, never model input.
All model weights frozen; no physical reset/fork at plan boundaries.

Full542steps, preserve all16rows including failures. Capture actual student
input, applied commands/PD targets, full measured trajectories, every plan's
pure history/raw tau/repaired points/q/score/choice/train donors. First8step
seed29 engineering smoke verifies wiring only. Compare3generated roles with
GT upper-bound descriptively, report held45, >=433held+terminal, clipping and
loss. Calibration requires GT>=3/4longheld+terminal. For each generated role,
original strong screen >=3/4longheld+terminal and clip<1% yields PROMISING;
otherwise local UNPROMISING if calibration passes, UNCLEAR if not. Scored
benefit remains UNCLEAR regardless, because roles are not matched forks.

## Resources / stop

One idleGPU2, new<=15GPUmin/1GiB. Smoke<=120s, one complete eval<=720s;
no training, automatic retries or extra seeds. Monitor GPU utilization/memory
and ETA at first step and128step intervals. Native projection deadline30s/plan;
global deadline and source hashes. Stop on nonfinite/identity/input/applied-
command mismatch or foreign GPU work; preserve failure artifacts. Tiny pure
runtime contract tests may use CPU where startup dominates fixture cost.

## Results

68e7ed3 seed29 engineering8-step smoke completed25.89s; own GPU7555MiB/util11%.
Independent reconstruction: bootstrap history error0, actual897features1.91e-6,
applied command2.38e-7. Correct own query history/choice/native q/FF; no GT leak.
Outputs `generated-tau-native-smoke-20261010-r1/`, log `generated-tau-native-launch-20261010-r1/smoke.log`.

First full seed298 r1 at a1eeccd was interrupted by root's SIGINT to its confirmed
owned PID, preserving FAILED/KeyboardInterrupt manifest and log. Engineering
cost:300-step projection14.57s/chunk predicts~1000s across68chunks, exceeding
720s cap. Not a behavior negative result; no complete trajectory produced.

## Engineering repair Decision Note

Latency blocks the cheapest complete execution. Check fixed60iteration fit on
the saved12query smoke inputs against300step geometry, with no actual outcome
selection. Require max coordinate-RMSE degradation<=.5mm, max point-coordinate
RMSE to300step<=1mm, latency<=5s. If this engineering check passes, run exactly
one repaired full seed298 r2 <=600s. No additional seed or model training; count
interrupted run and audit within the original15GPUmin total.60iteration always
outputs bounded coupled URDF geometry, though approximation quality may vary.
If convergence check fails, stop to identify the latency source. Original smoke,
failed full and300step geometry remain intact; do not rewrite them.

3ed2404 engineering latency check completed:60iterations3.95s versus300
14.57s; max target-coordinate RMSE degradation only.0174mm, but max fitted
point-coordinate difference2.398mm, exceeding the predeclared1mm bound.
Status FAIL; do not launch repaired r2 or relax the tolerance post hoc.
Outputs `generated-tau-projection-latency-20261010-r1/` (frozen smoke inputs,
source hashes/result/60step geometry). Native smoke25.89s+interrupted full
65.61s+latency check fit3.95s are within15GPUmin; no owned task process running.

First-query raw tau has17.00mm coordinate error for scored retrieval and
24.11mm for learned displacement even after300step projection, versus near
zero persistence. Old audit anchors were>=tick8; online t0 uses a padded
history outside the training-window protocol. This identifies a separate
startup-distribution limitation, not evidence of eventual holding success.
Do not silently hide it with a GT startup policy or actual future input.

Current native execution status UNCLEAR: engineering chain passes but full
holding outcomes unavailable. Next bounded action is to identify/accelerate
the existing300step implementation while preserving outputs (e.g. reusable
GPU computation capture), then complete the same frozen seed298 execution
within remaining declared resources. If that cannot fit, record a new bounded
Decision Note; no unbounded retry/sweep. Goal still covers the full ref8 chain
and eventual Mission utility, not merely an engineering smoke.

## Limitations / future evidence

## Continued engineering Decision Note

Previous goal turn made progress: actual pure-H native wiring was independently
verified, and the original300step latency/failed60step shortcut changed the next
action. Current HEAD3ec5236 and outputs confirm no owned live task. Keep full
Mission/ref8 scope and seed298; no convergence relaxation or narrower success.
First reproduce the 300step saved-smoke timing. Ranked falsifiable hypotheses:
1 unused link velocity computations dominate point-loss FK/backward (a pure
XYZ path should preserve positions/gradients and reduce cost);2 unrelated link
work (pruning only unobserved descendants should reduce cost);3 host/kernel
launch overhead (capture/reuse should reduce repeated cost). Test sequentially,
not together. New engineering checks count within remaining original15GPUmin;
same .5mm degradation/1mm fitted-point/5s gate, plus exact FK/gradient test.
If original300step is fast and equivalent, execute one full r2 <=600s on idleGPU2.
If not, diagnose the next specific source without another full expensive launch.
No external authorization boundary is changed.

## Limitations / future evidence

Single motion, few separate-live-role envs, observed-history training bank from
old actor+random residual distribution and oracle-trained executor warmstart.
Geometry repair can alter task intent; static reachability is not contact or
dynamic feasibility. This is not same-H generated-candidate ranking, Cm-on/off,
formal Validation, from-scratch learning or original placing task success.
