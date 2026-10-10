---
schema: ref2dex.probe.v2
probe_id: P-20261010-generated-tau-native-execution
experiment_id: P-20261010-generated-tau-native-execution
date: 2026-10-10
task: consequence-evaluator
branch: main
git_commit: ff65bb3
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

Project the chosen future using live current hand/q and static URDF,300steps,
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

Pending.

## Limitations / future evidence

Single motion, few separate-live-role envs, observed-history training bank from
old actor+random residual distribution and oracle-trained executor warmstart.
Geometry repair can alter task intent; static reachability is not contact or
dynamic feasibility. This is not same-H generated-candidate ranking, Cm-on/off,
formal Validation, from-scratch learning or original placing task success.
